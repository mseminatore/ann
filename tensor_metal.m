/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

//================================================================================================
// APPLE METAL GPU INFERENCE BACKEND
//================================================================================================
// Provides GPU-accelerated forward pass using Apple Metal and Metal Performance Shaders (MPS).
//
// Usage pattern:
//   1. ann_gpu_init()                  - initialize device + pipeline
//   2. ann_gpu_upload_network(pnet)    - upload weights/biases to GPU once
//   3. ann_predict(pnet, inputs, out)  - inference dispatches to GPU automatically
//      OR ann_predict_batch(...)       - batch inference (recommended for throughput)
//   4. ann_gpu_free_network(pnet)      - release GPU buffers
//
// Architecture:
//   - MPSMatrixMultiplication: Y = X * W^T for batched forward pass
//   - MPSMatrixVectorMultiplication: y = W * x for single-sample inference
//   - Custom Metal kernels (shaders.metal): activations, bias broadcast, softmax
//================================================================================================

#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdio.h>
#include <string.h>
#include "tensor.h"
#include "ann.h"
#include "ann_gpu_backend.h"

//-----------------------------------------------------------
// Internal GPU context (singleton)
//-----------------------------------------------------------
typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    // Inference kernels
    id<MTLComputePipelineState> pso_sigmoid;
    id<MTLComputePipelineState> pso_relu;
    id<MTLComputePipelineState> pso_leaky_relu;
    id<MTLComputePipelineState> pso_tanh;
    id<MTLComputePipelineState> pso_softsign;
    id<MTLComputePipelineState> pso_gelu;
    id<MTLComputePipelineState> pso_softmax;
    id<MTLComputePipelineState> pso_bias_add;
    // Training: activation derivative kernels
    id<MTLComputePipelineState> pso_deriv_sigmoid;
    id<MTLComputePipelineState> pso_deriv_relu;
    id<MTLComputePipelineState> pso_deriv_leaky_relu;
    id<MTLComputePipelineState> pso_deriv_tanh;
    id<MTLComputePipelineState> pso_deriv_softsign;
    id<MTLComputePipelineState> pso_deriv_gelu;
    // Training: gradient ops
    id<MTLComputePipelineState> pso_bias_grad_sum;
    id<MTLComputePipelineState> pso_gradient_clip;
    // Training: optimizer kernels
    id<MTLComputePipelineState> pso_sgd_update;
    id<MTLComputePipelineState> pso_momentum_update;
    id<MTLComputePipelineState> pso_adagrad_update;
    id<MTLComputePipelineState> pso_rmsprop_update;
    id<MTLComputePipelineState> pso_adam_update;
    // Training: regularization
    id<MTLComputePipelineState> pso_l2_regularize;
    id<MTLComputePipelineState> pso_l1_regularize;
    // Training: dropout
    id<MTLComputePipelineState> pso_dropout_forward;
    id<MTLComputePipelineState> pso_dropout_backward;
    // Training: output layer gradient and loss
    id<MTLComputePipelineState> pso_output_delta_loss;
    id<MTLComputePipelineState> pso_softmax_jacobian;
    // MPS GEMM kernels by shape, reused across batches
    NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *gemm_cache;
    // Per-batch targets and per-row losses (shared memory)
    id<MTLBuffer> targets_buf;
    id<MTLBuffer> row_loss_buf;
    int initialized;
    int training_buffers_ready;   // 1 once gpu_alloc_training_buffers() succeeds
    int training_batch_size;      // batch size for which training buffers are allocated
} MetalContext;

static MetalContext g_metal = {0};

//-----------------------------------------------------------
// Helper: create a compute pipeline state from a kernel name
//-----------------------------------------------------------
static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, const char *name)
{
    NSString *ns_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> fn = [lib newFunctionWithName:ns_name];
    if (!fn) {
        fprintf(stderr, "[Metal] Kernel '%s' not found in shader library\n", name);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [g_metal.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[Metal] Failed to create PSO for '%s': %s\n", name,
                [[err localizedDescription] UTF8String]);
    }
    return pso;
}

//-----------------------------------------------------------
// Initialize the Metal GPU context.
// Must be called once before any GPU operations.
// Returns 1 on success, 0 on failure.
//-----------------------------------------------------------
int tensor_metal_init(void)
{
    if (g_metal.initialized)
        return 1;

    g_metal.device = MTLCreateSystemDefaultDevice();
    if (!g_metal.device) {
        fprintf(stderr, "[Metal] No Metal-capable GPU found\n");
        return 0;
    }

    g_metal.queue = [g_metal.device newCommandQueue];
    if (!g_metal.queue) {
        fprintf(stderr, "[Metal] Failed to create command queue\n");
        return 0;
    }

    // Compile the shader source at runtime — works without full Xcode.app
    // (Command Line Tools only is sufficient)
    static const char *kShaderSource =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void bias_add(\n"
        "    device float *buf [[buffer(0)]],\n"
        "    device const float *bias [[buffer(1)]],\n"
        "    uint2 gid [[thread_position_in_grid]],\n"
        "    uint2 dims [[threads_per_grid]]) {\n"
        "    uint b = gid.y; uint n = gid.x;\n"
        "    buf[b * dims.x + n] += bias[n]; }\n"
        "kernel void activation_sigmoid(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    buf[gid] = 1.0f / (1.0f + exp(-buf[gid])); }\n"
        "kernel void activation_relu(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    buf[gid] = max(0.0f, buf[gid]); }\n"
        "kernel void activation_leaky_relu(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    float x = buf[gid]; buf[gid] = x > 0.0f ? x : 0.01f * x; }\n"
        "kernel void activation_tanh(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    buf[gid] = tanh(buf[gid]); }\n"
        "kernel void activation_softsign(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    float x = buf[gid]; buf[gid] = x / (1.0f + fabs(x)); }\n"
        "kernel void activation_gelu(\n"
        "    device float *buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    float x = buf[gid];\n"
        "    buf[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }\n"
        "kernel void softmax_rows(\n"
        "    device float *buf [[buffer(0)]],\n"
        "    constant uint &cols [[buffer(1)]],\n"
        "    uint bid [[threadgroup_position_in_grid]]) {\n"
        "    uint offset = bid * cols;\n"
        "    float mx = -INFINITY;\n"
        "    for (uint n = 0; n < cols; n++) mx = max(mx, buf[offset + n]);\n"
        "    float s = 0.0f;\n"
        "    for (uint n = 0; n < cols; n++) {\n"
        "        float e = exp(buf[offset + n] - mx);\n"
        "        buf[offset + n] = e; s += e; }\n"
        "    if (s > 0.0f)\n"
        "        for (uint n = 0; n < cols; n++) buf[offset + n] /= s; }\n"
        // ----------------------------------------------------------------
        // Activation derivative kernels (training backward pass)
        // buf0 = dl_dz [batch*nodes], buf1 = activations or pre-act z [batch*nodes]
        // Each kernel multiplies dl_dz[i] by the local derivative.
        // ----------------------------------------------------------------
        "kernel void deriv_sigmoid(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float ai = a[gid]; dl_dz[gid] *= ai * (1.0f - ai); }\n"
        "kernel void deriv_relu(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *z [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    dl_dz[gid] *= (z[gid] > 0.0f ? 1.0f : 0.0f); }\n"
        "kernel void deriv_leaky_relu(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *z [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    dl_dz[gid] *= (z[gid] > 0.0f ? 1.0f : 0.01f); }\n"
        "kernel void deriv_tanh(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float ai = a[gid]; dl_dz[gid] *= (1.0f - ai * ai); }\n"
        "kernel void deriv_softsign(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float d = 1.0f - fabs(a[gid]); dl_dz[gid] *= d * d; }\n"
        "kernel void deriv_gelu(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *z [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float x = z[gid];\n"
        "    float t = tanh(0.7978845608f * (x + 0.044715f * x * x * x));\n"
        "    dl_dz[gid] *= 0.5f * (1.0f + t) +\n"
        "        0.5f * x * (1.0f - t * t) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x); }\n"
        // ----------------------------------------------------------------
        // bias_grad_sum: column-reduce [batch x out] -> [out]
        // Each thread handles one output column.
        // ----------------------------------------------------------------
        "kernel void bias_grad_sum(\n"
        "    device const float *dl_dz [[buffer(0)]],\n"
        "    device float *bias_grad [[buffer(1)]],\n"
        "    constant uint &batch_size [[buffer(2)]],\n"
        "    constant uint &out_nodes [[buffer(3)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid >= out_nodes) return;\n"
        "    float s = 0.0f;\n"
        "    for (uint b = 0; b < batch_size; b++) s += dl_dz[b * out_nodes + gid];\n"
        "    bias_grad[gid] += s; }\n"
        // ----------------------------------------------------------------
        // gradient_clip: clamp each element to [-max_grad, max_grad]
        // ----------------------------------------------------------------
        "kernel void gradient_clip(\n"
        "    device float *g [[buffer(0)]],\n"
        "    constant float &max_grad [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    g[gid] = clamp(g[gid], -max_grad, max_grad); }\n"
        // ----------------------------------------------------------------
        // Optimizer update kernels
        // All take: buf0=weights, buf1=gradients, buf2=params (constants)
        // ----------------------------------------------------------------
        "kernel void sgd_update(\n"
        "    device float *w [[buffer(0)]],\n"
        "    device const float *g [[buffer(1)]],\n"
        "    constant float &lr [[buffer(2)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    w[gid] += lr * g[gid]; }\n"
        "kernel void momentum_update(\n"
        "    device float *w [[buffer(0)]],\n"
        "    device const float *g [[buffer(1)]],\n"
        "    device float *m [[buffer(2)]],\n"
        "    constant float &lr [[buffer(3)]],\n"
        "    constant float &beta [[buffer(4)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float mi = beta * m[gid] + (1.0f - beta) * g[gid];\n"
        "    m[gid] = mi; w[gid] += lr * mi; }\n"
        "kernel void adagrad_update(\n"
        "    device float *w [[buffer(0)]],\n"
        "    device const float *g [[buffer(1)]],\n"
        "    device float *v [[buffer(2)]],\n"
        "    constant float &lr [[buffer(3)]],\n"
        "    constant float &eps [[buffer(4)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float gi = g[gid]; v[gid] += gi * gi;\n"
        "    w[gid] += lr * gi / (sqrt(v[gid]) + eps); }\n"
        "kernel void rmsprop_update(\n"
        "    device float *w [[buffer(0)]],\n"
        "    device const float *g [[buffer(1)]],\n"
        "    device float *v [[buffer(2)]],\n"
        "    constant float &lr [[buffer(3)]],\n"
        "    constant float &beta [[buffer(4)]],\n"
        "    constant float &eps [[buffer(5)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float gi = g[gid];\n"
        "    v[gid] = beta * v[gid] + (1.0f - beta) * gi * gi;\n"
        "    w[gid] += lr * gi / (sqrt(v[gid]) + eps); }\n"
        "kernel void adam_update(\n"
        "    device float *w [[buffer(0)]],\n"
        "    device const float *g [[buffer(1)]],\n"
        "    device float *m [[buffer(2)]],\n"
        "    device float *v [[buffer(3)]],\n"
        "    constant float &lr [[buffer(4)]],\n"
        "    constant float &beta1 [[buffer(5)]],\n"
        "    constant float &beta2 [[buffer(6)]],\n"
        "    constant float &eps [[buffer(7)]],\n"
        "    constant float &bc1 [[buffer(8)]],\n"
        "    constant float &bc2 [[buffer(9)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float gi = g[gid];\n"
        "    float mi = beta1 * m[gid] + (1.0f - beta1) * gi;\n"
        "    float vi = beta2 * v[gid] + (1.0f - beta2) * gi * gi;\n"
        "    m[gid] = mi; v[gid] = vi;\n"
        "    float mhat = mi * bc1; float vhat = vi * bc2;\n"
        "    w[gid] += lr * mhat / (sqrt(vhat) + eps); }\n"
        // ----------------------------------------------------------------
        // Regularization kernels
        // ----------------------------------------------------------------
        "kernel void l2_regularize(\n"
        "    device float *w [[buffer(0)]],\n"
        "    constant float &decay [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    w[gid] *= decay; }\n"
        "kernel void l1_regularize(\n"
        "    device float *w [[buffer(0)]],\n"
        "    constant float &delta [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float wi = w[gid];\n"
        "    if (wi > 0.0f) w[gid] -= delta;\n"
        "    else if (wi < 0.0f) w[gid] += delta; }\n"
        // ----------------------------------------------------------------
        // Dropout kernels (mask holds 0 or 1/(1-rate), drawn on the CPU)
        // ----------------------------------------------------------------
        "kernel void dropout_forward(\n"
        "    device float *a [[buffer(0)]],\n"
        "    device const float *mask [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    a[gid] *= mask[gid]; }\n"
        "kernel void dropout_backward(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device float *a [[buffer(1)]],\n"
        "    device const float *mask [[buffer(2)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float m = mask[gid];\n"
        "    dl_dz[gid] *= m;\n"
        "    if (m != 0.0f) a[gid] /= m; }\n"
        // ----------------------------------------------------------------
        // Output layer: delta = t - y and each row's loss term, one thread
        // per row. mode 0 = MSE (squared error), 1 = categorical cross-entropy,
        // 2 = binary cross-entropy (sigmoid outputs); y is clamped like
        // ann_cross_entropy_term().
        // ----------------------------------------------------------------
        "kernel void output_delta_loss(\n"
        "    device const float *y [[buffer(0)]],\n"
        "    device const float *t [[buffer(1)]],\n"
        "    device float *delta [[buffer(2)]],\n"
        "    device float *row_loss [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]],\n"
        "    constant uint &mode [[buffer(5)]],\n"
        "    uint row [[thread_position_in_grid]]) {\n"
        "    float s = 0.0f;\n"
        "    for (uint n = 0; n < cols; n++) {\n"
        "        uint i = row * cols + n;\n"
        "        float yi = y[i], ti = t[i];\n"
        "        delta[i] = ti - yi;\n"
        "        if (mode == 0) { float d = yi - ti; s += d * d; }\n"
        "        else {\n"
        "            float c = clamp(yi, 1e-7f, 1.0f - 1e-7f);\n"
        "            s += mode == 2 ? -(ti * log(c) + (1.0f - ti) * log(1.0f - c)) : -ti * log(c);\n"
        "        } }\n"
        "    row_loss[row] = s; }\n"
        // ----------------------------------------------------------------
        // MSE through a softmax output: d = y * (d - sum(d * y)) per row
        // ----------------------------------------------------------------
        "kernel void softmax_jacobian(\n"
        "    device float *d [[buffer(0)]],\n"
        "    device const float *y [[buffer(1)]],\n"
        "    constant uint &cols [[buffer(2)]],\n"
        "    uint row [[thread_position_in_grid]]) {\n"
        "    uint o = row * cols; float dot = 0.0f;\n"
        "    for (uint n = 0; n < cols; n++) dot += d[o + n] * y[o + n];\n"
        "    for (uint n = 0; n < cols; n++) d[o + n] = y[o + n] * (d[o + n] - dot); }\n";

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSource];
    g_metal.library = [g_metal.device newLibraryWithSource:src options:nil error:&err];
    if (!g_metal.library) {
        fprintf(stderr, "[Metal] Failed to compile shader source: %s\n",
                [[err localizedDescription] UTF8String]);
        return 0;
    }

    // Build compute pipeline states for all activation kernels
    g_metal.pso_sigmoid    = make_pso(g_metal.library, "activation_sigmoid");
    g_metal.pso_relu       = make_pso(g_metal.library, "activation_relu");
    g_metal.pso_leaky_relu = make_pso(g_metal.library, "activation_leaky_relu");
    g_metal.pso_tanh       = make_pso(g_metal.library, "activation_tanh");
    g_metal.pso_softsign   = make_pso(g_metal.library, "activation_softsign");
    g_metal.pso_gelu       = make_pso(g_metal.library, "activation_gelu");
    g_metal.pso_softmax    = make_pso(g_metal.library, "softmax_rows");
    g_metal.pso_bias_add   = make_pso(g_metal.library, "bias_add");

    if (!g_metal.pso_sigmoid || !g_metal.pso_relu || !g_metal.pso_leaky_relu ||
        !g_metal.pso_tanh    || !g_metal.pso_softsign || !g_metal.pso_softmax ||
        !g_metal.pso_gelu    || !g_metal.pso_bias_add)
        return 0;

    // Build PSOs for training kernels
    g_metal.pso_deriv_sigmoid    = make_pso(g_metal.library, "deriv_sigmoid");
    g_metal.pso_deriv_relu       = make_pso(g_metal.library, "deriv_relu");
    g_metal.pso_deriv_leaky_relu = make_pso(g_metal.library, "deriv_leaky_relu");
    g_metal.pso_deriv_tanh       = make_pso(g_metal.library, "deriv_tanh");
    g_metal.pso_deriv_softsign   = make_pso(g_metal.library, "deriv_softsign");
    g_metal.pso_deriv_gelu       = make_pso(g_metal.library, "deriv_gelu");
    g_metal.pso_bias_grad_sum    = make_pso(g_metal.library, "bias_grad_sum");
    g_metal.pso_gradient_clip    = make_pso(g_metal.library, "gradient_clip");
    g_metal.pso_sgd_update       = make_pso(g_metal.library, "sgd_update");
    g_metal.pso_momentum_update  = make_pso(g_metal.library, "momentum_update");
    g_metal.pso_adagrad_update   = make_pso(g_metal.library, "adagrad_update");
    g_metal.pso_rmsprop_update   = make_pso(g_metal.library, "rmsprop_update");
    g_metal.pso_adam_update      = make_pso(g_metal.library, "adam_update");
    g_metal.pso_l2_regularize    = make_pso(g_metal.library, "l2_regularize");
    g_metal.pso_l1_regularize    = make_pso(g_metal.library, "l1_regularize");
    g_metal.pso_dropout_forward  = make_pso(g_metal.library, "dropout_forward");
    g_metal.pso_dropout_backward = make_pso(g_metal.library, "dropout_backward");
    g_metal.pso_output_delta_loss = make_pso(g_metal.library, "output_delta_loss");
    g_metal.pso_softmax_jacobian = make_pso(g_metal.library, "softmax_jacobian");

    if (!g_metal.pso_deriv_sigmoid || !g_metal.pso_deriv_relu   ||
        !g_metal.pso_deriv_leaky_relu || !g_metal.pso_deriv_tanh ||
        !g_metal.pso_deriv_softsign || !g_metal.pso_deriv_gelu ||
        !g_metal.pso_bias_grad_sum  ||
        !g_metal.pso_gradient_clip  || !g_metal.pso_sgd_update   ||
        !g_metal.pso_momentum_update || !g_metal.pso_adagrad_update ||
        !g_metal.pso_rmsprop_update || !g_metal.pso_adam_update  ||
        !g_metal.pso_l2_regularize  || !g_metal.pso_l1_regularize ||
        !g_metal.pso_dropout_forward || !g_metal.pso_dropout_backward ||
        !g_metal.pso_output_delta_loss || !g_metal.pso_softmax_jacobian)
        return 0;

    g_metal.gemm_cache = [NSMutableDictionary dictionary];
    g_metal.initialized = 1;
    return 1;
}

//-----------------------------------------------------------
// Release a GPU buffer (MTLBuffer*).
// Called from tensor_free() via forward declaration.
//-----------------------------------------------------------
void tensor_metal_release_buffer(void *gpu_buf)
{
    if (gpu_buf)
    {
        // Hand the retained reference back to ARC, which releases it
        CFBridgingRelease(gpu_buf);
    }
}

//-----------------------------------------------------------
// Upload a CPU tensor to GPU (allocate MTLBuffer + blit).
// After this call, t->gpu_buf is a valid MTLBuffer*.
// Returns 1 on success, 0 on failure.
//-----------------------------------------------------------
int tensor_metal_upload(PTensor t)
{
    if (!t || !g_metal.initialized)
        return 0;

    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);

    id<MTLBuffer> buf = [g_metal.device newBufferWithBytes:t->values
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[Metal] Failed to allocate MTLBuffer (%zu bytes)\n", bytes);
        return 0;
    }

    // Transfer Objective-C ownership to a raw void* (retained)
    if (t->gpu_buf)
        tensor_metal_release_buffer(t->gpu_buf);

    t->gpu_buf = (__bridge_retained void *)buf;
    return 1;
}

//-----------------------------------------------------------
// Download GPU tensor data back to CPU (blit GPU→CPU).
// Requires the tensor was previously uploaded with tensor_metal_upload().
// Returns 1 on success, 0 on failure.
//-----------------------------------------------------------
int tensor_metal_download(PTensor t)
{
    if (!t || !t->gpu_buf || !g_metal.initialized)
        return 0;

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t->gpu_buf;
    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);

    // For shared-mode MTLBuffers the GPU writes are visible after command completion;
    // just copy the contents back.
    memcpy(t->values, [buf contents], bytes);
    return 1;
}

//-----------------------------------------------------------
// Dispatch an element-wise activation kernel over a buffer.
// pso: the compute pipeline state for the activation kernel
// count: total number of elements (rows * cols)
//-----------------------------------------------------------
static void dispatch_activation(id<MTLCommandBuffer> cmd,
                                id<MTLComputePipelineState> pso,
                                id<MTLBuffer> buf,
                                NSUInteger count)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];

    NSUInteger tgsize = pso.maxTotalThreadsPerThreadgroup;
    if (tgsize > count) tgsize = count;

    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgsize, 1, 1)];
    [enc endEncoding];
}

//-----------------------------------------------------------
// Dispatch bias_add kernel: buf[b*cols + n] += bias[n]
// buf_buf: [batch x cols] output matrix
// bias_buf: [cols] bias vector
//-----------------------------------------------------------
static void dispatch_bias_add(id<MTLCommandBuffer> cmd,
                              id<MTLBuffer> buf_buf,
                              id<MTLBuffer> bias_buf,
                              NSUInteger batch_size,
                              NSUInteger cols)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_metal.pso_bias_add];
    [enc setBuffer:buf_buf  offset:0 atIndex:0];
    [enc setBuffer:bias_buf offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(cols, batch_size, 1);
    NSUInteger tgw = g_metal.pso_bias_add.maxTotalThreadsPerThreadgroup;
    if (tgw > cols) tgw = cols;
    MTLSize tgSize = MTLSizeMake(tgw, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
}

//-----------------------------------------------------------
// Dispatch softmax_rows kernel: per-row softmax over batch
// buf_buf: [batch x cols] matrix
// cols: number of output nodes
// batch_size: number of rows
//-----------------------------------------------------------
static void dispatch_softmax(id<MTLCommandBuffer> cmd,
                             id<MTLBuffer> buf_buf,
                             uint32_t cols,
                             NSUInteger batch_size)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_metal.pso_softmax];
    [enc setBuffer:buf_buf offset:0 atIndex:0];
    [enc setBytes:&cols length:sizeof(uint32_t) atIndex:1];

    // One thread per row — each thread sequentially computes softmax for its row.
    // This is correct for any output size and avoids parallel reduction complexity.
    [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
}

//-----------------------------------------------------------
// Select the activation PSO for a given activation type.
//-----------------------------------------------------------
static id<MTLComputePipelineState> pso_for_activation(Activation_type act)
{
    switch (act) {
        case ACTIVATION_SIGMOID:    return g_metal.pso_sigmoid;
        case ACTIVATION_RELU:       return g_metal.pso_relu;
        case ACTIVATION_LEAKY_RELU: return g_metal.pso_leaky_relu;
        case ACTIVATION_TANH:       return g_metal.pso_tanh;
        case ACTIVATION_SOFTSIGN:   return g_metal.pso_softsign;
        case ACTIVATION_GELU:       return g_metal.pso_gelu;
        default: return nil;  // ACTIVATION_NULL, ACTIVATION_SOFTMAX handled separately
    }
}

static void dispatch_binary_elementwise(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                                        id<MTLBuffer> buf0, id<MTLBuffer> buf1, NSUInteger count);

//-----------------------------------------------------------
// A cached MPS GEMM kernel: C = alpha * op(A) * op(B) + beta * C, with
// op(A) [M x K], op(B) [K x N]. Creating these is costly, so one is kept
// per shape and reused for every batch.
//-----------------------------------------------------------
static MPSMatrixMultiplication *gemm_kernel(BOOL trans_a, BOOL trans_b, NSUInteger M, NSUInteger N,
                                            NSUInteger K, double alpha, double beta)
{
    NSString *key = [NSString stringWithFormat:@"%d%d:%lu:%lu:%lu:%g:%g", trans_a, trans_b,
                     (unsigned long)M, (unsigned long)N, (unsigned long)K, alpha, beta];
    MPSMatrixMultiplication *k = g_metal.gemm_cache[key];
    if (!k) {
        k = [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                              transposeLeft:trans_a
                                             transposeRight:trans_b
                                                 resultRows:M
                                              resultColumns:N
                                            interiorColumns:K
                                                      alpha:alpha
                                                       beta:beta];
        g_metal.gemm_cache[key] = k;
    }
    return k;
}

//-----------------------------------------------------------
// A row-major [rows x cols] float matrix over a buffer
//-----------------------------------------------------------
static MPSMatrix *gpu_matrix(id<MTLBuffer> buf, NSUInteger rows, NSUInteger cols)
{
    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:cols * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
}

//-----------------------------------------------------------
// Encode C = alpha * op(A) * op(B) + beta * C. A, B and C are given with
// their stored shapes; op() transposes when asked.
//-----------------------------------------------------------
static void encode_gemm(id<MTLCommandBuffer> cmd,
                        id<MTLBuffer> a, NSUInteger a_rows, NSUInteger a_cols, BOOL trans_a,
                        id<MTLBuffer> b, NSUInteger b_rows, NSUInteger b_cols, BOOL trans_b,
                        id<MTLBuffer> c, double alpha, double beta)
{
    NSUInteger M = trans_a ? a_cols : a_rows;
    NSUInteger K = trans_a ? a_rows : a_cols;
    NSUInteger N = trans_b ? b_rows : b_cols;
    [gemm_kernel(trans_a, trans_b, M, N, K, alpha, beta) encodeToCommandBuffer:cmd
                                                                    leftMatrix:gpu_matrix(a, a_rows, a_cols)
                                                                   rightMatrix:gpu_matrix(b, b_rows, b_cols)
                                                                  resultMatrix:gpu_matrix(c, M, N)];
}

//-----------------------------------------------------------
// Encode a dense layer's forward pass for a batch: Y = X * W^T + b, then
// (when Z is given) save the pre-activation, apply the activation, and
// apply dropout when a mask is given.
//-----------------------------------------------------------
static void encode_dense_forward(id<MTLCommandBuffer> cmd, PLayer dst, int in_nodes, int batch_size,
                                 id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> Z, id<MTLBuffer> mask)
{
    int out_nodes = dst->node_count;
    NSUInteger count = (NSUInteger)batch_size * out_nodes;
    id<MTLBuffer> W = (__bridge id<MTLBuffer>)dst->t_weights->gpu_buf;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)dst->t_bias->gpu_buf;

    encode_gemm(cmd, X, batch_size, in_nodes, NO, W, out_nodes, in_nodes, YES, Y, 1.0, 0.0);
    dispatch_bias_add(cmd, Y, b, (NSUInteger)batch_size, (NSUInteger)out_nodes);

    if (Z) {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:Y sourceOffset:0 toBuffer:Z destinationOffset:0 size:count * sizeof(float)];
        [blit endEncoding];
    }

    if (dst->activation == ACTIVATION_SOFTMAX) {
        dispatch_softmax(cmd, Y, (uint32_t)out_nodes, (NSUInteger)batch_size);
    } else if (dst->activation != ACTIVATION_NULL) {
        id<MTLComputePipelineState> pso = pso_for_activation(dst->activation);
        if (pso)
            dispatch_activation(cmd, pso, Y, count);
    }

    if (mask)
        dispatch_binary_elementwise(cmd, g_metal.pso_dropout_forward, Y, mask, count);
}

//-----------------------------------------------------------
// GPU batched inference: Y = act(X * W^T + b) layer by layer, encoded in one
// command buffer.
//
// Inputs:
//   pnet       - trained network (weights already uploaded)
//   input_buf  - [batch_size x input_nodes] MTLBuffer (already on GPU)
//   output_buf - [batch_size x output_nodes] MTLBuffer (result destination)
//   batch_size - number of samples in the batch
//-----------------------------------------------------------
static int metal_forward_pass(PNetwork pnet,
                              id<MTLBuffer> input_buf,
                              id<MTLBuffer> output_buf,
                              int batch_size)
{
    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
    id<MTLBuffer> cur_buf = input_buf;

    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer dst = &pnet->layers[layer];
        if (!dst->t_weights || !dst->t_weights->gpu_buf || !dst->t_bias || !dst->t_bias->gpu_buf)
        {
            fprintf(stderr, "[Metal] Layer %d weights/biases not on GPU\n", layer);
            return 0;
        }

        // the last layer writes straight into the caller's buffer
        id<MTLBuffer> out = output_buf;
        if (layer < pnet->layer_count - 1) {
            out = [g_metal.device newBufferWithLength:(size_t)batch_size * dst->node_count * sizeof(float)
                                              options:MTLResourceStorageModePrivate];
            if (!out) return 0;
        }

        encode_dense_forward(cmd, dst, pnet->layers[layer - 1].node_count, batch_size, cur_buf, out, nil, nil);
        cur_buf = out;
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

//-----------------------------------------------------------
// Internal Metal backend functions (registered in metal_backend vtable).
// These are not part of the public libann API.
//-----------------------------------------------------------

static int metal_upload_network(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized)
        return ERR_NULL_PTR;

    // Force per-network training buffer rebind/allocation on next train call.
    g_metal.training_buffers_ready = 0;
    g_metal.training_batch_size = 0;

    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
            if (!tensor_metal_upload(l->params[p].value)) return ERR_FAIL;
    }

    return ERR_OK;
}

static void metal_free_network(PNetwork pnet)
{
    if (!pnet) return;

    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++) {
            PTensor v = l->params[p].value;
            if (v && v->gpu_buf) {
                tensor_metal_release_buffer(v->gpu_buf);
                v->gpu_buf = NULL;
            }
        }
    }

    // Network-local training buffers are released by tensor_free() during
    // ann_free_network(); reset global readiness so stale state is never reused.
    g_metal.training_buffers_ready = 0;
    g_metal.training_batch_size = 0;
}

//-----------------------------------------------------------
// GPU-accelerated single-sample inference (Metal).
// Returns 1 on success, 0 if GPU not ready (fallback to CPU).
//-----------------------------------------------------------
static int metal_eval_single(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized)
        return 0;

    // Check that weights are on GPU
    if (pnet->layer_count < 2 || !pnet->layers[1].t_weights || !pnet->layers[1].t_weights->gpu_buf)
        return 0;   // not uploaded, use CPU path

    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[pnet->layer_count - 1].node_count;

    // Wrap input (already set in t_values) in a shared MTLBuffer
    size_t in_bytes  = (size_t)input_nodes  * sizeof(float);
    size_t out_bytes = (size_t)output_nodes * sizeof(float);

    id<MTLBuffer> input_buf = [g_metal.device
        newBufferWithBytes:pnet->layers[0].t_values->values
                   length:in_bytes
                  options:MTLResourceStorageModeShared];

    id<MTLBuffer> output_buf = [g_metal.device
        newBufferWithLength:out_bytes
                    options:MTLResourceStorageModeShared];

    if (!input_buf || !output_buf)
        return 0;

    if (!metal_forward_pass(pnet, input_buf, output_buf, 1))
        return 0;

    // Write GPU output back into the output layer's t_values
    memcpy(pnet->layers[pnet->layer_count - 1].t_values->values,
           [output_buf contents],
           out_bytes);

    return 1;
}

//-----------------------------------------------------------
// GPU-accelerated batch inference.
//
// @param pnet       Trained network (ann_gpu_upload_network() must have been called)
// @param inputs     Row-major array [batch_size x input_nodes]
// @param outputs    Row-major array [batch_size x output_nodes] (written by this fn)
// @param batch_size Number of samples to process
// @return ERR_OK on success
//-----------------------------------------------------------
static int metal_eval_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
    if (!pnet || !inputs || !outputs || batch_size <= 0)
        return ERR_NULL_PTR;

    if (!g_metal.initialized)
        return ERR_FAIL;

    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[pnet->layer_count - 1].node_count;

    size_t in_bytes  = (size_t)batch_size * input_nodes  * sizeof(float);
    size_t out_bytes = (size_t)batch_size * output_nodes * sizeof(float);

    id<MTLBuffer> input_buf = [g_metal.device newBufferWithBytes:inputs
                                                          length:in_bytes
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [g_metal.device newBufferWithLength:out_bytes
                                                           options:MTLResourceStorageModeShared];
    if (!input_buf || !output_buf)
        return ERR_ALLOC;

    if (!metal_forward_pass((PNetwork)pnet, input_buf, output_buf, batch_size))
        return ERR_FAIL;

    memcpy(outputs, [output_buf contents], out_bytes);
    return ERR_OK;
}

// ============================================================================
// GPU TRAINING
//
// Each batch is encoded into one command buffer: zero the gradients, forward
// pass, output gradient and loss, backward pass, then clipping, the optimizer
// update and regularization. The CPU waits once, at the end, to read the loss.
// ============================================================================

//-----------------------------------------------------------
// Helper: alloc or realloc a GPU shared buffer on a tensor.
// MTLResourceStorageModeShared buffers are visible to both CPU and GPU, so
// inputs, targets and dropout masks are written into them directly.
//-----------------------------------------------------------
static int gpu_ensure_shared_buf(PTensor t)
{
    if (!t) return 0;
    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);
    if (t->gpu_buf) {
        // Already allocated — verify size matches (re-alloc if needed)
        id<MTLBuffer> existing = (__bridge id<MTLBuffer>)t->gpu_buf;
        if ([existing length] >= bytes) return 1;
        tensor_metal_release_buffer(t->gpu_buf);
        t->gpu_buf = NULL;
    }
    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (!buf) return 0;
    memset([buf contents], 0, bytes);
    t->gpu_buf = (__bridge_retained void *)buf;
    return 1;
}

//-----------------------------------------------------------
// Ensure a shared GPU buffer and fill it with the tensor's CPU values.
//-----------------------------------------------------------
static int gpu_upload_shared_buf(PTensor t)
{
    if (!gpu_ensure_shared_buf(t)) return 0;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t->gpu_buf;
    memcpy([buf contents], t->values, (size_t)(t->rows * t->cols) * sizeof(real));
    return 1;
}

//-----------------------------------------------------------
// A shared buffer of at least bytes, reallocated when too small
//-----------------------------------------------------------
static id<MTLBuffer> gpu_scratch(id<MTLBuffer> buf, size_t bytes)
{
    if (buf && [buf length] >= bytes)
        return buf;
    return [g_metal.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

//-----------------------------------------------------------
// The dropout mask for a hidden layer in this batch, copied to the GPU.
// ann.c draws the masks before each batch. Returns nil when the layer
// does not use dropout.
//-----------------------------------------------------------
static id<MTLBuffer> gpu_dropout_mask(PNetwork pnet, PLayer l, int batch_size)
{
    PTensor m = l->t_dropout_mask;
    if (!pnet->is_training || l->dropout_rate <= 0.0f || !m ||
        m->rows != batch_size || m->cols != l->node_count)
        return nil;
    if (!gpu_upload_shared_buf(m)) return nil;
    return (__bridge id<MTLBuffer>)m->gpu_buf;
}

//-----------------------------------------------------------
// Allocate GPU buffers for all training tensors.
// Parameter values already have gpu_buf from ann_gpu_upload_network().
// This adds the batch activations, pre-activations and gradients, and each
// parameter's gradient and optimizer state (carried over from the CPU so
// training continues where it left off). Called lazily by metal_train_batch().
//-----------------------------------------------------------
static int gpu_alloc_training_buffers(PNetwork pnet, int batch_size)
{
    for (int layer = 0; layer < pnet->layer_count; layer++) {
        PLayer l = &pnet->layers[layer];

        // t_batch_values: all layers [batch x nodes]
        if (!l->t_batch_values || !gpu_ensure_shared_buf(l->t_batch_values)) return 0;
        if (layer == 0)
            continue;

        // Non-input layers also need pre-activation and gradient buffers
        if (!l->t_batch_z     || !gpu_ensure_shared_buf(l->t_batch_z))     return 0;
        if (!l->t_batch_dl_dz || !gpu_ensure_shared_buf(l->t_batch_dl_dz)) return 0;

        for (int p = 0; p < l->param_count; p++) {
            Layer_param *param = &l->params[p];
            if (!gpu_ensure_shared_buf(param->grad)) return 0;
            if (param->m && !gpu_upload_shared_buf(param->m)) return 0;
            if (param->v && !gpu_upload_shared_buf(param->v)) return 0;
        }
    }

    g_metal.training_buffers_ready = 1;
    g_metal.training_batch_size = batch_size;
    return 1;
}

//-----------------------------------------------------------
// Encode a 1-D kernel over count elements: buffers at indices 0..n_bufs-1,
// then float constants at the following indices
//-----------------------------------------------------------
static void encode_1d(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                      id<MTLBuffer> __strong *bufs, int n_bufs,
                      const float *consts, int n_consts, NSUInteger count)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
    for (int i = 0; i < n_consts; i++)
        [enc setBytes:&consts[i] length:sizeof(float) atIndex:(NSUInteger)(n_bufs + i)];
    NSUInteger tg = MIN(pso.maxTotalThreadsPerThreadgroup, count);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

//-----------------------------------------------------------
// Helper: dispatch a 1-D element-wise kernel with 2 buffers.
//-----------------------------------------------------------
static void dispatch_binary_elementwise(id<MTLCommandBuffer> cmd,
                                        id<MTLComputePipelineState> pso,
                                        id<MTLBuffer> buf0,
                                        id<MTLBuffer> buf1,
                                        NSUInteger count)
{
    id<MTLBuffer> bufs[2] = { buf0, buf1 };
    encode_1d(cmd, pso, bufs, 2, NULL, 0, count);
}

//-----------------------------------------------------------
// Select the derivative PSO for a given activation.
// Returns nil for activations that need no kernel (ACTIVATION_NULL; softmax
// is handled by softmax_jacobian).
//-----------------------------------------------------------
static id<MTLComputePipelineState> pso_for_deriv(Activation_type act)
{
    switch (act) {
        case ACTIVATION_SIGMOID:    return g_metal.pso_deriv_sigmoid;
        case ACTIVATION_RELU:       return g_metal.pso_deriv_relu;
        case ACTIVATION_LEAKY_RELU: return g_metal.pso_deriv_leaky_relu;
        case ACTIVATION_TANH:       return g_metal.pso_deriv_tanh;
        case ACTIVATION_SOFTSIGN:   return g_metal.pso_deriv_softsign;
        case ACTIVATION_GELU:       return g_metal.pso_deriv_gelu;
        default: return nil;
    }
}

//-----------------------------------------------------------
// Encode dL/dz = dL/da * f'(z) for a layer, reading A (sigmoid, tanh,
// softsign) or Z (ReLU, leaky ReLU, GELU) as the CPU path does
//-----------------------------------------------------------
static void encode_activation_derivative(id<MTLCommandBuffer> cmd, PLayer l, int batch_size)
{
    NSUInteger count = (NSUInteger)batch_size * l->node_count;
    id<MTLBuffer> dl_dz = (__bridge id<MTLBuffer>)l->t_batch_dl_dz->gpu_buf;
    id<MTLBuffer> A = (__bridge id<MTLBuffer>)l->t_batch_values->gpu_buf;
    id<MTLBuffer> Z = (__bridge id<MTLBuffer>)l->t_batch_z->gpu_buf;

    if (l->activation == ACTIVATION_SOFTMAX) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        uint32_t cols = (uint32_t)l->node_count;
        [enc setComputePipelineState:g_metal.pso_softmax_jacobian];
        [enc setBuffer:dl_dz offset:0 atIndex:0];
        [enc setBuffer:A offset:0 atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)batch_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        return;
    }

    id<MTLComputePipelineState> pso = pso_for_deriv(l->activation);
    if (!pso)
        return;
    int uses_z = l->activation == ACTIVATION_RELU || l->activation == ACTIVATION_LEAKY_RELU ||
                 l->activation == ACTIVATION_GELU;
    dispatch_binary_elementwise(cmd, pso, dl_dz, uses_z ? Z : A, count);
}

//-----------------------------------------------------------
// Encode the output layer's delta (t - y, then the activation derivative for
// MSE; cross-entropy with softmax/sigmoid needs none) and each row's loss
//-----------------------------------------------------------
static void encode_output_delta_loss(id<MTLCommandBuffer> cmd, PNetwork pnet, int batch_size)
{
    PLayer out = &pnet->layers[pnet->layer_count - 1];
    uint32_t cols = (uint32_t)out->node_count;
    uint32_t mode = pnet->loss_type == LOSS_MSE ? 0 : (out->activation == ACTIVATION_SIGMOID ? 2 : 1);

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_metal.pso_output_delta_loss];
    [enc setBuffer:(__bridge id<MTLBuffer>)out->t_batch_values->gpu_buf offset:0 atIndex:0];
    [enc setBuffer:g_metal.targets_buf offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out->t_batch_dl_dz->gpu_buf offset:0 atIndex:2];
    [enc setBuffer:g_metal.row_loss_buf offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&mode length:sizeof(mode) atIndex:5];
    NSUInteger tg = MIN(g_metal.pso_output_delta_loss.maxTotalThreadsPerThreadgroup, (NSUInteger)batch_size);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)batch_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];

    if (pnet->loss_type == LOSS_MSE)
        encode_activation_derivative(cmd, out, batch_size);
}

//-----------------------------------------------------------
// Encode the backward pass. On entry the output layer's t_batch_dl_dz holds
// -dL/dz. For each layer, from the output down to layer 1:
//   1. hidden layers: undo dropout, then the activation derivative
//   2. dW += dl_dz^T * A_prev and bias_grad += column sums of dl_dz
//   3. dl_dz_prev = dl_dz * W (not for layer 1, whose input needs none)
//-----------------------------------------------------------
static void encode_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int batch_size)
{
    for (int li = pnet->layer_count - 1; li >= 1; li--) {
        PLayer layer      = &pnet->layers[li];
        PLayer prev_layer = &pnet->layers[li - 1];
        int nodes      = layer->node_count;
        int prev_nodes = prev_layer->node_count;
        NSUInteger count = (NSUInteger)batch_size * nodes;

        id<MTLBuffer> dl_dz  = (__bridge id<MTLBuffer>)layer->t_batch_dl_dz->gpu_buf;
        id<MTLBuffer> A_prev = (__bridge id<MTLBuffer>)prev_layer->t_batch_values->gpu_buf;

        if (li < pnet->layer_count - 1) {
            id<MTLBuffer> mask = gpu_dropout_mask(pnet, layer, batch_size);
            if (mask) {
                id<MTLBuffer> bufs[3] = { dl_dz, (__bridge id<MTLBuffer>)layer->t_batch_values->gpu_buf, mask };
                encode_1d(cmd, g_metal.pso_dropout_backward, bufs, 3, NULL, 0, count);
            }
            encode_activation_derivative(cmd, layer, batch_size);
        }

        // dW += dl_dz^T * A_prev  ([nodes x batch] * [batch x prev_nodes])
        encode_gemm(cmd, dl_dz, batch_size, nodes, YES, A_prev, batch_size, prev_nodes, NO,
                    (__bridge id<MTLBuffer>)layer->t_gradients->gpu_buf, 1.0, 1.0);

        // bias_grad += column sums of dl_dz
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            uint32_t bs = (uint32_t)batch_size, n = (uint32_t)nodes;
            [enc setComputePipelineState:g_metal.pso_bias_grad_sum];
            [enc setBuffer:dl_dz offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)layer->t_bias_grad->gpu_buf offset:0 atIndex:1];
            [enc setBytes:&bs length:sizeof(bs) atIndex:2];
            [enc setBytes:&n  length:sizeof(n)  atIndex:3];
            NSUInteger tg = MIN(g_metal.pso_bias_grad_sum.maxTotalThreadsPerThreadgroup, (NSUInteger)nodes);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)nodes, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        // dl_dz_prev = dl_dz * W  ([batch x nodes] * [nodes x prev_nodes])
        if (li > 1)
            encode_gemm(cmd, dl_dz, batch_size, nodes, NO,
                        (__bridge id<MTLBuffer>)layer->t_weights->gpu_buf, nodes, prev_nodes, NO,
                        (__bridge id<MTLBuffer>)prev_layer->t_batch_dl_dz->gpu_buf, 1.0, 0.0);
    }
}

//-----------------------------------------------------------
// Encode gradient clipping, the optimizer update and L1/L2 regularization for
// every parameter, as the CPU optimizers do. pnet->train_iteration must
// already count this batch (Adam's bias correction).
//-----------------------------------------------------------
static void encode_optimizer(id<MTLCommandBuffer> cmd, PNetwork pnet)
{
    float lr = (float)pnet->learning_rate;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, beta = 0.9f;
    float bc1 = 0.0f, bc2 = 0.0f;
    if (pnet->optimizer == OPT_ADAM) {
        float t = (float)pnet->train_iteration;
        bc1 = 1.0f / (1.0f - powf(beta1, t));
        bc2 = 1.0f / (1.0f - powf(beta2, t));
    }

    for (int layer = 1; layer < pnet->layer_count; layer++) {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++) {
            Layer_param *param = &l->params[p];
            NSUInteger count = (NSUInteger)(param->value->rows * param->value->cols);
            id<MTLBuffer> w = (__bridge id<MTLBuffer>)param->value->gpu_buf;
            id<MTLBuffer> g = (__bridge id<MTLBuffer>)param->grad->gpu_buf;
            id<MTLBuffer> m = param->m ? (__bridge id<MTLBuffer>)param->m->gpu_buf : nil;
            id<MTLBuffer> v = param->v ? (__bridge id<MTLBuffer>)param->v->gpu_buf : nil;

            if (pnet->max_gradient > 0.0f) {
                float mg = (float)pnet->max_gradient;
                encode_1d(cmd, g_metal.pso_gradient_clip, &g, 1, &mg, 1, count);
            }

            switch (pnet->optimizer) {
            case OPT_SGD: {
                id<MTLBuffer> bufs[2] = { w, g };
                encode_1d(cmd, g_metal.pso_sgd_update, bufs, 2, &lr, 1, count);
                break;
            }
            case OPT_MOMENTUM: {
                id<MTLBuffer> bufs[3] = { w, g, m };
                float c[2] = { lr, beta };
                encode_1d(cmd, g_metal.pso_momentum_update, bufs, 3, c, 2, count);
                break;
            }
            case OPT_ADAGRAD: {
                id<MTLBuffer> bufs[3] = { w, g, v };
                float c[2] = { lr, eps };
                encode_1d(cmd, g_metal.pso_adagrad_update, bufs, 3, c, 2, count);
                break;
            }
            case OPT_RMSPROP: {
                id<MTLBuffer> bufs[3] = { w, g, v };
                float c[3] = { lr, beta, eps };
                encode_1d(cmd, g_metal.pso_rmsprop_update, bufs, 3, c, 3, count);
                break;
            }
            case OPT_ADAM: {
                id<MTLBuffer> bufs[4] = { w, g, m, v };
                float c[6] = { lr, beta1, beta2, eps, bc1, bc2 };
                encode_1d(cmd, g_metal.pso_adam_update, bufs, 4, c, 6, count);
                break;
            }
            default:
                break;
            }

            if (!param->regularize)
                continue;
            if (pnet->l2_lambda > 0.0f) {
                float decay = 1.0f - (float)(pnet->learning_rate * pnet->l2_lambda);
                encode_1d(cmd, g_metal.pso_l2_regularize, &w, 1, &decay, 1, count);
            }
            if (pnet->l1_lambda > 0.0f) {
                float delta = (float)(pnet->learning_rate * pnet->l1_lambda);
                encode_1d(cmd, g_metal.pso_l1_regularize, &w, 1, &delta, 1, count);
            }
        }
    }
}

//-----------------------------------------------------------
// metal_sync_weights()
// Copy every parameter and its optimizer state from the GPU to the CPU, so
// ann_predict(), ann_save_network() and CPU (online) training see the latest
// values.
//-----------------------------------------------------------
static void metal_sync_weights(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized) return;

    for (int layer = 1; layer < pnet->layer_count; layer++) {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++) {
            PTensor state[3] = { l->params[p].value, l->params[p].m, l->params[p].v };
            for (int i = 0; i < 3; i++) {
                if (state[i] && state[i]->gpu_buf) {
                    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)state[i]->gpu_buf;
                    memcpy(state[i]->values, [buf contents],
                           (size_t)(state[i]->rows * state[i]->cols) * sizeof(real));
                }
            }
        }
    }
}

//-----------------------------------------------------------
// metal_train_batch()
// Top-level GPU training hook called from ann_train_network(). The batch
// inputs are in layers[0].t_batch_values and the dropout masks are drawn.
// Returns 1 when the GPU trained the batch (loss in *loss_out), 0 to fall
// back to the CPU.
//-----------------------------------------------------------
static int metal_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out)
{
    if (!pnet || !g_metal.initialized) return 0;
    if (pnet->layer_count < 2 || !pnet->layers[1].t_weights || !pnet->layers[1].t_weights->gpu_buf) return 0;
    if (!pnet->layers[0].t_batch_values) return 0;

    // Lazy-alloc training buffers on first call (or if batch size changed)
    if (!g_metal.training_buffers_ready || g_metal.training_batch_size != batch_size) {
        g_metal.training_buffers_ready = 0;
        if (!gpu_alloc_training_buffers(pnet, batch_size)) return 0;
    }

    int output_layer_idx = pnet->layer_count - 1;
    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[output_layer_idx].node_count;

    // Defensive re-allocation for stale backend state across network lifetimes.
    if (!pnet->layers[0].t_batch_values->gpu_buf ||
        !pnet->layers[output_layer_idx].t_batch_values ||
        !pnet->layers[output_layer_idx].t_batch_values->gpu_buf ||
        !pnet->layers[output_layer_idx].t_batch_dl_dz ||
        !pnet->layers[output_layer_idx].t_batch_dl_dz->gpu_buf) {
        g_metal.training_buffers_ready = 0;
        if (!gpu_alloc_training_buffers(pnet, batch_size)) return 0;
    }

    // Inputs and targets go into shared buffers before anything is encoded
    size_t target_bytes = (size_t)batch_size * output_nodes * sizeof(float);
    g_metal.targets_buf  = gpu_scratch(g_metal.targets_buf, target_bytes);
    g_metal.row_loss_buf = gpu_scratch(g_metal.row_loss_buf, (size_t)batch_size * sizeof(float));
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)pnet->layers[0].t_batch_values->gpu_buf;
    if (!g_metal.targets_buf || !g_metal.row_loss_buf || ![in_buf contents]) return 0;
    memcpy([in_buf contents], pnet->layers[0].t_batch_values->values, (size_t)batch_size * input_nodes * sizeof(float));
    memcpy([g_metal.targets_buf contents], batch_targets->values, target_bytes);

    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];

    // Zero the gradients
    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        for (int layer = 1; layer < pnet->layer_count; layer++) {
            PLayer l = &pnet->layers[layer];
            for (int p = 0; p < l->param_count; p++) {
                PTensor g = l->params[p].grad;
                [blit fillBuffer:(__bridge id<MTLBuffer>)g->gpu_buf
                           range:NSMakeRange(0, (NSUInteger)(g->rows * g->cols) * sizeof(float))
                           value:0];
            }
        }
        [blit endEncoding];
    }

    // Forward pass, saving pre-activations; dropout on hidden layers
    for (int layer = 1; layer < pnet->layer_count; layer++) {
        PLayer dst = &pnet->layers[layer];
        id<MTLBuffer> mask = layer < output_layer_idx ? gpu_dropout_mask(pnet, dst, batch_size) : nil;
        encode_dense_forward(cmd, dst, pnet->layers[layer - 1].node_count, batch_size,
                             (__bridge id<MTLBuffer>)pnet->layers[layer - 1].t_batch_values->gpu_buf,
                             (__bridge id<MTLBuffer>)dst->t_batch_values->gpu_buf,
                             (__bridge id<MTLBuffer>)dst->t_batch_z->gpu_buf, mask);
    }

    encode_output_delta_loss(cmd, pnet, batch_size);
    encode_backward(cmd, pnet, batch_size);

    // Adam's bias correction counts this batch
    pnet->train_iteration++;
    encode_optimizer(cmd, pnet);

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
        pnet->train_iteration--;
        return 0;
    }

    // Loss: averaged the same way as the CPU path
    const float *row_loss = (const float *)[g_metal.row_loss_buf contents];
    real total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++)
        total_loss += row_loss[b];
    if (pnet->loss_type == LOSS_MSE)
        total_loss /= (real)(batch_size * output_nodes);
    else
        total_loss /= (real)batch_size;
    if (loss_out) *loss_out = total_loss;

    return 1;
}

//-----------------------------------------------------------
// Metal backend vtable instance.
// ann_gpu_init() (in ann.c) sets g_gpu_backend = &metal_backend.
//-----------------------------------------------------------
//-----------------------------------------------------------
// Autorelease pool wrappers.
// Command buffers, encoders and other temporaries are autoreleased. Callers
// are plain C with no pool of their own, so each entry point that creates
// Metal objects or touches existing ones drains a pool before returning;
// otherwise autoreleased objects accumulate for the life of the thread.
//-----------------------------------------------------------
static int pooled_init(void)
{
    @autoreleasepool { return tensor_metal_init(); }
}

static int pooled_upload_network(PNetwork pnet)
{
    @autoreleasepool { return metal_upload_network(pnet); }
}

static void pooled_free_network(PNetwork pnet)
{
    @autoreleasepool { metal_free_network(pnet); }
}

static void pooled_sync_weights(PNetwork pnet)
{
    @autoreleasepool { metal_sync_weights(pnet); }
}

static void pooled_release_buffer(void *gpu_buf)
{
    @autoreleasepool { tensor_metal_release_buffer(gpu_buf); }
}

static int pooled_eval_single(PNetwork pnet)
{
    @autoreleasepool { return metal_eval_single(pnet); }
}

static int pooled_eval_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
    @autoreleasepool { return metal_eval_batch(pnet, inputs, outputs, batch_size); }
}

static int pooled_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out)
{
    @autoreleasepool { return metal_train_batch(pnet, batch_targets, batch_size, loss_out); }
}

GpuBackend metal_backend = {
    pooled_init,             /* init           */
    pooled_upload_network,   /* upload_network */
    pooled_free_network,     /* free_network   */
    pooled_sync_weights,     /* sync_weights   */
    pooled_release_buffer,   /* release_buffer */
    pooled_eval_single,      /* eval_single    */
    pooled_eval_batch,       /* eval_batch     */
    pooled_train_batch,      /* train_batch    */
};

#endif /* USE_METAL */
