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
    id<MTLComputePipelineState> pso_softmax;
    id<MTLComputePipelineState> pso_bias_add;
    // Training: activation derivative kernels
    id<MTLComputePipelineState> pso_deriv_sigmoid;
    id<MTLComputePipelineState> pso_deriv_relu;
    id<MTLComputePipelineState> pso_deriv_leaky_relu;
    id<MTLComputePipelineState> pso_deriv_tanh;
    id<MTLComputePipelineState> pso_deriv_softsign;
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
        "    else if (wi < 0.0f) w[gid] += delta; }\n";

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
    g_metal.pso_softmax    = make_pso(g_metal.library, "softmax_rows");
    g_metal.pso_bias_add   = make_pso(g_metal.library, "bias_add");

    if (!g_metal.pso_sigmoid || !g_metal.pso_relu || !g_metal.pso_leaky_relu ||
        !g_metal.pso_tanh    || !g_metal.pso_softsign || !g_metal.pso_softmax ||
        !g_metal.pso_bias_add)
        return 0;

    // Build PSOs for training kernels
    g_metal.pso_deriv_sigmoid    = make_pso(g_metal.library, "deriv_sigmoid");
    g_metal.pso_deriv_relu       = make_pso(g_metal.library, "deriv_relu");
    g_metal.pso_deriv_leaky_relu = make_pso(g_metal.library, "deriv_leaky_relu");
    g_metal.pso_deriv_tanh       = make_pso(g_metal.library, "deriv_tanh");
    g_metal.pso_deriv_softsign   = make_pso(g_metal.library, "deriv_softsign");
    g_metal.pso_bias_grad_sum    = make_pso(g_metal.library, "bias_grad_sum");
    g_metal.pso_gradient_clip    = make_pso(g_metal.library, "gradient_clip");
    g_metal.pso_sgd_update       = make_pso(g_metal.library, "sgd_update");
    g_metal.pso_momentum_update  = make_pso(g_metal.library, "momentum_update");
    g_metal.pso_adagrad_update   = make_pso(g_metal.library, "adagrad_update");
    g_metal.pso_rmsprop_update   = make_pso(g_metal.library, "rmsprop_update");
    g_metal.pso_adam_update      = make_pso(g_metal.library, "adam_update");
    g_metal.pso_l2_regularize    = make_pso(g_metal.library, "l2_regularize");
    g_metal.pso_l1_regularize    = make_pso(g_metal.library, "l1_regularize");

    if (!g_metal.pso_deriv_sigmoid || !g_metal.pso_deriv_relu   ||
        !g_metal.pso_deriv_leaky_relu || !g_metal.pso_deriv_tanh ||
        !g_metal.pso_deriv_softsign || !g_metal.pso_bias_grad_sum ||
        !g_metal.pso_gradient_clip  || !g_metal.pso_sgd_update   ||
        !g_metal.pso_momentum_update || !g_metal.pso_adagrad_update ||
        !g_metal.pso_rmsprop_update || !g_metal.pso_adam_update  ||
        !g_metal.pso_l2_regularize  || !g_metal.pso_l1_regularize)
        return 0;

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
        // Release the Objective-C object via ARC bridge
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)gpu_buf;
        buf = nil;  // triggers dealloc
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
// Allocate an uninitialized GPU buffer for a tensor.
// Used for intermediate (output) tensors during forward pass.
// Returns 1 on success, 0 on failure.
//-----------------------------------------------------------
static int tensor_metal_alloc_gpu(PTensor t)
{
    if (!t || !g_metal.initialized)
        return 0;

    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);

    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (!buf)
        return 0;

    if (t->gpu_buf)
        tensor_metal_release_buffer(t->gpu_buf);

    t->gpu_buf = (__bridge_retained void *)buf;
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
        default: return nil;  // ACTIVATION_NULL, ACTIVATION_SOFTMAX handled separately
    }
}

//-----------------------------------------------------------
// GPU batched forward pass.
// Performs Y = X * W^T + b + activation(Y) for each layer.
//
// Inputs:
//   pnet       - trained network (weights already uploaded)
//   input_buf  - [batch_size x input_nodes] MTLBuffer (already on GPU)
//   output_buf - [batch_size x output_nodes] MTLBuffer (result destination)
//   batch_size - number of samples in the batch
//
// Intermediate buffers are allocated per call (could be cached for perf).
//-----------------------------------------------------------
static int metal_forward_pass(PNetwork pnet,
                              id<MTLBuffer> input_buf,
                              id<MTLBuffer> output_buf,
                              int batch_size)
{
    int layer_count = pnet->layer_count;

    // Track current activation buffer as we propagate forward
    id<MTLBuffer> cur_buf = input_buf;
    int cur_cols = pnet->layers[0].node_count;

    // Allocate intermediate buffers for each non-input layer
    NSMutableArray<id<MTLBuffer>> *intermediates = [NSMutableArray arrayWithCapacity:layer_count];

    for (int layer = 0; layer < layer_count - 1; layer++)
    {
        PLayer src = &pnet->layers[layer];
        PLayer dst = &pnet->layers[layer + 1];

        if (!src->t_weights || !src->t_weights->gpu_buf ||
            !src->t_bias    || !src->t_bias->gpu_buf)
        {
            fprintf(stderr, "[Metal] Layer %d weights/biases not on GPU\n", layer);
            return 0;
        }

        id<MTLBuffer> W_buf   = (__bridge id<MTLBuffer>)src->t_weights->gpu_buf;
        id<MTLBuffer> b_buf   = (__bridge id<MTLBuffer>)src->t_bias->gpu_buf;
        int out_nodes = dst->node_count;

        // Allocate output buffer for this layer: [batch_size x out_nodes]
        size_t out_bytes = (size_t)batch_size * out_nodes * sizeof(float);
        id<MTLBuffer> layer_out = [g_metal.device newBufferWithLength:out_bytes
                                                              options:MTLResourceStorageModeShared];
        if (!layer_out) return 0;
        [intermediates addObject:layer_out];

        // --- Matrix multiply: Y = X * W^T ---
        // X: [batch x in_nodes], W: [out_nodes x in_nodes] → Y: [batch x out_nodes]
        MPSMatrixDescriptor *X_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)batch_size
                             columns:(NSUInteger)cur_cols
                            rowBytes:(NSUInteger)(cur_cols * sizeof(float))
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *W_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)src->t_weights->rows
                             columns:(NSUInteger)src->t_weights->cols
                            rowBytes:(NSUInteger)(src->t_weights->cols * sizeof(float))
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *Y_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)batch_size
                             columns:(NSUInteger)out_nodes
                            rowBytes:(NSUInteger)(out_nodes * sizeof(float))
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *X_mat   = [[MPSMatrix alloc] initWithBuffer:cur_buf   descriptor:X_desc];
        MPSMatrix *W_mat   = [[MPSMatrix alloc] initWithBuffer:W_buf     descriptor:W_desc];
        MPSMatrix *Y_mat   = [[MPSMatrix alloc] initWithBuffer:layer_out descriptor:Y_desc];

        // Y = X * W^T  (transposeRight=YES)
        MPSMatrixMultiplication *gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                            transposeLeft:NO
                                           transposeRight:YES
                                               resultRows:(NSUInteger)batch_size
                                            resultColumns:(NSUInteger)out_nodes
                                          interiorColumns:(NSUInteger)cur_cols
                                                    alpha:1.0
                                                     beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
        [gemm encodeToCommandBuffer:cmd
                         leftMatrix:X_mat
                        rightMatrix:W_mat
                       resultMatrix:Y_mat];

        // --- Bias add: Y[b][n] += b[n] ---
        dispatch_bias_add(cmd, layer_out, b_buf, (NSUInteger)batch_size, (NSUInteger)out_nodes);

        // --- Activation ---
        NSUInteger elem_count = (NSUInteger)(batch_size * out_nodes);
        Activation_type act = dst->activation;

        if (act == ACTIVATION_SOFTMAX)
        {
            dispatch_softmax(cmd, layer_out, (uint32_t)out_nodes, (NSUInteger)batch_size);
        }
        else if (act != ACTIVATION_NULL)
        {
            id<MTLComputePipelineState> pso = pso_for_activation(act);
            if (pso)
                dispatch_activation(cmd, pso, layer_out, elem_count);
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        cur_buf  = layer_out;
        cur_cols = out_nodes;
    }

    // Copy final layer output into caller-supplied output_buf
    size_t final_bytes = (size_t)batch_size * cur_cols * sizeof(float);
    memcpy([output_buf contents], [cur_buf contents], final_bytes);

    return 1;
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

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];

        if (l->t_weights && !tensor_metal_upload(l->t_weights)) return ERR_FAIL;
        if (l->t_bias    && !tensor_metal_upload(l->t_bias))    return ERR_FAIL;
    }

    return ERR_OK;
}

static void metal_free_network(PNetwork pnet)
{
    if (!pnet) return;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];

        if (l->t_weights && l->t_weights->gpu_buf) {
            tensor_metal_release_buffer(l->t_weights->gpu_buf);
            l->t_weights->gpu_buf = NULL;
        }
        if (l->t_bias && l->t_bias->gpu_buf) {
            tensor_metal_release_buffer(l->t_bias->gpu_buf);
            l->t_bias->gpu_buf = NULL;
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
    if (!pnet->layers[0].t_weights || !pnet->layers[0].t_weights->gpu_buf)
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
// PHASE 2+3: GPU TRAINING SUPPORT
// ============================================================================

//-----------------------------------------------------------
// Helper: alloc or realloc a GPU shared buffer on a tensor.
// On Apple Silicon, MTLResourceStorageModeShared means the
// buffer is accessible by both CPU and GPU — no explicit
// blit needed for zero-init or reading back scalars.
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
// Allocate GPU buffers for all training tensors.
// Weights/biases already have gpu_buf from ann_gpu_upload_network().
// This function adds gpu_buf to all training-time tensors.
// Called lazily on first ann_gpu_train_batch() call.
//-----------------------------------------------------------
static int gpu_alloc_training_buffers(PNetwork pnet, int batch_size)
{
    for (int layer = 0; layer < pnet->layer_count; layer++) {
        PLayer l = &pnet->layers[layer];

        // t_batch_values: all layers [batch x nodes]
        if (!l->t_batch_values || !gpu_ensure_shared_buf(l->t_batch_values)) return 0;

        if (layer > 0) {
            // Non-input layers also need pre-activation and gradient buffers
            if (!l->t_batch_z     || !gpu_ensure_shared_buf(l->t_batch_z))     return 0;
            if (!l->t_batch_dl_dz || !gpu_ensure_shared_buf(l->t_batch_dl_dz)) return 0;
        }

        if (layer < pnet->layer_count - 1) {
            // Weight-carrying layers: gradients, optimizer state
            if (!l->t_gradients || !gpu_ensure_shared_buf(l->t_gradients)) return 0;
            if (!l->t_bias_grad  || !gpu_ensure_shared_buf(l->t_bias_grad))  return 0;
            // Optimizer state (may be NULL for SGD which doesn't use them)
            if (l->t_m     && !gpu_ensure_shared_buf(l->t_m))     return 0;
            if (l->t_v     && !gpu_ensure_shared_buf(l->t_v))     return 0;
            if (l->t_bias_m && !gpu_ensure_shared_buf(l->t_bias_m)) return 0;
            if (l->t_bias_v && !gpu_ensure_shared_buf(l->t_bias_v)) return 0;
        }
    }

    g_metal.training_buffers_ready = 1;
    g_metal.training_batch_size = batch_size;
    return 1;
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
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf0 offset:0 atIndex:0];
    [enc setBuffer:buf1 offset:0 atIndex:1];
    NSUInteger tg = MIN(pso.maxTotalThreadsPerThreadgroup, count);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

//-----------------------------------------------------------
// Select the derivative PSO for a given activation.
// Returns nil for activations that don't need a kernel
// (ACTIVATION_NULL handled via memset delta, ACTIVATION_SOFTMAX
//  uses delta = T - Y directly — no extra derivative needed).
//-----------------------------------------------------------
static id<MTLComputePipelineState> pso_for_deriv(Activation_type act)
{
    switch (act) {
        case ACTIVATION_SIGMOID:    return g_metal.pso_deriv_sigmoid;
        case ACTIVATION_RELU:       return g_metal.pso_deriv_relu;
        case ACTIVATION_LEAKY_RELU: return g_metal.pso_deriv_leaky_relu;
        case ACTIVATION_TANH:       return g_metal.pso_deriv_tanh;
        case ACTIVATION_SOFTSIGN:   return g_metal.pso_deriv_softsign;
        default: return nil;
    }
}

//-----------------------------------------------------------
// gpu_forward_training()
// Full forward pass saving pre-activation Z for backprop.
// On entry: layer[0].t_batch_values->gpu_buf has the batch inputs.
// On exit: all layers have t_batch_values->gpu_buf with activations,
//          non-input layers have t_batch_z->gpu_buf with pre-activations.
//-----------------------------------------------------------
static int gpu_forward_training(PNetwork pnet, int batch_size)
{
    for (int layer = 0; layer < pnet->layer_count - 1; layer++) {
        PLayer src = &pnet->layers[layer];
        PLayer dst = &pnet->layers[layer + 1];

        id<MTLBuffer> X_buf = (__bridge id<MTLBuffer>)src->t_batch_values->gpu_buf;
        id<MTLBuffer> W_buf = (__bridge id<MTLBuffer>)src->t_weights->gpu_buf;
        id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)src->t_bias->gpu_buf;
        id<MTLBuffer> Y_buf = (__bridge id<MTLBuffer>)dst->t_batch_values->gpu_buf;
        id<MTLBuffer> Z_buf = (__bridge id<MTLBuffer>)dst->t_batch_z->gpu_buf;

        int in_nodes  = src->node_count;
        int out_nodes = dst->node_count;

        // --- GEMM: Y = X * W^T ---
        MPSMatrixDescriptor *X_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)batch_size
                             columns:(NSUInteger)in_nodes
                            rowBytes:(NSUInteger)(in_nodes * sizeof(float))
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *W_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)src->t_weights->rows
                             columns:(NSUInteger)src->t_weights->cols
                            rowBytes:(NSUInteger)(src->t_weights->cols * sizeof(float))
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *Y_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)batch_size
                             columns:(NSUInteger)out_nodes
                            rowBytes:(NSUInteger)(out_nodes * sizeof(float))
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *X_mat = [[MPSMatrix alloc] initWithBuffer:X_buf descriptor:X_desc];
        MPSMatrix *W_mat = [[MPSMatrix alloc] initWithBuffer:W_buf descriptor:W_desc];
        MPSMatrix *Y_mat = [[MPSMatrix alloc] initWithBuffer:Y_buf descriptor:Y_desc];

        MPSMatrixMultiplication *gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                            transposeLeft:NO
                                           transposeRight:YES
                                               resultRows:(NSUInteger)batch_size
                                            resultColumns:(NSUInteger)out_nodes
                                          interiorColumns:(NSUInteger)in_nodes
                                                    alpha:1.0
                                                     beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
        [gemm encodeToCommandBuffer:cmd leftMatrix:X_mat rightMatrix:W_mat resultMatrix:Y_mat];

        // --- Bias add ---
        dispatch_bias_add(cmd, Y_buf, b_buf, (NSUInteger)batch_size, (NSUInteger)out_nodes);

        [cmd commit];
        [cmd waitUntilCompleted];

        // --- Save pre-activation Z from Y (shared memory copy, no GPU round-trip) ---
        size_t nbytes = (size_t)(batch_size * out_nodes) * sizeof(real);
        memcpy([Z_buf contents], [Y_buf contents], nbytes);

        // --- Activation ---
        Activation_type act = dst->activation;
        cmd = [g_metal.queue commandBuffer];

        if (act == ACTIVATION_SOFTMAX) {
            dispatch_softmax(cmd, Y_buf, (uint32_t)out_nodes, (NSUInteger)batch_size);
        } else if (act != ACTIVATION_NULL) {
            id<MTLComputePipelineState> pso = pso_for_activation(act);
            if (pso)
                dispatch_activation(cmd, pso, Y_buf, (NSUInteger)(batch_size * out_nodes));
        }

        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 1;
}

//-----------------------------------------------------------
// gpu_backward_pass()
// Backward pass after delta (= T - Y) is written to
// output_layer.t_batch_dl_dz->gpu_buf by the caller.
//
// For each layer backward (output-1 down to 1):
//   1. Apply activation derivative kernel to dl_dz using A or Z
//   2. MPS GEMM (transA): dW += dl_dz^T * A_prev / (accumulated)
//   3. bias_grad_sum kernel: bias_grad += col_sum(dl_dz)
//   4. MPS GEMM: dl_dz_prev = dl_dz * W  (propagate)
//-----------------------------------------------------------
static int gpu_backward_pass(PNetwork pnet, int batch_size)
{
    int layer_count = pnet->layer_count;

    for (int li = layer_count - 1; li >= 1; li--) {
        PLayer layer      = &pnet->layers[li];
        PLayer prev_layer = &pnet->layers[li - 1];

        int nodes      = layer->node_count;
        int prev_nodes = prev_layer->node_count;

        id<MTLBuffer> dl_dz_buf  = (__bridge id<MTLBuffer>)layer->t_batch_dl_dz->gpu_buf;
        id<MTLBuffer> A_buf      = (__bridge id<MTLBuffer>)layer->t_batch_values->gpu_buf;
        id<MTLBuffer> Z_buf      = (__bridge id<MTLBuffer>)layer->t_batch_z->gpu_buf;
        id<MTLBuffer> A_prev_buf = (__bridge id<MTLBuffer>)prev_layer->t_batch_values->gpu_buf;
        id<MTLBuffer> W_buf      = (__bridge id<MTLBuffer>)prev_layer->t_weights->gpu_buf;
        id<MTLBuffer> dW_buf     = (__bridge id<MTLBuffer>)prev_layer->t_gradients->gpu_buf;
        id<MTLBuffer> db_buf     = (__bridge id<MTLBuffer>)prev_layer->t_bias_grad->gpu_buf;

        NSUInteger elem_count = (NSUInteger)(batch_size * nodes);

        // --- Apply activation derivative ---
        // The CPU path never applies an activation derivative on the output layer:
        // back_propagate_output_batched() uses raw delta = T - Y for all loss types.
        // Only hidden layers get the activation derivative applied.
        int is_output_layer = (li == layer_count - 1);
        int skip_output_derivative = is_output_layer;
        if (!skip_output_derivative) {
            id<MTLComputePipelineState> pso_d = pso_for_deriv(layer->activation);
            if (pso_d) {
                // Use Z (pre-activation) for ReLU/LeakyReLU, A (post-activation) for others
                id<MTLBuffer> src_buf = (layer->activation == ACTIVATION_RELU ||
                                         layer->activation == ACTIVATION_LEAKY_RELU)
                                        ? Z_buf : A_buf;
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                dispatch_binary_elementwise(cmd, pso_d, dl_dz_buf, src_buf, elem_count);
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }

        // --- dW += dl_dz^T * A_prev  (GEMM transA) ---
        // dl_dz: [batch x nodes], A_prev: [batch x prev_nodes]
        // dl_dz^T: [nodes x batch], result dW: [nodes x prev_nodes]
        {
            MPSMatrixDescriptor *dLdz_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)batch_size
                                 columns:(NSUInteger)nodes
                                rowBytes:(NSUInteger)(nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *Ap_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)batch_size
                                 columns:(NSUInteger)prev_nodes
                                rowBytes:(NSUInteger)(prev_nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dW_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)nodes
                                 columns:(NSUInteger)prev_nodes
                                rowBytes:(NSUInteger)(prev_nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];

            MPSMatrix *dLdz_mat = [[MPSMatrix alloc] initWithBuffer:dl_dz_buf descriptor:dLdz_desc];
            MPSMatrix *Ap_mat   = [[MPSMatrix alloc] initWithBuffer:A_prev_buf descriptor:Ap_desc];
            MPSMatrix *dW_mat   = [[MPSMatrix alloc] initWithBuffer:dW_buf     descriptor:dW_desc];

            // dW = dl_dz^T * A_prev (transposeLeft=YES)
            // beta=1.0 so we accumulate (dW += ...)
            MPSMatrixMultiplication *gemm_dW =
                [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                                transposeLeft:YES
                                               transposeRight:NO
                                                   resultRows:(NSUInteger)nodes
                                                resultColumns:(NSUInteger)prev_nodes
                                              interiorColumns:(NSUInteger)batch_size
                                                        alpha:1.0
                                                         beta:1.0];

            id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
            [gemm_dW encodeToCommandBuffer:cmd leftMatrix:dLdz_mat rightMatrix:Ap_mat resultMatrix:dW_mat];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // --- bias_grad += col_sum(dl_dz) ---
        {
            id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_metal.pso_bias_grad_sum];
            [enc setBuffer:dl_dz_buf offset:0 atIndex:0];
            [enc setBuffer:db_buf    offset:0 atIndex:1];
            uint32_t bs_u32  = (uint32_t)batch_size;
            uint32_t out_u32 = (uint32_t)nodes;
            [enc setBytes:&bs_u32  length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&out_u32 length:sizeof(uint32_t) atIndex:3];
            NSUInteger tg = MIN(g_metal.pso_bias_grad_sum.maxTotalThreadsPerThreadgroup, (NSUInteger)nodes);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)nodes, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // --- Propagate: dl_dz_prev = dl_dz * W ---
        // (Skip for input layer since it has no t_batch_dl_dz)
        if (li > 1) {
            id<MTLBuffer> dl_dz_prev_buf = (__bridge id<MTLBuffer>)prev_layer->t_batch_dl_dz->gpu_buf;

            MPSMatrixDescriptor *dLdz_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)batch_size
                                 columns:(NSUInteger)nodes
                                rowBytes:(NSUInteger)(nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *W_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)nodes
                                 columns:(NSUInteger)prev_nodes
                                rowBytes:(NSUInteger)(prev_nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *prev_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:(NSUInteger)batch_size
                                 columns:(NSUInteger)prev_nodes
                                rowBytes:(NSUInteger)(prev_nodes * sizeof(float))
                                dataType:MPSDataTypeFloat32];

            MPSMatrix *dLdz_mat  = [[MPSMatrix alloc] initWithBuffer:dl_dz_buf      descriptor:dLdz_desc];
            MPSMatrix *W_mat     = [[MPSMatrix alloc] initWithBuffer:W_buf           descriptor:W_desc];
            MPSMatrix *prev_mat  = [[MPSMatrix alloc] initWithBuffer:dl_dz_prev_buf  descriptor:prev_desc];

            // dl_dz_prev = dl_dz * W (no transpose)
            MPSMatrixMultiplication *gemm_prop =
                [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                                transposeLeft:NO
                                               transposeRight:NO
                                                   resultRows:(NSUInteger)batch_size
                                                resultColumns:(NSUInteger)prev_nodes
                                              interiorColumns:(NSUInteger)nodes
                                                        alpha:1.0
                                                         beta:0.0];

            id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
            [gemm_prop encodeToCommandBuffer:cmd leftMatrix:dLdz_mat rightMatrix:W_mat resultMatrix:prev_mat];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    return 1;
}

//-----------------------------------------------------------
// gpu_dispatch_elementwise()
// Dispatch an element-wise 1-buffer kernel (for regularization/clip).
//-----------------------------------------------------------
static void gpu_dispatch_elementwise1(id<MTLComputePipelineState> pso,
                                      id<MTLBuffer> buf,
                                      NSUInteger count,
                                      float param,
                                      int param_idx)
{
    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&param length:sizeof(float) atIndex:(NSUInteger)param_idx];
    NSUInteger tg = MIN(pso.maxTotalThreadsPerThreadgroup, count);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

//-----------------------------------------------------------
// gpu_optimizer_step()
// Dispatches the appropriate optimizer kernel for each layer.
// Applies gradient clipping before update and regularization after.
//-----------------------------------------------------------
static void gpu_optimizer_step(PNetwork pnet)
{
    Optimizer_type opt = pnet->optimizer;
    float lr      = (float)pnet->learning_rate;
    float beta1   = 0.9f;
    float beta2   = 0.999f;
    float eps     = 1e-8f;
    float beta_m  = 0.9f;  // momentum beta
    float one_minus_beta_m = 0.1f;

    // Adam bias correction scalars
    float bc1 = 0.0f, bc2 = 0.0f;
    if (opt == OPT_ADAM) {
        float t = (float)pnet->train_iteration;
        bc1 = 1.0f / (1.0f - powf(beta1, t));
        bc2 = 1.0f / (1.0f - powf(beta2, t));
    }

    for (int layer = 0; layer < pnet->layer_count - 1; layer++) {
        PLayer l = &pnet->layers[layer];

        id<MTLBuffer> W_buf  = (__bridge id<MTLBuffer>)l->t_weights->gpu_buf;
        id<MTLBuffer> g_buf  = (__bridge id<MTLBuffer>)l->t_gradients->gpu_buf;
        id<MTLBuffer> b_buf  = (__bridge id<MTLBuffer>)l->t_bias->gpu_buf;
        id<MTLBuffer> bg_buf = (__bridge id<MTLBuffer>)l->t_bias_grad->gpu_buf;

        NSUInteger w_count = (NSUInteger)(l->t_weights->rows * l->t_weights->cols);
        NSUInteger b_count = (NSUInteger)(l->t_bias->cols);

        // Optional gradient clipping
        if (pnet->max_gradient > 0.0f) {
            float mg = (float)pnet->max_gradient;
            gpu_dispatch_elementwise1(g_metal.pso_gradient_clip, g_buf,  w_count, mg, 1);
            gpu_dispatch_elementwise1(g_metal.pso_gradient_clip, bg_buf, b_count, mg, 1);
        }

        switch (opt) {
          case OPT_SGD: {
            // Weights
            id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_metal.pso_sgd_update];
            [enc setBuffer:W_buf offset:0 atIndex:0];
            [enc setBuffer:g_buf offset:0 atIndex:1];
            [enc setBytes:&lr length:sizeof(float) atIndex:2];
            NSUInteger tg = MIN(g_metal.pso_sgd_update.maxTotalThreadsPerThreadgroup, w_count);
            [enc dispatchThreads:MTLSizeMake(w_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            // Biases
            id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
            [enc2 setComputePipelineState:g_metal.pso_sgd_update];
            [enc2 setBuffer:b_buf  offset:0 atIndex:0];
            [enc2 setBuffer:bg_buf offset:0 atIndex:1];
            [enc2 setBytes:&lr length:sizeof(float) atIndex:2];
            NSUInteger tg2 = MIN(g_metal.pso_sgd_update.maxTotalThreadsPerThreadgroup, b_count);
            [enc2 dispatchThreads:MTLSizeMake(b_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
            [enc2 endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            break;
          }

          case OPT_MOMENTUM: {
            id<MTLBuffer> m_buf  = (__bridge id<MTLBuffer>)l->t_m->gpu_buf;
            id<MTLBuffer> bm_buf = (__bridge id<MTLBuffer>)l->t_bias_m->gpu_buf;
            // Weights
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_momentum_update];
                [enc setBuffer:W_buf  offset:0 atIndex:0];
                [enc setBuffer:g_buf  offset:0 atIndex:1];
                [enc setBuffer:m_buf  offset:0 atIndex:2];
                [enc setBytes:&lr      length:sizeof(float) atIndex:3];
                [enc setBytes:&beta_m  length:sizeof(float) atIndex:4];
                NSUInteger tg = MIN(g_metal.pso_momentum_update.maxTotalThreadsPerThreadgroup, w_count);
                [enc dispatchThreads:MTLSizeMake(w_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            // Biases
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_momentum_update];
                [enc setBuffer:b_buf  offset:0 atIndex:0];
                [enc setBuffer:bg_buf offset:0 atIndex:1];
                [enc setBuffer:bm_buf offset:0 atIndex:2];
                [enc setBytes:&lr      length:sizeof(float) atIndex:3];
                [enc setBytes:&beta_m  length:sizeof(float) atIndex:4];
                NSUInteger tg = MIN(g_metal.pso_momentum_update.maxTotalThreadsPerThreadgroup, b_count);
                [enc dispatchThreads:MTLSizeMake(b_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            break;
          }

          case OPT_ADAGRAD: {
            id<MTLBuffer> v_buf  = (__bridge id<MTLBuffer>)l->t_v->gpu_buf;
            id<MTLBuffer> bv_buf = (__bridge id<MTLBuffer>)l->t_bias_v->gpu_buf;
            // Weights
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_adagrad_update];
                [enc setBuffer:W_buf  offset:0 atIndex:0];
                [enc setBuffer:g_buf  offset:0 atIndex:1];
                [enc setBuffer:v_buf  offset:0 atIndex:2];
                [enc setBytes:&lr  length:sizeof(float) atIndex:3];
                [enc setBytes:&eps length:sizeof(float) atIndex:4];
                NSUInteger tg = MIN(g_metal.pso_adagrad_update.maxTotalThreadsPerThreadgroup, w_count);
                [enc dispatchThreads:MTLSizeMake(w_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            // Biases
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_adagrad_update];
                [enc setBuffer:b_buf  offset:0 atIndex:0];
                [enc setBuffer:bg_buf offset:0 atIndex:1];
                [enc setBuffer:bv_buf offset:0 atIndex:2];
                [enc setBytes:&lr  length:sizeof(float) atIndex:3];
                [enc setBytes:&eps length:sizeof(float) atIndex:4];
                NSUInteger tg = MIN(g_metal.pso_adagrad_update.maxTotalThreadsPerThreadgroup, b_count);
                [enc dispatchThreads:MTLSizeMake(b_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            break;
          }

          case OPT_RMSPROP: {
            id<MTLBuffer> v_buf  = (__bridge id<MTLBuffer>)l->t_v->gpu_buf;
            id<MTLBuffer> bv_buf = (__bridge id<MTLBuffer>)l->t_bias_v->gpu_buf;
            float beta_rms = 0.9f;
            // Weights
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_rmsprop_update];
                [enc setBuffer:W_buf  offset:0 atIndex:0];
                [enc setBuffer:g_buf  offset:0 atIndex:1];
                [enc setBuffer:v_buf  offset:0 atIndex:2];
                [enc setBytes:&lr       length:sizeof(float) atIndex:3];
                [enc setBytes:&beta_rms length:sizeof(float) atIndex:4];
                [enc setBytes:&eps      length:sizeof(float) atIndex:5];
                NSUInteger tg = MIN(g_metal.pso_rmsprop_update.maxTotalThreadsPerThreadgroup, w_count);
                [enc dispatchThreads:MTLSizeMake(w_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            // Biases
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_rmsprop_update];
                [enc setBuffer:b_buf  offset:0 atIndex:0];
                [enc setBuffer:bg_buf offset:0 atIndex:1];
                [enc setBuffer:bv_buf offset:0 atIndex:2];
                [enc setBytes:&lr       length:sizeof(float) atIndex:3];
                [enc setBytes:&beta_rms length:sizeof(float) atIndex:4];
                [enc setBytes:&eps      length:sizeof(float) atIndex:5];
                NSUInteger tg = MIN(g_metal.pso_rmsprop_update.maxTotalThreadsPerThreadgroup, b_count);
                [enc dispatchThreads:MTLSizeMake(b_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            break;
          }

          case OPT_ADAM: {
            id<MTLBuffer> m_buf  = (__bridge id<MTLBuffer>)l->t_m->gpu_buf;
            id<MTLBuffer> v_buf  = (__bridge id<MTLBuffer>)l->t_v->gpu_buf;
            id<MTLBuffer> bm_buf = (__bridge id<MTLBuffer>)l->t_bias_m->gpu_buf;
            id<MTLBuffer> bv_buf = (__bridge id<MTLBuffer>)l->t_bias_v->gpu_buf;
            // Weights
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_adam_update];
                [enc setBuffer:W_buf  offset:0 atIndex:0];
                [enc setBuffer:g_buf  offset:0 atIndex:1];
                [enc setBuffer:m_buf  offset:0 atIndex:2];
                [enc setBuffer:v_buf  offset:0 atIndex:3];
                [enc setBytes:&lr    length:sizeof(float) atIndex:4];
                [enc setBytes:&beta1 length:sizeof(float) atIndex:5];
                [enc setBytes:&beta2 length:sizeof(float) atIndex:6];
                [enc setBytes:&eps   length:sizeof(float) atIndex:7];
                [enc setBytes:&bc1   length:sizeof(float) atIndex:8];
                [enc setBytes:&bc2   length:sizeof(float) atIndex:9];
                NSUInteger tg = MIN(g_metal.pso_adam_update.maxTotalThreadsPerThreadgroup, w_count);
                [enc dispatchThreads:MTLSizeMake(w_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            // Biases
            {
                id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:g_metal.pso_adam_update];
                [enc setBuffer:b_buf  offset:0 atIndex:0];
                [enc setBuffer:bg_buf offset:0 atIndex:1];
                [enc setBuffer:bm_buf offset:0 atIndex:2];
                [enc setBuffer:bv_buf offset:0 atIndex:3];
                [enc setBytes:&lr    length:sizeof(float) atIndex:4];
                [enc setBytes:&beta1 length:sizeof(float) atIndex:5];
                [enc setBytes:&beta2 length:sizeof(float) atIndex:6];
                [enc setBytes:&eps   length:sizeof(float) atIndex:7];
                [enc setBytes:&bc1   length:sizeof(float) atIndex:8];
                [enc setBytes:&bc2   length:sizeof(float) atIndex:9];
                NSUInteger tg = MIN(g_metal.pso_adam_update.maxTotalThreadsPerThreadgroup, b_count);
                [enc dispatchThreads:MTLSizeMake(b_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }
            break;
          }

          default:
            break;
        }

        // Apply L2 regularization to weights
        if (pnet->l2_lambda > 0.0f) {
            float decay = 1.0f - (float)(pnet->learning_rate * pnet->l2_lambda);
            gpu_dispatch_elementwise1(g_metal.pso_l2_regularize, W_buf, w_count, decay, 1);
        }

        // Apply L1 regularization to weights
        if (pnet->l1_lambda > 0.0f) {
            float delta = (float)(pnet->learning_rate * pnet->l1_lambda);
            gpu_dispatch_elementwise1(g_metal.pso_l1_regularize, W_buf, w_count, delta, 1);
        }
    }
}

//-----------------------------------------------------------
// metal_sync_weights()
// Download all layer weights and biases from GPU to CPU.
// Call after ann_train_network() completes to make the trained
// weights available for ann_predict(), ann_save_network(), etc.
//-----------------------------------------------------------
static void metal_sync_weights(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized) return;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++) {
        PLayer l = &pnet->layers[layer];

        if (l->t_weights && l->t_weights->gpu_buf) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)l->t_weights->gpu_buf;
            size_t bytes = (size_t)(l->t_weights->rows * l->t_weights->cols) * sizeof(real);
            memcpy(l->t_weights->values, [buf contents], bytes);
        }
        if (l->t_bias && l->t_bias->gpu_buf) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)l->t_bias->gpu_buf;
            size_t bytes = (size_t)(l->t_bias->cols) * sizeof(real);
            memcpy(l->t_bias->values, [buf contents], bytes);
        }
    }
}

//-----------------------------------------------------------
// metal_train_batch()
// Top-level GPU training hook called from ann_train_network().
//
// Steps:
//   1. Lazy-alloc training buffers on first call
//   2. Zero gradient buffers (CPU memset on shared memory)
//   3. Copy batch inputs to layer[0] GPU buffer
//   4. GPU forward pass (saving pre-activations)
//   5. Compute delta = T - Y and loss on CPU (reads shared mem)
//   6. Write delta to output layer t_batch_dl_dz GPU buf
//   7. GPU backward pass
//   8. Increment train_iteration (for Adam bias correction)
//   9. GPU optimizer step
//
// Returns 1 if GPU handled the batch, 0 to fall back to CPU.
//-----------------------------------------------------------
static int metal_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out)
{
    if (!pnet || !g_metal.initialized) return 0;
    if (!pnet->layers[0].t_weights || !pnet->layers[0].t_weights->gpu_buf) return 0;
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

    // --- Step 2: Zero gradient buffers (CPU memset on shared memory — no kernel needed) ---
    for (int layer = 0; layer < pnet->layer_count - 1; layer++) {
        PLayer l = &pnet->layers[layer];
        if (l->t_gradients && l->t_gradients->gpu_buf) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)l->t_gradients->gpu_buf;
            memset([buf contents], 0, (size_t)(l->t_gradients->rows * l->t_gradients->cols) * sizeof(real));
        }
        if (l->t_bias_grad && l->t_bias_grad->gpu_buf) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)l->t_bias_grad->gpu_buf;
            memset([buf contents], 0, (size_t)(l->t_bias_grad->cols) * sizeof(real));
        }
    }

    // --- Step 3: Copy batch inputs to GPU (CPU writes to shared buffer) ---
    {
        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)pnet->layers[0].t_batch_values->gpu_buf;
        size_t nbytes = (size_t)(batch_size * input_nodes) * sizeof(real);
        if (!in_buf || ![in_buf contents]) return 0;
        memcpy([in_buf contents], pnet->layers[0].t_batch_values->values, nbytes);
    }

    // --- Step 4: GPU forward pass ---
    if (!gpu_forward_training(pnet, batch_size)) return 0;

    // --- Step 5: Compute delta and loss on CPU (reads output from shared memory) ---
    {
        id<MTLBuffer> Y_buf = (__bridge id<MTLBuffer>)
            pnet->layers[output_layer_idx].t_batch_values->gpu_buf;
        id<MTLBuffer> delta_buf = (__bridge id<MTLBuffer>)
            pnet->layers[output_layer_idx].t_batch_dl_dz->gpu_buf;

        if (!Y_buf || !delta_buf) return 0;
        float *Y     = (float *)[Y_buf     contents];
        float *delta = (float *)[delta_buf contents];
        float *T     = batch_targets->values;

        real total_loss = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            int offset = b * output_nodes;
            for (int n = 0; n < output_nodes; n++) {
                float y = Y[offset + n];
                float t = T[offset + n];
                delta[offset + n] = t - y;

                if (pnet->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY) {
                    if (y > 1e-7f) total_loss -= t * logf(y);
                } else {
                    float diff = y - t;
                    total_loss += diff * diff;
                }
            }
        }
        if (pnet->loss_type == LOSS_MSE)
            total_loss /= (real)(batch_size * output_nodes);
        else
            total_loss /= (real)batch_size;

        if (loss_out) *loss_out = total_loss;
    }

    // --- Step 7: GPU backward pass (delta already in output delta_buf) ---
    if (!gpu_backward_pass(pnet, batch_size)) return 0;

    // --- Step 8: Increment train_iteration (Adam bias correction) ---
    pnet->train_iteration++;

    // --- Step 9: GPU optimizer step ---
    gpu_optimizer_step(pnet);

    return 1;
}

//-----------------------------------------------------------
// Metal backend vtable instance.
// ann_gpu_init() (in ann.c) sets g_gpu_backend = &metal_backend.
//-----------------------------------------------------------
GpuBackend metal_backend = {
    tensor_metal_init,       /* init           */
    metal_upload_network,    /* upload_network */
    metal_free_network,      /* free_network   */
    metal_sync_weights,      /* sync_weights   */
    tensor_metal_release_buffer, /* release_buffer */
    metal_eval_single,       /* eval_single    */
    metal_eval_batch,        /* eval_batch     */
    metal_train_batch,       /* train_batch    */
};

#endif /* USE_METAL */
