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
// APPLE METAL GPU BACKEND
//================================================================================================
// Provides GPU-accelerated inference and training using Apple Metal and Metal Performance
// Shaders (MPS), for dense, Conv2D, MaxPool2D and Flatten layers.
//
// Usage pattern:
//   1. ann_gpu_init()                  - initialize device + pipeline
//   2. ann_gpu_upload_network(pnet)    - upload weights/biases to GPU once
//   3. ann_predict(pnet, inputs, out)  - inference dispatches to GPU automatically
//      OR ann_predict_batch(...)       - batch inference (recommended for throughput)
//   4. ann_gpu_free_network(pnet)      - release GPU buffers
//
// Architecture:
//   - MPSMatrixMultiplication: Y = X * W^T for dense layers; conv layers run one GEMM per
//     batch over im2col columns
//   - Custom Metal kernels (compiled at runtime from kShaderSource below): activations, bias
//     broadcast, softmax, im2col/col2im, max pooling, optimizers
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
// Training steps that can be in flight at once (see metal_train_batch)
#define TRAIN_SLOTS 2

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
    // Conv2D and MaxPool2D
    id<MTLComputePipelineState> pso_conv_im2col;
    id<MTLComputePipelineState> pso_conv_output;
    id<MTLComputePipelineState> pso_conv_grad_rows;
    id<MTLComputePipelineState> pso_conv_bias_grad;
    id<MTLComputePipelineState> pso_conv_weight_grad_lanes;
    id<MTLComputePipelineState> pso_conv_weight_grad_lanes_vec4;
    id<MTLComputePipelineState> pso_sum_partials;
    id<MTLComputePipelineState> pso_conv_col2im;
    id<MTLComputePipelineState> pso_maxpool_forward;
    id<MTLComputePipelineState> pso_maxpool_backward;
    id<MTLComputePipelineState> pso_maxpool_backward_tiled;
    id<MTLComputePipelineState> pso_conv_direct_forward;
    id<MTLComputePipelineState> pso_conv_direct_input_grad;
    id<MTLComputePipelineState> pso_conv_direct_weight_grad;
    id<MTLComputePipelineState> pso_conv_grouped_forward;
    id<MTLComputePipelineState> pso_conv_grouped_input_grad;
    id<MTLComputePipelineState> pso_conv_grouped_weight_grad;
    id<MTLComputePipelineState> pso_upsample_forward;
    id<MTLComputePipelineState> pso_upsample_backward;
    id<MTLComputePipelineState> pso_global_avgpool_forward;
    id<MTLComputePipelineState> pso_global_avgpool_backward;
    id<MTLComputePipelineState> pso_global_maxpool_forward;
    id<MTLComputePipelineState> pso_global_maxpool_backward;
    id<MTLComputePipelineState> pso_bn_stats;
    id<MTLComputePipelineState> pso_bn_forward_train;
    id<MTLComputePipelineState> pso_bn_forward_infer;
    id<MTLComputePipelineState> pso_bn_grad;
    id<MTLComputePipelineState> pso_bn_input_grad;
    id<MTLComputePipelineState> pso_embedding_forward;
    id<MTLComputePipelineState> pso_embedding_weight_grad;
    id<MTLComputePipelineState> pso_mask_steps;
    id<MTLComputePipelineState> pso_copy_buffer;
    id<MTLComputePipelineState> pso_transpose;
    // Transposed copies of GEMM right-hand operands, by source buffer (see
    // encode_gemm)
    NSMapTable<id<MTLBuffer>, id<MTLBuffer>> *transposed;
    // MPS GEMM kernels by shape, reused across batches
    NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *gemm_cache;
    // The open compute encoder and its command buffer, see compute_encoder()
    id<MTLComputeCommandEncoder> enc;
    id<MTLCommandBuffer> enc_cmd;
    // Per-layer work buffers (conv columns, pool argmax), see layer_scratch()
    NSMutableDictionary<NSNumber *, id<MTLBuffer>> *scratch;
    // The padding masks of the pass being encoded, by layer (nil without
    // any), and the steps each keeps over the batch; see set_pad_masks()
    NSDictionary<NSNumber *, id<MTLBuffer>> *pad;
    NSMutableDictionary<NSNumber *, NSNumber *> *pad_kept;
    // The embedding's id index for the step being encoded (layer 1)
    id<MTLBuffer> emb_unique, emb_offsets, emb_positions;
    uint32_t emb_count;
    // Training steps in flight: metal_train_batch() commits a step without
    // waiting, so the CPU prepares the next batch while the GPU runs this one.
    // Each slot has its own copies of what the CPU writes for a step (inputs,
    // targets, dropout masks) and reads back (per-row losses).
    struct {
        id<MTLCommandBuffer> cmd;       // nil when the slot is free
        id<MTLBuffer> inputs, targets, row_loss;
        NSMutableDictionary<NSNumber *, id<MTLBuffer>> *masks;  // by layer
        NSMutableDictionary<NSNumber *, id<MTLBuffer>> *pad;    // padding masks, by layer
        id<MTLBuffer> emb_unique, emb_offsets, emb_positions;   // the embedding's ids, see ann_embedding_index
        uint32_t emb_count;
        int batch_size;
        int divisor;                    // batch loss = sum of row losses / divisor
    } slots[TRAIN_SLOTS];
    int next_slot;                      // the slot the next step uses (the older pending one)
    real loss_sum;                      // losses of the steps finished since metal_train_finish()
    int train_failed;                   // a step since metal_train_finish() did not complete
    int initialized;
    int training_buffers_ready;   // 1 once gpu_alloc_training_buffers() succeeds
    int training_batch_size;      // batch size for which training buffers are allocated
} MetalContext;

static MetalContext g_metal = {0};

//-----------------------------------------------------------
// Wait for a slot's training step, then add its batch loss (averaged as on
// the CPU) to g_metal.loss_sum, or flag the failure
//-----------------------------------------------------------
static void finish_slot(int s)
{
    id<MTLCommandBuffer> cmd = g_metal.slots[s].cmd;
    if (!cmd)
        return;
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusCompleted) {
        const float *row_loss = (const float *)[g_metal.slots[s].row_loss contents];
        real total = 0.0f;
        for (int b = 0; b < g_metal.slots[s].batch_size; b++)
            total += row_loss[b];
        g_metal.loss_sum += total / (real)g_metal.slots[s].divisor;
    } else
        g_metal.train_failed = 1;
    g_metal.slots[s].cmd = nil;
}

//-----------------------------------------------------------
// Wait for every training step in flight, oldest first. Anything that reads
// or replaces GPU state the steps use (weights, training buffers) calls this
// first.
//-----------------------------------------------------------
static void finish_all_slots(void)
{
    for (int i = 0; i < TRAIN_SLOTS; i++)
        finish_slot((g_metal.next_slot + i) % TRAIN_SLOTS);
}

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
        // An activation applied as part of another kernel; act is an
        // Activation_type (the ACT_* values are defined ahead of this source),
        // and anything else, softmax included, passes x through. Same math as
        // the activation_* kernels below.
        "inline float activate(float x, uint act) {\n"
        "    switch (act) {\n"
        "    case ACT_SIGMOID: return 1.0f / (1.0f + exp(-x));\n"
        "    case ACT_RELU: return max(0.0f, x);\n"
        "    case ACT_LEAKY_RELU: return x > 0.0f ? x : 0.01f * x;\n"
        "    case ACT_TANH: return tanh(x);\n"
        "    case ACT_SOFTSIGN: return x / (1.0f + fabs(x));\n"
        "    case ACT_GELU: return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));\n"
        "    default: return x; } }\n"
        // The activation's derivative, from a (z for GELU) as the deriv_*
        // kernels below compute it; 1 for anything else
        "inline float derivative(float v, uint act) {\n"
        "    switch (act) {\n"
        "    case ACT_SIGMOID: return v * (1.0f - v);\n"
        "    case ACT_RELU: return v > 0.0f ? 1.0f : 0.0f;\n"
        "    case ACT_LEAKY_RELU: return v > 0.0f ? 1.0f : 0.01f;\n"
        "    case ACT_TANH: return 1.0f - v * v;\n"
        "    case ACT_SOFTSIGN: { float d = 1.0f - fabs(v); return d * d; }\n"
        "    case ACT_GELU: {\n"
        "        float t = tanh(0.7978845608f * (v + 0.044715f * v * v * v));\n"
        "        return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * v * v); }\n"
        "    default: return 1.0f; } }\n"
        // y = activate(y + bias), first saving y + bias in z when save_z
        "kernel void bias_add(\n"
        "    device float *buf [[buffer(0)]],\n"
        "    device const float *bias [[buffer(1)]],\n"
        "    device float *z [[buffer(2)]],\n"
        "    constant uint2 &act_save_z [[buffer(3)]],\n"
        "    uint2 gid [[thread_position_in_grid]],\n"
        "    uint2 dims [[threads_per_grid]]) {\n"
        "    uint i = gid.y * dims.x + gid.x;\n"
        "    float v = buf[i] + bias[gid.x];\n"
        "    if (act_save_z.y) z[i] = v;\n"
        "    buf[i] = activate(v, act_save_z.x); }\n"
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
        // buf0 = dl_dz [batch*nodes], buf1 = activations, or for GELU the
        // values before activation (z) [batch*nodes]
        // Each kernel multiplies dl_dz[i] by the local derivative.
        // ----------------------------------------------------------------
        "kernel void deriv_sigmoid(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    float ai = a[gid]; dl_dz[gid] *= ai * (1.0f - ai); }\n"
        // ReLU and leaky ReLU: a > 0 exactly when z > 0
        "kernel void deriv_relu(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    dl_dz[gid] *= (a[gid] > 0.0f ? 1.0f : 0.0f); }\n"
        "kernel void deriv_leaky_relu(\n"
        "    device float *dl_dz [[buffer(0)]],\n"
        "    device const float *a [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    dl_dz[gid] *= (a[gid] > 0.0f ? 1.0f : 0.01f); }\n"
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
        // Output layer: delta = t - y and each row's loss term, one
        // threadgroup (a power of two, at most 256 threads) per row.
        // mode 0 = MSE (squared error), 1 = categorical cross-entropy,
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
        "    uint row [[threadgroup_position_in_grid]],\n"
        "    uint tid [[thread_position_in_threadgroup]],\n"
        "    uint tn [[threads_per_threadgroup]]) {\n"
        "    threadgroup float part[256];\n"
        "    float s = 0.0f;\n"
        "    for (uint n = tid; n < cols; n += tn) {\n"
        "        uint i = row * cols + n;\n"
        "        float yi = y[i], ti = t[i];\n"
        "        delta[i] = ti - yi;\n"
        "        if (mode == 0) { float d = yi - ti; s += d * d; }\n"
        "        else {\n"
        "            float c = clamp(yi, 1e-7f, 1.0f - 1e-7f);\n"
        "            s += mode == 2 ? -(ti * log(c) + (1.0f - ti) * log(1.0f - c)) : -ti * log(c);\n"
        "        } }\n"
        "    part[tid] = s;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint w = tn / 2; w > 0; w /= 2) {\n"
        "        if (tid < w) part[tid] += part[tid + w];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
        "    if (tid == 0) row_loss[row] = part[0]; }\n"
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
        "    for (uint n = 0; n < cols; n++) d[o + n] = y[o + n] * (d[o + n] - dot); }\n"
        // ----------------------------------------------------------------
        // Conv2D and MaxPool2D. Activations are [batch][C][H][W]; WindowDims
        // describes the layer (C,H,W in; F,OH,OW out; K kernel, S stride,
        // P padding for the height, KW, SW, PW for the width). A convolution
        // is a GEMM over im2col columns stored one row per output position:
        // cols [batch*OH*OW x C*K*KW], column index (c*K + i)*KW + j, as the
        // CPU path orders them.
        // ----------------------------------------------------------------
        "kernel void copy_buffer(\n"
        "    device const float *src [[buffer(0)]],\n"
        "    device float *dst [[buffer(1)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    dst[gid] = src[gid]; }\n"
        // dst [cols x rows] = src [rows x cols]^T; grid (cols, rows)
        "kernel void transpose(\n"
        "    device const float *src [[buffer(0)]],\n"
        "    device float *dst [[buffer(1)]],\n"
        "    uint2 gid [[thread_position_in_grid]],\n"
        "    uint2 dims [[threads_per_grid]]) {\n"
        "    dst[gid.x * dims.y + gid.y] = src[gid.y * dims.x + gid.x]; }\n"
        "struct WindowDims { uint C, H, W, F, OH, OW, K, S, P, batch, KW, SW, PW; };\n"
        // one thread per (sample, channel, output position) writes that
        // position's K*K column values; grid (OW, OH, batch*C)
        "kernel void conv_im2col(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *cols [[buffer(1)]],\n"
        "    constant WindowDims &d [[buffer(2)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.C, c = gid.z % d.C, kk = d.K * d.KW;\n"
        "    device const float *xs = x + gid.z * d.H * d.W;\n"
        "    device float *out = cols + ((b * d.OH + gid.y) * d.OW + gid.x) * d.C * kk + c * kk;\n"
        "    int ih0 = int(gid.y * d.S) - int(d.P), iw0 = int(gid.x * d.SW) - int(d.PW);\n"
        "    for (uint i = 0; i < d.K; i++) {\n"
        "        int ih = ih0 + int(i);\n"
        "        bool row_in = ih >= 0 && ih < int(d.H);\n"
        "        for (uint j = 0; j < d.KW; j++) {\n"
        "            int iw = iw0 + int(j);\n"
        "            out[i * d.KW + j] = row_in && iw >= 0 && iw < int(d.W) ? xs[ih * int(d.W) + iw] : 0.0f; } } }\n"
        // rows [batch*P x F] + bias -> y [batch][F][P], activated as in
        // bias_add (z gets the values before activation when save_z);
        // grid (P, F, batch)
        "kernel void conv_output(\n"
        "    device const float *rows [[buffer(0)]],\n"
        "    device const float *bias [[buffer(1)]],\n"
        "    device float *y [[buffer(2)]],\n"
        "    device float *z [[buffer(3)]],\n"
        "    constant uint2 &act_save_z [[buffer(4)]],\n"
        "    uint3 gid [[thread_position_in_grid]],\n"
        "    uint3 dims [[threads_per_grid]]) {\n"
        "    uint pos_n = dims.x, f_n = dims.y, i = (gid.z * f_n + gid.y) * pos_n + gid.x;\n"
        "    float v = rows[(gid.z * pos_n + gid.x) * f_n + gid.y] + bias[gid.y];\n"
        "    if (act_save_z.y) z[i] = v;\n"
        "    y[i] = activate(v, act_save_z.x); }\n"
        // dz [batch][F][P] times the activation's derivative (act, from av:
        // the layer's A, or Z for GELU) -> rows [batch*P x F]; grid (P, F, batch)
        "kernel void conv_grad_rows(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *rows [[buffer(1)]],\n"
        "    device const float *av [[buffer(2)]],\n"
        "    constant uint &act [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]],\n"
        "    uint3 dims [[threads_per_grid]]) {\n"
        "    uint pos_n = dims.x, f_n = dims.y, i = (gid.z * f_n + gid.y) * pos_n + gid.x;\n"
        "    rows[(gid.z * pos_n + gid.x) * f_n + gid.y] = dz[i] * derivative(av[i], act); }\n"
        // db[f] += sum of the dz rows' column f; one threadgroup (a power of
        // two, at most 256 threads) per filter
        "kernel void conv_bias_grad(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device float *db [[buffer(1)]],\n"
        "    constant WindowDims &d [[buffer(2)]],\n"
        "    uint f [[threadgroup_position_in_grid]],\n"
        "    uint t [[thread_position_in_threadgroup]],\n"
        "    uint n [[threads_per_threadgroup]]) {\n"
        "    threadgroup float part[256];\n"
        "    float s = 0.0f;\n"
        "    for (uint i = t; i < d.batch * d.OH * d.OW; i += n)\n"
        "        s += dz_rows[i * d.F + f];\n"
        "    part[t] = s;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint w = n / 2; w > 0; w /= 2) {\n"
        "        if (t < w) part[t] += part[t + w];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
        "    if (t == 0) db[f] += part[0]; }\n"
        // Reading im2col columns without building them: a ColWalk follows row
        // r's (sample, oh, ow) as r advances, and col_input gives column
        // (c, i, j) of that row from the input x (for conv_direct_weight_grad)
        "struct ColWalk { uint b, oh, ow; };\n"
        "inline ColWalk col_walk_start(uint r, constant WindowDims &d) {\n"
        "    uint pos = d.OH * d.OW; ColWalk w;\n"
        "    w.b = r / pos; w.oh = (r % pos) / d.OW; w.ow = r % d.OW; return w; }\n"
        "inline void col_walk_advance(thread ColWalk &w, uint step, constant WindowDims &d) {\n"
        "    w.ow += step;\n"
        "    while (w.ow >= d.OW) { w.ow -= d.OW; if (++w.oh == d.OH) { w.oh = 0; w.b++; } } }\n"
        "inline float col_input(device const float *x, constant WindowDims &d, thread const ColWalk &w,\n"
        "                       uint c, uint i, uint j) {\n"
        "    int ih = int(w.oh * d.S + i) - int(d.P), iw = int(w.ow * d.SW + j) - int(d.PW);\n"
        "    return ih >= 0 && ih < int(d.H) && iw >= 0 && iw < int(d.W)\n"
        "        ? x[((w.b * d.C + c) * d.H + uint(ih)) * d.W + uint(iw)] : 0.0f; }\n"
        // dW [F x P] += dz_rows^T * cols (P = C*K*K), a sum over batch*OH*OW rows
        // with few outputs (F*P < 8192), and db [F] += column sums of dz_rows,
        // taken as column P of cols, all ones. Threadgroups (p blocks over P + 1
        // columns, f blocks of 16, row chunks) of (PX, Y) threads: a thread takes
        // one p, so a SIMD group reads cols rows contiguously, keeps the sums for
        // its 16 filters, and takes every Y-th row of the chunk (n = rows, rows
        // per chunk, F, P); the threadgroup then adds up its Y sums. With one
        // chunk the sums are added to dW and db, else stored in partial[chunk]
        // (dW's F*P sums, then db's F) for sum_partials. The _vec4 version, for
        // F % 4 == 0, reads dz_rows as float4.
        "kernel void conv_weight_grad_lanes(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *cols [[buffer(1)]],\n"
        "    device float *out [[buffer(2)]],\n"
        "    device float *db [[buffer(3)]],\n"
        "    constant uint4 &n [[buffer(4)]],\n"
        "    uint3 tg [[threadgroup_position_in_grid]],\n"
        "    uint3 tgs [[threadgroups_per_grid]],\n"
        "    uint3 tid [[thread_position_in_threadgroup]],\n"
        "    uint3 tpg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float red[16 * 256];\n"
        "    uint F = n.z, P = n.w, PX = tpg.x, Y = tpg.y;\n"
        "    uint p = tg.x * PX + tid.x, f0 = tg.y * 16, fb = min(16u, F - f0);\n"
        "    uint r1 = min((tg.z + 1) * n.y, n.x);\n"
        "    float acc[16];\n"
        "    for (uint i = 0; i < 16; i++) acc[i] = 0.0f;\n"
        "    if (p <= P)\n"
        "        for (uint r = tg.z * n.y + tid.y; r < r1; r += Y) {\n"
        "            float c = p < P ? cols[r * P + p] : 1.0f;\n"
        "            device const float *g = dz_rows + r * F + f0;\n"
        "            for (uint i = 0; i < 16; i++)\n"
        "                if (i < fb) acc[i] += g[i] * c; }\n"
        "    for (uint i = 0; i < 16; i++) red[(tid.y * 16 + i) * PX + tid.x] = acc[i];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint o = tid.y * PX + tid.x; o < 16 * PX; o += PX * Y) {\n"
        "        uint i = o / PX, x = o % PX, pp = tg.x * PX + x, f = f0 + i;\n"
        "        if (pp > P || i >= fb) continue;\n"
        "        float s = 0.0f;\n"
        "        for (uint y = 0; y < Y; y++) s += red[(y * 16 + i) * PX + x];\n"
        "        if (tgs.z > 1) out[tg.z * F * (P + 1) + (pp < P ? f * P + pp : F * P + f)] = s;\n"
        "        else if (pp < P) out[f * P + pp] += s;\n"
        "        else db[f] += s; } }\n"
        "kernel void conv_weight_grad_lanes_vec4(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *cols [[buffer(1)]],\n"
        "    device float *out [[buffer(2)]],\n"
        "    device float *db [[buffer(3)]],\n"
        "    constant uint4 &n [[buffer(4)]],\n"
        "    uint3 tg [[threadgroup_position_in_grid]],\n"
        "    uint3 tgs [[threadgroups_per_grid]],\n"
        "    uint3 tid [[thread_position_in_threadgroup]],\n"
        "    uint3 tpg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float4 red[4 * 256];\n"
        "    uint F4 = n.z / 4, P = n.w, PX = tpg.x, Y = tpg.y;\n"
        "    uint p = tg.x * PX + tid.x, fb4 = min(4u, F4 - tg.y * 4);\n"
        "    uint r1 = min((tg.z + 1) * n.y, n.x);\n"
        "    float4 acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };\n"
        "    if (p <= P) {\n"
        "        device const float4 *g = (device const float4 *)dz_rows + tg.y * 4;\n"
        "        if (fb4 == 4)\n"
        "            for (uint r = tg.z * n.y + tid.y; r < r1; r += Y) {\n"
        "                float c = p < P ? cols[r * P + p] : 1.0f;\n"
        "                acc[0] += g[r * F4] * c; acc[1] += g[r * F4 + 1] * c;\n"
        "                acc[2] += g[r * F4 + 2] * c; acc[3] += g[r * F4 + 3] * c; }\n"
        "        else\n"
        "            for (uint r = tg.z * n.y + tid.y; r < r1; r += Y) {\n"
        "                float c = p < P ? cols[r * P + p] : 1.0f;\n"
        "                for (uint i = 0; i < fb4; i++) acc[i] += g[r * F4 + i] * c; } }\n"
        "    for (uint i = 0; i < 4; i++) red[(tid.y * 4 + i) * PX + tid.x] = acc[i];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint o = tid.y * PX + tid.x; o < 4 * PX; o += PX * Y) {\n"
        "        uint i = o / PX, x = o % PX, pp = tg.x * PX + x;\n"
        "        if (pp > P || i >= fb4) continue;\n"
        "        float4 s = 0.0f;\n"
        "        for (uint y = 0; y < Y; y++) s += red[(y * 4 + i) * PX + x];\n"
        "        for (uint j = 0; j < 4; j++) {\n"
        "            uint f = tg.y * 16 + i * 4 + j;\n"
        "            if (tgs.z > 1) out[tg.z * n.z * (P + 1) + (pp < P ? f * P + pp : n.z * P + f)] = s[j];\n"
        "            else if (pp < P) out[f * P + pp] += s[j];\n"
        "            else db[f] += s[j]; } } }\n"
        // dW and db for a direct conv (F <= 4), without im2col columns: one
        // threadgroup (a power of two, at most 256 threads) per column p of the
        // P + 1 (column P is the bias, all ones) and row chunk (n = rows, rows
        // per chunk). Its threads take consecutive rows, so a SIMD group reads
        // x and dz_rows contiguously; each keeps a sum per filter, and the
        // threadgroup adds them up into dW and db, or into partial[chunk] for
        // sum_partials when there are chunks (laid out as for the lanes kernels).
        "kernel void conv_direct_weight_grad(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *x [[buffer(1)]],\n"
        "    device float *out [[buffer(2)]],\n"
        "    device float *db [[buffer(3)]],\n"
        "    constant uint2 &n [[buffer(4)]],\n"
        "    constant WindowDims &d [[buffer(5)]],\n"
        "    uint2 tg [[threadgroup_position_in_grid]],\n"
        "    uint2 tgs [[threadgroups_per_grid]],\n"
        "    uint2 tid [[thread_position_in_threadgroup]],\n"
        "    uint2 tpg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float4 red[256];\n"
        "    uint t = tid.x, T = tpg.x, kk = d.K * d.KW, P = d.C * kk, p = tg.x;\n"
        "    uint ci = p / kk, ki = p % kk / d.KW, kj = p % d.KW;\n"
        "    uint r = tg.y * n.y + t, r1 = min((tg.y + 1) * n.y, n.x);\n"
        "    ColWalk w = col_walk_start(r, d);\n"
        "    float4 acc = 0.0f;\n"
        "    for (; r < r1; r += T) {\n"
        "        float c = p < P ? col_input(x, d, w, ci, ki, kj) : 1.0f;\n"
        "        col_walk_advance(w, T, d);\n"
        "        device const float *g = dz_rows + r * d.F;\n"
        "        for (uint f = 0; f < 4; f++)\n"
        "            if (f < d.F) acc[f] += g[f] * c; }\n"
        "    red[t] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = T / 2; s > 0; s /= 2) {\n"
        "        if (t < s) red[t] += red[t + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
        "    if (t >= d.F) return;\n"
        "    float v = red[0][t];\n"
        "    if (tgs.y > 1) out[tg.y * d.F * (P + 1) + (p < P ? t * P + p : d.F * P + t)] = v;\n"
        "    else if (p < P) out[t * P + p] += v;\n"
        "    else db[t] += v; }\n"
        // adds up the chunks' partial sums, each dW [F x P] followed by db [F],
        // into dW and db (n = chunks, F*P); grid F*(P + 1)
        "kernel void sum_partials(\n"
        "    device const float *partial [[buffer(0)]],\n"
        "    device float *dw [[buffer(1)]],\n"
        "    device float *db [[buffer(2)]],\n"
        "    constant uint2 &n [[buffer(3)]],\n"
        "    uint gid [[thread_position_in_grid]],\n"
        "    uint count [[threads_per_grid]]) {\n"
        "    float s = 0.0f;\n"
        "    for (uint c = 0; c < n.x; c++) s += partial[c * count + gid];\n"
        "    if (gid < n.y) dw[gid] += s;\n"
        "    else db[gid - n.y] += s; }\n"
        // dx[b][c][ih][iw] = sum of the column gradients that read it (a
        // gather, so no atomics); grid (W, H, batch*C)
        "kernel void conv_col2im(\n"
        "    device const float *dcols [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    constant WindowDims &d [[buffer(2)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.C, c = gid.z % d.C, kk = d.K * d.KW, ckk = d.C * kk;\n"
        "    float s = 0.0f;\n"
        "    for (uint i = 0; i < d.K; i++) {\n"
        "        int ty = int(gid.y + d.P) - int(i);\n"
        "        if (ty < 0) continue;\n"
        "        uint oh = uint(ty);\n"
        "        if (d.S > 1) { if (oh % d.S) continue; oh /= d.S; }\n"
        "        if (oh >= d.OH) continue;\n"
        "        for (uint j = 0; j < d.KW; j++) {\n"
        "            int tx = int(gid.x + d.PW) - int(j);\n"
        "            if (tx < 0) continue;\n"
        "            uint ow = uint(tx);\n"
        "            if (d.SW > 1) { if (ow % d.SW) continue; ow /= d.SW; }\n"
        "            if (ow >= d.OW) continue;\n"
        "            uint row = (b * d.OH + oh) * d.OW + ow;\n"
        "            s += dcols[row * ckk + c * kk + i * d.KW + j]; } }\n"
        "    dx[(gid.z * d.H + gid.y) * d.W + gid.x] = s; }\n"
        // Direct convolution for layers with few filters, where a GEMM over
        // im2col columns has too few outputs to be efficient: one thread per
        // output sums its window (in the im2col order, c then i then j), adds
        // the bias and activates as conv_output does; grid (OW, OH, batch*F)
        "kernel void conv_direct_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device const float *w [[buffer(1)]],\n"
        "    device const float *bias [[buffer(2)]],\n"
        "    device float *y [[buffer(3)]],\n"
        "    device float *z [[buffer(4)]],\n"
        "    constant WindowDims &d [[buffer(5)]],\n"
        "    constant uint2 &act_save_z [[buffer(6)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.F, f = gid.z % d.F, kk = d.K * d.KW;\n"
        "    int ih0 = int(gid.y * d.S) - int(d.P), iw0 = int(gid.x * d.SW) - int(d.PW);\n"
        "    float s = 0.0f;\n"
        "    for (uint c = 0; c < d.C; c++) {\n"
        "        device const float *xs = x + (b * d.C + c) * d.H * d.W;\n"
        "        device const float *wf = w + (f * d.C + c) * kk;\n"
        "        for (uint i = 0; i < d.K; i++) {\n"
        "            int ih = ih0 + int(i);\n"
        "            if (ih < 0 || ih >= int(d.H)) continue;\n"
        "            for (uint j = 0; j < d.KW; j++) {\n"
        "                int iw = iw0 + int(j);\n"
        "                if (iw >= 0 && iw < int(d.W)) s += xs[ih * int(d.W) + iw] * wf[i * d.KW + j]; } } }\n"
        "    uint o = (gid.z * d.OH + gid.y) * d.OW + gid.x;\n"
        "    float v = s + bias[f];\n"
        "    if (act_save_z.y) z[o] = v;\n"
        "    y[o] = activate(v, act_save_z.x); }\n"
        // dx[b][c][ih][iw] = sum of dz_rows * W over the outputs whose
        // windows read it (conv_col2im without the column gradients);
        // grid (W, H, batch*C)
        "kernel void conv_direct_input_grad(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *w [[buffer(1)]],\n"
        "    device float *dx [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.C, c = gid.z % d.C, kk = d.K * d.KW;\n"
        "    float s = 0.0f;\n"
        "    for (uint i = 0; i < d.K; i++) {\n"
        "        int ty = int(gid.y + d.P) - int(i);\n"
        "        if (ty < 0) continue;\n"
        "        uint oh = uint(ty);\n"
        "        if (d.S > 1) { if (oh % d.S) continue; oh /= d.S; }\n"
        "        if (oh >= d.OH) continue;\n"
        "        for (uint j = 0; j < d.KW; j++) {\n"
        "            int tx = int(gid.x + d.PW) - int(j);\n"
        "            if (tx < 0) continue;\n"
        "            uint ow = uint(tx);\n"
        "            if (d.SW > 1) { if (ow % d.SW) continue; ow /= d.SW; }\n"
        "            if (ow >= d.OW) continue;\n"
        "            device const float *g = dz_rows + ((b * d.OH + oh) * d.OW + ow) * d.F;\n"
        "            for (uint f = 0; f < d.F; f++) s += g[f] * w[(f * d.C + c) * kk + i * d.KW + j]; } }\n"
        "    dx[(gid.z * d.H + gid.y) * d.W + gid.x] = s; }\n"
        // Each output takes its window's largest input (the first on ties,
        // scanning as the CPU does) and records its index within the sample;
        // grid (OW, OH, batch*C)
        "kernel void maxpool_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    device uint *arg [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    device const float *xs = x + gid.z * d.H * d.W;\n"
        "    uint row0 = gid.y * d.S * d.W + gid.x * d.SW, best = row0;\n"
        "    for (uint i = 0; i < d.K; i++)\n"
        "        for (uint j = 0; j < d.KW; j++) {\n"
        "            uint idx = row0 + i * d.W + j;\n"
        "            if (xs[idx] > xs[best]) best = idx; }\n"
        "    uint out = (gid.z * d.OH + gid.y) * d.OW + gid.x;\n"
        "    y[out] = xs[best];\n"
        "    arg[out] = (gid.z % d.C) * d.H * d.W + best; }\n"
        // dx = sum of dz over the outputs that took this input (a gather over
        // the windows covering it); grid (W, H, batch*C)
        "kernel void maxpool_backward(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    device const uint *arg [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint me = ((gid.z % d.C) * d.H + gid.y) * d.W + gid.x;\n"
        "    uint oh0 = gid.y + 1 > d.K ? (gid.y + 1 - d.K + d.S - 1) / d.S : 0;\n"
        "    uint ow0 = gid.x + 1 > d.KW ? (gid.x + 1 - d.KW + d.SW - 1) / d.SW : 0;\n"
        "    uint oh1 = min(gid.y / d.S + 1, d.OH), ow1 = min(gid.x / d.SW + 1, d.OW);\n"
        "    float s = 0.0f;\n"
        "    for (uint oh = oh0; oh < oh1; oh++)\n"
        "        for (uint ow = ow0; ow < ow1; ow++) {\n"
        "            uint out = (gid.z * d.OH + oh) * d.OW + ow;\n"
        "            if (arg[out] == me) s += dz[out]; }\n"
        "    dx[(gid.z * d.H + gid.y) * d.W + gid.x] = s; }\n"
        // The same when K == S and KW == SW: the windows tile the input, so each output
        // writes its window, dz at the input it took and zero elsewhere; the
        // last row and column of outputs also zero the inputs no window
        // covers. grid (OW, OH, batch*C)
        "kernel void maxpool_backward_tiled(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    device const uint *arg [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint out = (gid.z * d.OH + gid.y) * d.OW + gid.x;\n"
        "    uint best = arg[out] - (gid.z % d.C) * d.H * d.W;\n"
        "    float g = dz[out];\n"
        "    uint h0 = gid.y * d.S, w0 = gid.x * d.SW;\n"
        "    uint h1 = gid.y + 1 == d.OH ? d.H : h0 + d.K, w1 = gid.x + 1 == d.OW ? d.W : w0 + d.KW;\n"
        "    device float *plane = dx + gid.z * d.H * d.W;\n"
        "    for (uint h = h0; h < h1; h++)\n"
        "        for (uint w = w0; w < w1; w++)\n"
        "            plane[h * d.W + w] = h * d.W + w == best ? g : 0.0f; }\n"
        // Nearest-neighbour upsampling by K (the scale): each input fills a
        // K x K block; grid (OW, OH, batch*C)
        "kernel void upsample_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    constant WindowDims &d [[buffer(2)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    y[(gid.z * d.OH + gid.y) * d.OW + gid.x] = x[(gid.z * d.H + gid.y / d.K) * d.W + gid.x / d.K]; }\n"
        // dx = sum of dz over the input's block, in the CPU's order; grid (W, H, batch*C)
        "kernel void upsample_backward(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    constant WindowDims &d [[buffer(2)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    device const float *block = dz + (gid.z * d.OH + gid.y * d.K) * d.OW + gid.x * d.K;\n"
        "    float s = 0.0f;\n"
        "    for (uint i = 0; i < d.K; i++)\n"
        "        for (uint j = 0; j < d.K; j++) s += block[i * d.OW + j];\n"
        "    dx[(gid.z * d.H + gid.y) * d.W + gid.x] = s; }\n"
        // Global average pooling: each channel's mean, summed in the CPU's
        // order; grid (batch*C). Over a masked sequence (masked = 1, mask
        // [batch][H*W], 1 kept) the mean of the kept steps, 0 if none.
        "inline uint kept_steps(device const float *m, uint area) {\n"
        "    uint n = 0;\n"
        "    for (uint i = 0; i < area; i++) n += m[i] != 0.0f;\n"
        "    return n; }\n"
        "kernel void global_avgpool_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    device const float *mask [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    constant uint &masked [[buffer(4)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    uint area = d.H * d.W;\n"
        "    device const float *plane = x + gid * area, *m = mask + (gid / d.C) * area;\n"
        "    float s = 0.0f;\n"
        "    if (!masked) {\n"
        "        for (uint i = 0; i < area; i++) s += plane[i];\n"
        "        y[gid] = s / (float)area;\n"
        "        return; }\n"
        "    for (uint i = 0; i < area; i++) if (m[i] != 0.0f) s += plane[i];\n"
        "    uint kept = kept_steps(m, area);\n"
        "    y[gid] = kept ? s / (float)kept : 0.0f; }\n"
        // dx = the channel's dz / (H*W) (masked: / the kept steps, 0 at the
        // masked ones); grid (W, H, batch*C)
        "kernel void global_avgpool_backward(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    device const float *mask [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    constant uint &masked [[buffer(4)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint area = d.H * d.W, i = gid.y * d.W + gid.x;\n"
        "    float g = dz[gid.z] / (float)area;\n"
        "    if (masked) {\n"
        "        device const float *m = mask + (gid.z / d.C) * area;\n"
        "        uint kept = kept_steps(m, area);\n"
        "        g = m[i] != 0.0f ? dz[gid.z] / (float)kept : 0.0f; }\n"
        "    dx[gid.z * area + i] = g; }\n"
        // Global max pooling: each channel's largest value (the first on
        // ties, as the CPU), recording its index in the plane; grid
        // (batch*C). Masked: of the kept steps; none gives 0 and index ~0u.
        "kernel void global_maxpool_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    device uint *arg [[buffer(2)]],\n"
        "    device const float *mask [[buffer(3)]],\n"
        "    constant WindowDims &d [[buffer(4)]],\n"
        "    constant uint &masked [[buffer(5)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    uint area = d.H * d.W, best = ~0u;\n"
        "    device const float *plane = x + gid * area, *m = mask + (gid / d.C) * area;\n"
        "    for (uint i = 0; i < area; i++)\n"
        "        if ((!masked || m[i] != 0.0f) && (best == ~0u || plane[i] > plane[best])) best = i;\n"
        "    y[gid] = best == ~0u ? 0.0f : plane[best];\n"
        "    arg[gid] = best; }\n"
        // dx = the channel's dz at the input it took, else 0; grid (W, H, batch*C)
        "kernel void global_maxpool_backward(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device float *dx [[buffer(1)]],\n"
        "    device const uint *arg [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint i = gid.y * d.W + gid.x;\n"
        "    dx[gid.z * d.H * d.W + i] = arg[gid.z] == i ? dz[gid.z] : 0.0f; }\n"
        // ----------------------------------------------------------------
        // Batch norm over [batch][C][area] (area = H*W, 1 after a dense
        // layer). stats holds 4 rows of C: the batch mean, 1 / sqrt(var +
        // eps), then mean(dz) and mean(dz * x_hat) for the input gradient.
        // ----------------------------------------------------------------
        // masked = 1: mask [batch][area] (1 kept) leaves steps out of the
        // statistics, kept of them in all; x_hat and dx are 0 at the others
        "struct BnDims { uint C, area, batch; float momentum, eps; uint masked, kept; };\n"
        "inline float bn_reduce(threadgroup float *part, float s, uint tid, uint tn) {\n"
        "    part[tid] = s;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint w = tn / 2; w > 0; w /= 2) {\n"
        "        if (tid < w) part[tid] += part[tid + w];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
        "    float r = part[0];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    return r; }\n"
        // The batch's mean and variance per channel (a threadgroup each),
        // then the running statistics (unbiased variance, as the CPU's)
        "kernel void bn_stats(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *stats [[buffer(1)]],\n"
        "    device float *mean_run [[buffer(2)]],\n"
        "    device float *var_run [[buffer(3)]],\n"
        "    device const float *mask [[buffer(4)]],\n"
        "    constant BnDims &d [[buffer(5)]],\n"
        "    uint c [[threadgroup_position_in_grid]],\n"
        "    uint tid [[thread_position_in_threadgroup]],\n"
        "    uint tn [[threads_per_threadgroup]]) {\n"
        "    threadgroup float part[256];\n"
        "    uint all = d.batch * d.area, n = d.masked ? d.kept : all, nodes = d.C * d.area;\n"
        "    float s = 0.0f;\n"
        "    for (uint j = tid; j < all; j += tn)\n"
        "        if (!d.masked || mask[j] != 0.0f) s += x[(j / d.area) * nodes + c * d.area + j % d.area];\n"
        "    float total = bn_reduce(part, s, tid, tn), mean = n ? total / (float)n : 0.0f;\n"
        "    s = 0.0f;\n"
        "    for (uint j = tid; j < all; j += tn)\n"
        "        if (!d.masked || mask[j] != 0.0f) { float v = x[(j / d.area) * nodes + c * d.area + j % d.area] - mean; s += v * v; }\n"
        "    float sq = bn_reduce(part, s, tid, tn);\n"
        "    if (tid == 0 && n == 0) {\n"
        "        stats[c] = 0.0f;\n"
        "        stats[d.C + c] = 1.0f / sqrt(d.eps); }\n"
        "    else if (tid == 0) {\n"
        "        float var = sq / (float)n;\n"
        "        stats[c] = mean;\n"
        "        stats[d.C + c] = 1.0f / sqrt(var + d.eps);\n"
        "        mean_run[c] = (1.0f - d.momentum) * mean_run[c] + d.momentum * mean;\n"
        "        var_run[c] = (1.0f - d.momentum) * var_run[c] + d.momentum * (n > 1 ? sq / (float)(n - 1) : var); } }\n"
        // y = gamma * x_hat + beta with the batch's statistics; x_hat is kept for backward
        "kernel void bn_forward_train(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    device float *x_hat [[buffer(2)]],\n"
        "    device const float *stats [[buffer(3)]],\n"
        "    device const float *gamma [[buffer(4)]],\n"
        "    device const float *beta [[buffer(5)]],\n"
        "    device const float *mask [[buffer(6)]],\n"
        "    constant BnDims &d [[buffer(7)]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        "    uint c = (i / d.area) % d.C;\n"
        "    bool kept = !d.masked || mask[(i / (d.C * d.area)) * d.area + i % d.area] != 0.0f;\n"
        "    float h = kept ? (x[i] - stats[c]) * stats[d.C + c] : 0.0f;\n"
        "    x_hat[i] = h;\n"
        "    y[i] = gamma[c] * h + beta[c]; }\n"
        // prediction: the running statistics
        "kernel void bn_forward_infer(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device float *y [[buffer(1)]],\n"
        "    device const float *gamma [[buffer(2)]],\n"
        "    device const float *beta [[buffer(3)]],\n"
        "    device const float *mean [[buffer(4)]],\n"
        "    device const float *var [[buffer(5)]],\n"
        "    constant BnDims &d [[buffer(6)]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        "    uint c = (i / d.area) % d.C;\n"
        "    y[i] = (x[i] - mean[c]) * (gamma[c] / sqrt(var[c] + d.eps)) + beta[c]; }\n"
        // gamma and beta gradients per channel (a threadgroup each), and the
        // means the input gradient needs
        "kernel void bn_grad(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device const float *x_hat [[buffer(1)]],\n"
        "    device float *stats [[buffer(2)]],\n"
        "    device float *d_gamma [[buffer(3)]],\n"
        "    device float *d_beta [[buffer(4)]],\n"
        "    constant BnDims &d [[buffer(5)]],\n"
        "    uint c [[threadgroup_position_in_grid]],\n"
        "    uint tid [[thread_position_in_threadgroup]],\n"
        "    uint tn [[threads_per_threadgroup]]) {\n"
        "    threadgroup float part[256];\n"
        "    uint n = d.batch * d.area, nodes = d.C * d.area, kept = d.masked ? d.kept : n;\n"
        "    float s = 0.0f, t = 0.0f;\n"
        "    for (uint j = tid; j < n; j += tn) {\n"
        "        uint i = (j / d.area) * nodes + c * d.area + j % d.area;\n"
        "        s += dz[i]; t += dz[i] * x_hat[i]; }\n"
        "    float sum = bn_reduce(part, s, tid, tn);\n"
        "    float dot = bn_reduce(part, t, tid, tn);\n"
        "    if (tid == 0) {\n"
        "        d_gamma[c] += dot;\n"
        "        d_beta[c] += sum;\n"
        "        stats[2 * d.C + c] = kept ? sum / (float)kept : 0.0f;\n"
        "        stats[3 * d.C + c] = kept ? dot / (float)kept : 0.0f; } }\n"
        // dx = gamma * inv_std * (dz - mean(dz) - x_hat * mean(dz * x_hat))
        "kernel void bn_input_grad(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device const float *x_hat [[buffer(1)]],\n"
        "    device const float *stats [[buffer(2)]],\n"
        "    device const float *gamma [[buffer(3)]],\n"
        "    device float *dx [[buffer(4)]],\n"
        "    device const float *mask [[buffer(5)]],\n"
        "    constant BnDims &d [[buffer(6)]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        "    uint c = (i / d.area) % d.C;\n"
        "    bool kept = !d.masked || mask[(i / (d.C * d.area)) * d.area + i % d.area] != 0.0f;\n"
        "    dx[i] = kept ? gamma[c] * stats[d.C + c] * (dz[i] - stats[2 * d.C + c] - x_hat[i] * stats[3 * d.C + c]) : 0.0f; }\n"
        // ----------------------------------------------------------------
        // Embedding (vocab V, dim D, L steps): ids [batch][L] (reals,
        // rounded), W [V][D], output [batch][D][L]. The padding id (pad,
        // -1 for none) and ids outside the vocabulary give zeros.
        // ----------------------------------------------------------------
        "struct EmbDims { uint V, D, L; int pad; };\n"
        "kernel void embedding_forward(\n"
        "    device const float *ids [[buffer(0)]],\n"
        "    device const float *w [[buffer(1)]],\n"
        "    device float *y [[buffer(2)]],\n"
        "    constant EmbDims &d [[buffer(3)]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        "    uint b = i / (d.D * d.L), r = i % (d.D * d.L), dim = r / d.L, t = r % d.L;\n"
        "    float id = floor(ids[b * d.L + t] + 0.5f);\n"
        "    bool ok = id >= 0.0f && id < (float)d.V && (int)id != d.pad;\n"
        "    y[i] = ok ? w[(uint)id * d.D + dim] : 0.0f; }\n"
        // dW[unique[u]][dim] += the sum of dz over the positions holding
        // that id (positions[offsets[u] .. offsets[u+1]), b * L + t), in the
        // CPU's order: no two threads write the same weight; grid (D, ids)
        "kernel void embedding_weight_grad(\n"
        "    device const float *dz [[buffer(0)]],\n"
        "    device const uint *unique [[buffer(1)]],\n"
        "    device const uint *offsets [[buffer(2)]],\n"
        "    device const uint *positions [[buffer(3)]],\n"
        "    device float *dw [[buffer(4)]],\n"
        "    constant EmbDims &d [[buffer(5)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    uint dim = gid.x, u = gid.y;\n"
        "    float s = 0.0f;\n"
        "    for (uint k = offsets[u]; k < offsets[u + 1]; k++) {\n"
        "        uint p = positions[k];\n"
        "        s += dz[(p / d.L) * d.D * d.L + dim * d.L + p % d.L]; }\n"
        "    dw[unique[u] * d.D + dim] += s; }\n"
        // zero a masked layer's values or gradients at its masked steps:
        // v [batch][C][steps], mask [batch][steps]
        "kernel void mask_steps(\n"
        "    device float *v [[buffer(0)]],\n"
        "    device const float *mask [[buffer(1)]],\n"
        "    constant uint2 &steps_nodes [[buffer(2)]],\n"
        "    uint i [[thread_position_in_grid]]) {\n"
        "    uint steps = steps_nodes.x, nodes = steps_nodes.y;\n"
        "    if (mask[(i / nodes) * steps + (i % nodes) % steps] == 0.0f) v[i] = 0.0f; }\n"
        // ----------------------------------------------------------------
        // Grouped convolution (G groups; depthwise when G = C), computed
        // directly: filter f reads channels [g*C/G, (g+1)*C/G) for
        // g = f / (F/G), and W is [F][C/G][K][K].
        // ----------------------------------------------------------------
        // one thread per output sums its window, adds the bias and
        // activates; grid (OW, OH, batch*F)
        "kernel void conv_grouped_forward(\n"
        "    device const float *x [[buffer(0)]],\n"
        "    device const float *w [[buffer(1)]],\n"
        "    device const float *bias [[buffer(2)]],\n"
        "    device float *y [[buffer(3)]],\n"
        "    device float *z [[buffer(4)]],\n"
        "    constant WindowDims &d [[buffer(5)]],\n"
        "    constant uint2 &act_save_z [[buffer(6)]],\n"
        "    constant uint &groups [[buffer(7)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.F, f = gid.z % d.F, kk = d.K * d.KW, cg = d.C / groups;\n"
        "    uint c0 = f / (d.F / groups) * cg;\n"
        "    int ih0 = int(gid.y * d.S) - int(d.P), iw0 = int(gid.x * d.SW) - int(d.PW);\n"
        "    float s = 0.0f;\n"
        "    for (uint c = 0; c < cg; c++) {\n"
        "        device const float *xs = x + (b * d.C + c0 + c) * d.H * d.W;\n"
        "        device const float *wf = w + (f * cg + c) * kk;\n"
        "        for (uint i = 0; i < d.K; i++) {\n"
        "            int ih = ih0 + int(i);\n"
        "            if (ih < 0 || ih >= int(d.H)) continue;\n"
        "            for (uint j = 0; j < d.KW; j++) {\n"
        "                int iw = iw0 + int(j);\n"
        "                if (iw >= 0 && iw < int(d.W)) s += xs[ih * int(d.W) + iw] * wf[i * d.KW + j]; } } }\n"
        "    uint o = (gid.z * d.OH + gid.y) * d.OW + gid.x;\n"
        "    float v = s + bias[f];\n"
        "    if (act_save_z.y) z[o] = v;\n"
        "    y[o] = activate(v, act_save_z.x); }\n"
        // dx[b][c][ih][iw] = sum of dz_rows * W over the group's filters and
        // the outputs whose windows read it; grid (W, H, batch*C)
        "kernel void conv_grouped_input_grad(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *w [[buffer(1)]],\n"
        "    device float *dx [[buffer(2)]],\n"
        "    constant WindowDims &d [[buffer(3)]],\n"
        "    constant uint &groups [[buffer(4)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint b = gid.z / d.C, c = gid.z % d.C, kk = d.K * d.KW, cg = d.C / groups, fg = d.F / groups;\n"
        "    uint f0 = c / cg * fg, ci = c % cg;\n"
        "    float s = 0.0f;\n"
        "    for (uint i = 0; i < d.K; i++) {\n"
        "        int ty = int(gid.y + d.P) - int(i);\n"
        "        if (ty < 0) continue;\n"
        "        uint oh = uint(ty);\n"
        "        if (d.S > 1) { if (oh % d.S) continue; oh /= d.S; }\n"
        "        if (oh >= d.OH) continue;\n"
        "        for (uint j = 0; j < d.KW; j++) {\n"
        "            int tx = int(gid.x + d.PW) - int(j);\n"
        "            if (tx < 0) continue;\n"
        "            uint ow = uint(tx);\n"
        "            if (d.SW > 1) { if (ow % d.SW) continue; ow /= d.SW; }\n"
        "            if (ow >= d.OW) continue;\n"
        "            device const float *g = dz_rows + ((b * d.OH + oh) * d.OW + ow) * d.F;\n"
        "            for (uint f = f0; f < f0 + fg; f++) s += g[f] * w[(f * cg + ci) * kk + i * d.KW + j]; } }\n"
        "    dx[(gid.z * d.H + gid.y) * d.W + gid.x] = s; }\n"
        // dW[f][p] += sum over rows of dz_rows[r][f] * the input p reads for
        // row r (p = (c, i, j) of the filter's group; p = C/G*K*K is the bias,
        // all ones); one threadgroup (a power of two, at most 256 threads)
        // per (p, f), its threads taking every T-th row
        "kernel void conv_grouped_weight_grad(\n"
        "    device const float *dz_rows [[buffer(0)]],\n"
        "    device const float *x [[buffer(1)]],\n"
        "    device float *dw [[buffer(2)]],\n"
        "    device float *db [[buffer(3)]],\n"
        "    constant WindowDims &d [[buffer(4)]],\n"
        "    constant uint &groups [[buffer(5)]],\n"
        "    uint2 tg [[threadgroup_position_in_grid]],\n"
        "    uint2 tid [[thread_position_in_threadgroup]],\n"
        "    uint2 tpg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float part[256];\n"
        "    uint t = tid.x, T = tpg.x;\n"
        "    uint kk = d.K * d.KW, cg = d.C / groups, P = cg * kk, p = tg.x, f = tg.y;\n"
        "    uint c = f / (d.F / groups) * cg + p / kk, ki = p % kk / d.KW, kj = p % d.KW;\n"
        "    uint rows = d.batch * d.OH * d.OW;\n"
        "    ColWalk walk = col_walk_start(t, d);\n"
        "    float s = 0.0f;\n"
        "    for (uint r = t; r < rows; r += T) {\n"
        "        s += dz_rows[r * d.F + f] * (p < P ? col_input(x, d, walk, c, ki, kj) : 1.0f);\n"
        "        col_walk_advance(walk, T, d); }\n"
        "    part[t] = s;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint w = T / 2; w > 0; w /= 2) {\n"
        "        if (t < w) part[t] += part[t + w];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
        "    if (t == 0) { if (p < P) dw[f * P + p] += part[0]; else db[f] += part[0]; } }\n";

    NSError *err = nil;
    NSString *src = [NSString stringWithFormat:
        @"#define ACT_SIGMOID %d\n#define ACT_RELU %d\n#define ACT_LEAKY_RELU %d\n"
        @"#define ACT_TANH %d\n#define ACT_SOFTSIGN %d\n#define ACT_GELU %d\n%s",
        ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_LEAKY_RELU, ACTIVATION_TANH,
        ACTIVATION_SOFTSIGN, ACTIVATION_GELU, kShaderSource];
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

    g_metal.pso_conv_im2col      = make_pso(g_metal.library, "conv_im2col");
    g_metal.pso_conv_output      = make_pso(g_metal.library, "conv_output");
    g_metal.pso_conv_grad_rows   = make_pso(g_metal.library, "conv_grad_rows");
    g_metal.pso_conv_bias_grad   = make_pso(g_metal.library, "conv_bias_grad");
    g_metal.pso_conv_col2im      = make_pso(g_metal.library, "conv_col2im");
    g_metal.pso_conv_weight_grad_lanes = make_pso(g_metal.library, "conv_weight_grad_lanes");
    g_metal.pso_conv_weight_grad_lanes_vec4 = make_pso(g_metal.library, "conv_weight_grad_lanes_vec4");
    g_metal.pso_sum_partials     = make_pso(g_metal.library, "sum_partials");
    g_metal.pso_maxpool_forward  = make_pso(g_metal.library, "maxpool_forward");
    g_metal.pso_maxpool_backward = make_pso(g_metal.library, "maxpool_backward");
    g_metal.pso_maxpool_backward_tiled = make_pso(g_metal.library, "maxpool_backward_tiled");
    g_metal.pso_conv_direct_forward = make_pso(g_metal.library, "conv_direct_forward");
    g_metal.pso_conv_direct_input_grad = make_pso(g_metal.library, "conv_direct_input_grad");
    g_metal.pso_conv_direct_weight_grad = make_pso(g_metal.library, "conv_direct_weight_grad");
    g_metal.pso_conv_grouped_forward = make_pso(g_metal.library, "conv_grouped_forward");
    g_metal.pso_conv_grouped_input_grad = make_pso(g_metal.library, "conv_grouped_input_grad");
    g_metal.pso_conv_grouped_weight_grad = make_pso(g_metal.library, "conv_grouped_weight_grad");
    g_metal.pso_upsample_forward = make_pso(g_metal.library, "upsample_forward");
    g_metal.pso_upsample_backward = make_pso(g_metal.library, "upsample_backward");
    g_metal.pso_global_avgpool_forward = make_pso(g_metal.library, "global_avgpool_forward");
    g_metal.pso_global_avgpool_backward = make_pso(g_metal.library, "global_avgpool_backward");
    g_metal.pso_global_maxpool_forward = make_pso(g_metal.library, "global_maxpool_forward");
    g_metal.pso_global_maxpool_backward = make_pso(g_metal.library, "global_maxpool_backward");
    g_metal.pso_bn_stats         = make_pso(g_metal.library, "bn_stats");
    g_metal.pso_bn_forward_train = make_pso(g_metal.library, "bn_forward_train");
    g_metal.pso_bn_forward_infer = make_pso(g_metal.library, "bn_forward_infer");
    g_metal.pso_bn_grad          = make_pso(g_metal.library, "bn_grad");
    g_metal.pso_bn_input_grad    = make_pso(g_metal.library, "bn_input_grad");
    g_metal.pso_embedding_forward = make_pso(g_metal.library, "embedding_forward");
    g_metal.pso_embedding_weight_grad = make_pso(g_metal.library, "embedding_weight_grad");
    g_metal.pso_mask_steps       = make_pso(g_metal.library, "mask_steps");
    g_metal.pso_copy_buffer      = make_pso(g_metal.library, "copy_buffer");
    g_metal.pso_transpose        = make_pso(g_metal.library, "transpose");

    if (!g_metal.pso_conv_im2col || !g_metal.pso_conv_output || !g_metal.pso_conv_grad_rows ||
        !g_metal.pso_conv_bias_grad || !g_metal.pso_conv_col2im ||
        !g_metal.pso_conv_weight_grad_lanes || !g_metal.pso_conv_weight_grad_lanes_vec4 ||
        !g_metal.pso_sum_partials ||
        !g_metal.pso_maxpool_forward || !g_metal.pso_maxpool_backward ||
        !g_metal.pso_maxpool_backward_tiled ||
        !g_metal.pso_conv_direct_forward || !g_metal.pso_conv_direct_input_grad ||
        !g_metal.pso_conv_direct_weight_grad || !g_metal.pso_copy_buffer ||
        !g_metal.pso_embedding_forward || !g_metal.pso_embedding_weight_grad || !g_metal.pso_mask_steps ||
        !g_metal.pso_conv_grouped_forward || !g_metal.pso_conv_grouped_input_grad ||
        !g_metal.pso_conv_grouped_weight_grad ||
        !g_metal.pso_upsample_forward || !g_metal.pso_upsample_backward ||
        !g_metal.pso_global_avgpool_forward || !g_metal.pso_global_avgpool_backward ||
        !g_metal.pso_global_maxpool_forward || !g_metal.pso_global_maxpool_backward ||
        !g_metal.pso_bn_stats || !g_metal.pso_bn_forward_train || !g_metal.pso_bn_forward_infer ||
        !g_metal.pso_bn_grad || !g_metal.pso_bn_input_grad ||
        !g_metal.pso_transpose)
        return 0;

    g_metal.gemm_cache = [NSMutableDictionary dictionary];
    g_metal.transposed = [NSMapTable strongToStrongObjectsMapTable];
    g_metal.scratch = [NSMutableDictionary dictionary];
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
        // Drop its transposed copy, if any, which also holds a reference
        [g_metal.transposed removeObjectForKey:(__bridge id<MTLBuffer>)gpu_buf];
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
// Kernels share one compute encoder until something else must be encoded
// (an MPS GEMM, a blit, or the commit), which calls end_compute() first.
// A batch has dozens of small kernels and each encoder has a fixed cost, so
// this matters for small networks. Encoders dispatch serially, so each kernel
// still sees the writes of the ones before it.
//-----------------------------------------------------------
static id<MTLComputeCommandEncoder> compute_encoder(id<MTLCommandBuffer> cmd)
{
    if (g_metal.enc && g_metal.enc_cmd == cmd)
        return g_metal.enc;
    if (g_metal.enc)
        [g_metal.enc endEncoding];
    g_metal.enc = [cmd computeCommandEncoder];
    g_metal.enc_cmd = cmd;
    return g_metal.enc;
}

static void end_compute(void)
{
    if (g_metal.enc)
        [g_metal.enc endEncoding];
    g_metal.enc = nil;
    g_metal.enc_cmd = nil;
}

//-----------------------------------------------------------
// Commit a command buffer and wait for it; 1 if it completed
//-----------------------------------------------------------
static int commit_and_wait(id<MTLCommandBuffer> cmd)
{
    end_compute();
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
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
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];

    NSUInteger tgsize = pso.maxTotalThreadsPerThreadgroup;
    if (tgsize > count) tgsize = count;

    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgsize, 1, 1)];
    }

//-----------------------------------------------------------
// Dispatch bias_add kernel: buf[b*cols + n] = activate(buf[b*cols + n] + bias[n])
// buf_buf: [batch x cols] output matrix
// bias_buf: [cols] bias vector
// act: the activation to apply (softmax is left to dispatch_softmax)
// z_buf: where to save the values before activation, or nil
//-----------------------------------------------------------
static void dispatch_bias_add(id<MTLCommandBuffer> cmd,
                              id<MTLBuffer> buf_buf,
                              id<MTLBuffer> bias_buf,
                              NSUInteger batch_size,
                              NSUInteger cols,
                              Activation_type act,
                              id<MTLBuffer> z_buf)
{
    uint32_t act_save_z[2] = { (uint32_t)act, z_buf != nil };
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:g_metal.pso_bias_add];
    [enc setBuffer:buf_buf  offset:0 atIndex:0];
    [enc setBuffer:bias_buf offset:0 atIndex:1];
    [enc setBuffer:z_buf ? z_buf : buf_buf offset:0 atIndex:2];
    [enc setBytes:act_save_z length:sizeof(act_save_z) atIndex:3];

    MTLSize gridSize = MTLSizeMake(cols, batch_size, 1);
    NSUInteger tgw = g_metal.pso_bias_add.maxTotalThreadsPerThreadgroup;
    if (tgw > cols) tgw = cols;
    MTLSize tgSize = MTLSizeMake(tgw, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
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
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:g_metal.pso_softmax];
    [enc setBuffer:buf_buf offset:0 atIndex:0];
    [enc setBytes:&cols length:sizeof(uint32_t) atIndex:1];

    // One thread per row — each thread sequentially computes softmax for its row.
    // This is correct for any output size and avoids parallel reduction complexity.
    [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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
static id<MTLBuffer> gpu_scratch(id<MTLBuffer> buf, size_t bytes);
static void encode_grid(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                        id<MTLBuffer> __strong *bufs, int n_bufs, const void *consts, size_t consts_len,
                        MTLSize grid);

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
    // A transposed right operand (the weights, in forward passes) is copied
    // transposed first: MPS was about twice as slow with transposeRight on an
    // Intel Iris Plus 655 (e.g. 0.63 vs 0.28 ms for an 86528x8x9 conv GEMM),
    // and the copy is one small kernel in the open compute encoder
    if (trans_b) {
        id<MTLBuffer> bt = gpu_scratch([g_metal.transposed objectForKey:b], b_rows * b_cols * sizeof(float));
        if (bt) {
            [g_metal.transposed setObject:bt forKey:b];
            id<MTLBuffer> bufs[2] = { b, bt };
            encode_grid(cmd, g_metal.pso_transpose, bufs, 2, NULL, 0, MTLSizeMake(b_cols, b_rows, 1));
            b = bt;
            NSUInteger rows = b_rows;
            b_rows = b_cols;
            b_cols = rows;
            trans_b = NO;
        }
    }
    end_compute();
    [gemm_kernel(trans_a, trans_b, M, N, K, alpha, beta) encodeToCommandBuffer:cmd
                                                                    leftMatrix:gpu_matrix(a, a_rows, a_cols)
                                                                   rightMatrix:gpu_matrix(b, b_rows, b_cols)
                                                                  resultMatrix:gpu_matrix(c, M, N)];
}

//-----------------------------------------------------------
// Copy count floats between buffers (a kernel rather than a blit, so it
// stays in the open compute encoder)
//-----------------------------------------------------------
static void encode_copy(id<MTLCommandBuffer> cmd, id<MTLBuffer> src, id<MTLBuffer> dst, NSUInteger count)
{
    dispatch_binary_elementwise(cmd, g_metal.pso_copy_buffer, src, dst, count);
}

//-----------------------------------------------------------
// A conv or pool layer's geometry, laid out like the kernels' WindowDims
//-----------------------------------------------------------
typedef struct {
    uint32_t C, H, W, F, OH, OW, K, S, P, batch;
    uint32_t KW, SW, PW;    // the window's width settings (K, S and P are the height's)
} WindowDims;

static WindowDims window_dims(PNetwork pnet, int layer, int batch_size)
{
    PLayer l = &pnet->layers[layer];
    int kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0;
    ann_get_layer_window(pnet, layer, &kh, &kw, &sh, &sw, &ph, &pw);

    WindowDims d;
    d.C = (uint32_t)l->in_shape.c; d.H = (uint32_t)l->in_shape.h; d.W = (uint32_t)l->in_shape.w;
    d.F = (uint32_t)l->shape.c;    d.OH = (uint32_t)l->shape.h;   d.OW = (uint32_t)l->shape.w;
    d.K = (uint32_t)kh; d.S = (uint32_t)sh; d.P = (uint32_t)ph;
    d.KW = (uint32_t)kw; d.SW = (uint32_t)sw; d.PW = (uint32_t)pw;
    d.batch = (uint32_t)batch_size;
    return d;
}

//-----------------------------------------------------------
// Encode a kernel over a grid of up to 3 dimensions: buffers at indices
// 0..n_bufs-1, then the constants (e.g. a WindowDims) when given
//-----------------------------------------------------------
static void encode_grid(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                        id<MTLBuffer> __strong *bufs, int n_bufs, const void *consts, size_t consts_len,
                        MTLSize grid)
{
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
    if (consts)
        [enc setBytes:consts length:consts_len atIndex:(NSUInteger)n_bufs];

    NSUInteger tw = MIN(pso.threadExecutionWidth, grid.width);
    NSUInteger th = MIN(pso.maxTotalThreadsPerThreadgroup / tw, grid.height);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    }

//-----------------------------------------------------------
// Work buffers kept per layer between the forward and backward passes of a
// batch (shared by training and inference, which never overlap). Slots:
//   SCRATCH_COLS  - conv im2col columns [batch*OH*OW x C*K*K]; pool argmax
//   SCRATCH_ROWS  - conv output rows, then dZ rows [batch*OH*OW x F]
//   SCRATCH_DCOLS - conv column gradients [batch*OH*OW x C*K*K]
//   SCRATCH_PARTIAL - conv weight and bias gradient partial sums [chunks x F*(C*K*K + 1)]
//-----------------------------------------------------------
enum { SCRATCH_COLS, SCRATCH_ROWS, SCRATCH_DCOLS, SCRATCH_PARTIAL, SCRATCH_SLOTS };

static id<MTLBuffer> layer_scratch(int layer, int slot, size_t bytes)
{
    NSNumber *key = @(layer * SCRATCH_SLOTS + slot);
    id<MTLBuffer> buf = gpu_scratch(g_metal.scratch[key], bytes);
    if (buf)
        g_metal.scratch[key] = buf;
    return buf;
}

// Conv2D layers with at most this many filters convolve directly, with no
// im2col columns: a GEMM with so few outputs per column row runs slowly
// (fashion_dae's 16 -> 1 output conv at batch 32 on an Intel Iris Plus 655:
// MPS took 2.6 ms, the direct kernel 0.3 ms), and the columns are C*K*K
// times the input's size to write and read back. The weight gradient reads
// the input too (conv_direct_weight_grad), unless the layer is big enough
// for the GEMM; with 8 filters the GEMM path was as fast or faster.
#define CONV_DIRECT_FILTERS 4
// Their weight gradient's threadgroup size and the number of threadgroups
// to aim for
#define CONV_DIRECT_THREADS 128
#define CONV_DIRECT_GROUPS 256

static int conv_direct(const WindowDims *d)
{
    return d->F <= CONV_DIRECT_FILTERS;
}

//-----------------------------------------------------------
// Conv2D forward for a batch, as on the CPU but with one GEMM for the whole
// batch: rows [batch*OH*OW x F] = im2col(X) * W^T, then Y = rows + b
// rearranged to [batch][F][OH*OW]; or directly (conv_direct)
//-----------------------------------------------------------
static int encode_conv_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                               id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> Z)
{
    PLayer l = &pnet->layers[layer];
    WindowDims d = window_dims(pnet, layer, batch_size);
    NSUInteger positions = d.OH * d.OW, rows = (NSUInteger)batch_size * positions, patch = d.C * d.K * d.KW;
    uint32_t act_save_z[2] = { (uint32_t)l->activation, Z != nil };
    uint32_t groups = (uint32_t)ann_get_layer_groups(pnet, layer);
    if (groups > 1 || conv_direct(&d)) {
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        id<MTLComputePipelineState> pso = groups > 1 ? g_metal.pso_conv_grouped_forward : g_metal.pso_conv_direct_forward;
        [enc setComputePipelineState:pso];
        [enc setBuffer:X offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)l->params[0].value->gpu_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)l->params[1].value->gpu_buf offset:0 atIndex:2];
        [enc setBuffer:Y offset:0 atIndex:3];
        [enc setBuffer:Z ? Z : Y offset:0 atIndex:4];
        [enc setBytes:&d length:sizeof(d) atIndex:5];
        [enc setBytes:act_save_z length:sizeof(act_save_z) atIndex:6];
        if (groups > 1)
            [enc setBytes:&groups length:sizeof(groups) atIndex:7];
        NSUInteger tw = MIN(pso.threadExecutionWidth, (NSUInteger)d.OW);
        NSUInteger th = MIN(pso.maxTotalThreadsPerThreadgroup / tw, (NSUInteger)d.OH);
        [enc dispatchThreads:MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.F) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
        return 1;
    }

    id<MTLBuffer> cols = layer_scratch(layer, SCRATCH_COLS, rows * patch * sizeof(float));
    id<MTLBuffer> out_rows = layer_scratch(layer, SCRATCH_ROWS, rows * d.F * sizeof(float));
    if (!cols || !out_rows)
        return 0;

    id<MTLBuffer> im2col_bufs[2] = { X, cols };
    encode_grid(cmd, g_metal.pso_conv_im2col, im2col_bufs, 2, &d, sizeof(d),
                MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.C));
    encode_gemm(cmd, cols, rows, patch, NO, (__bridge id<MTLBuffer>)l->params[0].value->gpu_buf, d.F, patch, YES,
                out_rows, 1.0, 0.0);
    id<MTLBuffer> out_bufs[4] = { out_rows, (__bridge id<MTLBuffer>)l->params[1].value->gpu_buf, Y, Z ? Z : Y };
    encode_grid(cmd, g_metal.pso_conv_output, out_bufs, 4, act_save_z, sizeof(act_save_z),
                MTLSizeMake(positions, d.F, (NSUInteger)batch_size));
    return 1;
}

// Weight gradients with fewer outputs (F*C*K*K) than this use split sums
// (measured on Apple silicon: faster below ~4K outputs, slower above ~16K)
#define CONV_SPLIT_OUTPUTS 8192
// Threadgroups to aim for in the split weight gradient (measured on an Intel
// Iris Plus 655: 16 to 64 are within 25% of each other; more is slower)
#define CONV_SPLIT_GROUPS 64

//-----------------------------------------------------------
// Conv2D backward for a batch: dZ_rows = dZ * f'(z) for the activation act
// (ACTIVATION_NULL when dZ already has it), dW += dZ_rows^T * cols,
// db += sums of dZ_rows, and, when dX is given, dX = col2im(dZ_rows * W).
// cols is kept from the forward pass. A direct conv has none: its weight
// gradient reads the layer's input X (or builds cols from it for the GEMM),
// and dX is computed directly.
//-----------------------------------------------------------
static int encode_conv_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                id<MTLBuffer> X, id<MTLBuffer> dZ, id<MTLBuffer> dX, Activation_type act)
{
    PLayer l = &pnet->layers[layer];
    WindowDims d = window_dims(pnet, layer, batch_size);
    NSUInteger positions = d.OH * d.OW, rows = (NSUInteger)batch_size * positions, patch = d.C * d.K * d.KW;
    NSUInteger outputs = d.F * patch;
    id<MTLBuffer> W = (__bridge id<MTLBuffer>)l->params[0].value->gpu_buf;
    id<MTLBuffer> dz_rows = layer_scratch(layer, SCRATCH_ROWS, rows * d.F * sizeof(float));
    if (!dz_rows)
        return 0;

    // A grouped conv works from the input and dz_rows directly: a
    // threadgroup per weight (and bias) sums its rows, and each input
    // gathers its gradient from its group's filters
    uint32_t groups = (uint32_t)ann_get_layer_groups(pnet, layer);
    if (groups > 1) {
        uint32_t act_code = (uint32_t)act;
        id<MTLBuffer> av = (__bridge id<MTLBuffer>)(act == ACTIVATION_GELU ? l->t_batch_z : l->t_batch_values)->gpu_buf;
        id<MTLBuffer> row_bufs[3] = { dZ, dz_rows, av };
        encode_grid(cmd, g_metal.pso_conv_grad_rows, row_bufs, 3, &act_code, sizeof(act_code),
                    MTLSizeMake(positions, d.F, (NSUInteger)batch_size));

        NSUInteger tg = 1;
        while (tg < rows && tg < 256 && tg * 2 <= g_metal.pso_conv_grouped_weight_grad.maxTotalThreadsPerThreadgroup)
            tg *= 2;
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        [enc setComputePipelineState:g_metal.pso_conv_grouped_weight_grad];
        [enc setBuffer:dz_rows offset:0 atIndex:0];
        [enc setBuffer:X offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)l->params[0].grad->gpu_buf offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)l->params[1].grad->gpu_buf offset:0 atIndex:3];
        [enc setBytes:&d length:sizeof(d) atIndex:4];
        [enc setBytes:&groups length:sizeof(groups) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(patch / groups + 1, d.F, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        if (dX) {
            id<MTLComputePipelineState> pso = g_metal.pso_conv_grouped_input_grad;
            enc = compute_encoder(cmd);
            [enc setComputePipelineState:pso];
            [enc setBuffer:dz_rows offset:0 atIndex:0];
            [enc setBuffer:W offset:0 atIndex:1];
            [enc setBuffer:dX offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            [enc setBytes:&groups length:sizeof(groups) atIndex:4];
            NSUInteger tw = MIN(pso.threadExecutionWidth, (NSUInteger)d.W);
            NSUInteger th = MIN(pso.maxTotalThreadsPerThreadgroup / tw, (NSUInteger)d.H);
            [enc dispatchThreads:MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
        }
        return 1;
    }

    // A direct conv has no columns from the forward pass: its weight
    // gradient reads the input instead, unless it is big enough for the GEMM
    id<MTLBuffer> cols = X;
    int implicit = conv_direct(&d) && outputs < CONV_SPLIT_OUTPUTS;
    if (!implicit) {
        if (!(cols = layer_scratch(layer, SCRATCH_COLS, rows * patch * sizeof(float))))
            return 0;
        if (conv_direct(&d)) {
            id<MTLBuffer> im2col_bufs[2] = { X, cols };
            encode_grid(cmd, g_metal.pso_conv_im2col, im2col_bufs, 2, &d, sizeof(d),
                        MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.C));
        }
    }

    // the derivative reads A, or Z for GELU (the only one the forward pass saves)
    uint32_t act_code = (uint32_t)act;
    id<MTLBuffer> av = (__bridge id<MTLBuffer>)(act == ACTIVATION_GELU ? l->t_batch_z : l->t_batch_values)->gpu_buf;
    id<MTLBuffer> row_bufs[3] = { dZ, dz_rows, av };
    encode_grid(cmd, g_metal.pso_conv_grad_rows, row_bufs, 3, &act_code, sizeof(act_code),
                MTLSizeMake(positions, d.F, (NSUInteger)batch_size));

    // dW += dz_rows^T * cols and db += column sums of dz_rows. With few
    // outputs, each summing many rows, a GEMM runs with little parallelism,
    // so below CONV_SPLIT_OUTPUTS the rows are split into chunks, about
    // CONV_SPLIT_GROUPS threadgroups in all, whose partial sums are then
    // added up; the bias is one more column there. Larger layers use the
    // (tiled) MPS GEMM and a reduction per filter for the bias.
    id<MTLBuffer> dW = (__bridge id<MTLBuffer>)l->params[0].grad->gpu_buf;
    id<MTLBuffer> db = (__bridge id<MTLBuffer>)l->params[1].grad->gpu_buf;
    if (outputs >= CONV_SPLIT_OUTPUTS) {
        encode_gemm(cmd, dz_rows, rows, d.F, YES, cols, rows, patch, NO, dW, 1.0, 1.0);

        // one threadgroup per filter; the kernel's reduction needs a power of two
        NSUInteger tg = 256;
        while (tg > g_metal.pso_conv_bias_grad.maxTotalThreadsPerThreadgroup)
            tg /= 2;
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        [enc setComputePipelineState:g_metal.pso_conv_bias_grad];
        [enc setBuffer:dz_rows offset:0 atIndex:0];
        [enc setBuffer:db offset:0 atIndex:1];
        [enc setBytes:&d length:sizeof(d) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(d.F, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    } else if (implicit) {
        // a threadgroup per column and row chunk, about CONV_DIRECT_GROUPS in all
        NSUInteger columns = patch + 1, threads = CONV_DIRECT_THREADS;
        while (threads > g_metal.pso_conv_direct_weight_grad.maxTotalThreadsPerThreadgroup)
            threads /= 2;
        NSUInteger chunks = MIN(MAX(CONV_DIRECT_GROUPS / columns, (NSUInteger)1), (rows + threads - 1) / threads);
        uint32_t n[2] = { (uint32_t)rows, (uint32_t)((rows + chunks - 1) / chunks) };
        chunks = (rows + n[1] - 1) / n[1];
        uint32_t sum_n[2] = { (uint32_t)chunks, (uint32_t)outputs };
        id<MTLBuffer> partial = dW;
        if (chunks > 1 && !(partial = layer_scratch(layer, SCRATCH_PARTIAL, chunks * d.F * columns * sizeof(float))))
            return 0;
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        [enc setComputePipelineState:g_metal.pso_conv_direct_weight_grad];
        [enc setBuffer:dz_rows offset:0 atIndex:0];
        [enc setBuffer:X offset:0 atIndex:1];
        [enc setBuffer:partial offset:0 atIndex:2];
        [enc setBuffer:db offset:0 atIndex:3];
        [enc setBytes:n length:sizeof(n) atIndex:4];
        [enc setBytes:&d length:sizeof(d) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(columns, chunks, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (chunks > 1) {
            id<MTLBuffer> sum_bufs[3] = { partial, dW, db };
            encode_grid(cmd, g_metal.pso_sum_partials, sum_bufs, 3, sum_n, sizeof(sum_n),
                        MTLSizeMake(d.F * columns, 1, 1));
        }
    } else {
        id<MTLComputePipelineState> pso = d.F % 4 == 0 ? g_metal.pso_conv_weight_grad_lanes_vec4
                                                       : g_metal.pso_conv_weight_grad_lanes;
        NSUInteger columns = patch + 1;
        NSUInteger threads = MIN((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        NSUInteger px = columns <= threads ? columns : threads / 2, y = threads / px;
        NSUInteger blocks = ((columns + px - 1) / px) * ((d.F + 15) / 16);
        NSUInteger chunks = MIN(MAX(CONV_SPLIT_GROUPS / blocks, (NSUInteger)1), (rows + y - 1) / y);
        uint32_t n[4] = { (uint32_t)rows, (uint32_t)((rows + chunks - 1) / chunks), d.F, (uint32_t)patch };
        chunks = (rows + n[1] - 1) / n[1];
        uint32_t sum_n[2] = { (uint32_t)chunks, (uint32_t)outputs };
        id<MTLBuffer> partial = dW;
        if (chunks > 1 && !(partial = layer_scratch(layer, SCRATCH_PARTIAL, chunks * d.F * columns * sizeof(float))))
            return 0;
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        [enc setComputePipelineState:pso];
        [enc setBuffer:dz_rows offset:0 atIndex:0];
        [enc setBuffer:cols offset:0 atIndex:1];
        [enc setBuffer:partial offset:0 atIndex:2];
        [enc setBuffer:db offset:0 atIndex:3];
        [enc setBytes:n length:sizeof(n) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((columns + px - 1) / px, (d.F + 15) / 16, chunks)
            threadsPerThreadgroup:MTLSizeMake(px, y, 1)];
        if (chunks > 1) {
            id<MTLBuffer> sum_bufs[3] = { partial, dW, db };
            encode_grid(cmd, g_metal.pso_sum_partials, sum_bufs, 3, sum_n, sizeof(sum_n),
                        MTLSizeMake(d.F * columns, 1, 1));
        }
    }

    if (!dX)
        return 1;
    if (conv_direct(&d)) {
        id<MTLBuffer> grad_bufs[3] = { dz_rows, W, dX };
        encode_grid(cmd, g_metal.pso_conv_direct_input_grad, grad_bufs, 3, &d, sizeof(d),
                    MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
        return 1;
    }
    id<MTLBuffer> dcols = layer_scratch(layer, SCRATCH_DCOLS, rows * patch * sizeof(float));
    if (!dcols)
        return 0;
    encode_gemm(cmd, dz_rows, rows, d.F, NO, W, d.F, patch, NO, dcols, 1.0, 0.0);
    id<MTLBuffer> col2im_bufs[2] = { dcols, dX };
    encode_grid(cmd, g_metal.pso_conv_col2im, col2im_bufs, 2, &d, sizeof(d),
                MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
    return 1;
}

//-----------------------------------------------------------
// MaxPool2D forward; each output's argmax goes to the layer's scratch
//-----------------------------------------------------------
static int encode_maxpool_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                  id<MTLBuffer> X, id<MTLBuffer> Y)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> arg = layer_scratch(layer, SCRATCH_COLS,
                                      (size_t)batch_size * pnet->layers[layer].node_count * sizeof(uint32_t));
    if (!arg)
        return 0;
    id<MTLBuffer> bufs[3] = { X, Y, arg };
    encode_grid(cmd, g_metal.pso_maxpool_forward, bufs, 3, &d, sizeof(d), MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.C));
    return 1;
}

static int encode_maxpool_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                   id<MTLBuffer> dZ, id<MTLBuffer> dX)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> arg = layer_scratch(layer, SCRATCH_COLS,
                                      (size_t)batch_size * pnet->layers[layer].node_count * sizeof(uint32_t));
    if (!arg)
        return 0;
    id<MTLBuffer> bufs[3] = { dZ, dX, arg };
    if (d.K == d.S && d.KW == d.SW)
        encode_grid(cmd, g_metal.pso_maxpool_backward_tiled, bufs, 3, &d, sizeof(d),
                    MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.C));
    else
        encode_grid(cmd, g_metal.pso_maxpool_backward, bufs, 3, &d, sizeof(d),
                    MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
    return 1;
}

//-----------------------------------------------------------
// Upsample2D forward and backward: gathers, so no atomics or zeroing
//-----------------------------------------------------------
static void encode_upsample_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                    id<MTLBuffer> X, id<MTLBuffer> Y)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> bufs[2] = { X, Y };
    encode_grid(cmd, g_metal.pso_upsample_forward, bufs, 2, &d, sizeof(d), MTLSizeMake(d.OW, d.OH, (NSUInteger)batch_size * d.C));
}

static void encode_upsample_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                     id<MTLBuffer> dZ, id<MTLBuffer> dX)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> bufs[2] = { dZ, dX };
    encode_grid(cmd, g_metal.pso_upsample_backward, bufs, 2, &d, sizeof(d), MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
}

//-----------------------------------------------------------
// A global pooling kernel: bufs, then the WindowDims and whether the
// sequence is masked (the last buffer is then its mask)
//-----------------------------------------------------------
static void encode_pooled(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso, id<MTLBuffer> __strong *bufs,
                          int n_bufs, const WindowDims *d, uint32_t masked, MTLSize grid)
{
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
    [enc setBytes:d length:sizeof(*d) atIndex:(NSUInteger)n_bufs];
    [enc setBytes:&masked length:sizeof(masked) atIndex:(NSUInteger)n_bufs + 1];
    NSUInteger tw = MIN(pso.threadExecutionWidth, grid.width);
    NSUInteger th = MIN(pso.maxTotalThreadsPerThreadgroup / tw, grid.height);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

//-----------------------------------------------------------
// GlobalMaxPool2D forward and backward; each channel's argmax goes to the
// layer's scratch
//-----------------------------------------------------------
static int encode_global_maxpool(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                 id<MTLBuffer> src, id<MTLBuffer> dst, int backward)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> arg = layer_scratch(layer, SCRATCH_COLS, (size_t)batch_size * d.C * sizeof(uint32_t));
    if (!arg)
        return 0;
    if (backward) {
        id<MTLBuffer> bufs[3] = { src, dst, arg };
        encode_grid(cmd, g_metal.pso_global_maxpool_backward, bufs, 3, &d, sizeof(d),
                    MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
        return 1;
    }
    id<MTLBuffer> mask = g_metal.pad[@(layer - 1)];
    uint32_t masked = mask != nil;
    id<MTLBuffer> bufs[4] = { src, dst, arg, mask ? mask : src };
    encode_pooled(cmd, g_metal.pso_global_maxpool_forward, bufs, 4, &d, masked, MTLSizeMake((NSUInteger)batch_size * d.C, 1, 1));
    return 1;
}

//-----------------------------------------------------------
// GlobalAvgPool2D forward and backward: one thread per output, no atomics
//-----------------------------------------------------------
static void encode_global_avgpool_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                          id<MTLBuffer> X, id<MTLBuffer> Y)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> mask = g_metal.pad[@(layer - 1)];
    id<MTLBuffer> bufs[3] = { X, Y, mask ? mask : X };
    encode_pooled(cmd, g_metal.pso_global_avgpool_forward, bufs, 3, &d, mask != nil,
                  MTLSizeMake((NSUInteger)batch_size * d.C, 1, 1));
}

static void encode_global_avgpool_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                           id<MTLBuffer> dZ, id<MTLBuffer> dX)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    id<MTLBuffer> mask = g_metal.pad[@(layer - 1)];
    id<MTLBuffer> bufs[3] = { dZ, dX, mask ? mask : dZ };
    encode_pooled(cmd, g_metal.pso_global_avgpool_backward, bufs, 3, &d, mask != nil,
                  MTLSizeMake(d.W, d.H, (NSUInteger)batch_size * d.C));
}

//-----------------------------------------------------------
// BatchNorm: params are gamma, beta, running mean, running variance.
// Training keeps x_hat in SCRATCH_COLS and the per-channel statistics (4 rows
// of C, see bn_stats) in SCRATCH_PARTIAL for the backward pass.
//-----------------------------------------------------------
typedef struct {
    uint32_t C, area, batch;
    float momentum, eps;
    uint32_t masked, kept;  // over a masked sequence: its mask's kept steps
} BnDims;

static BnDims bn_dims(PNetwork pnet, int layer, int batch_size)
{
    PLayer l = &pnet->layers[layer];
    BnDims d;
    d.C = (uint32_t)l->shape.c;
    d.area = (uint32_t)(l->shape.h * l->shape.w);
    d.batch = (uint32_t)batch_size;
    d.momentum = 0.1f;
    d.eps = (float)ANN_BATCHNORM_EPSILON;
    d.masked = g_metal.pad[@(layer)] != nil;
    d.kept = d.masked ? (uint32_t)[g_metal.pad_kept[@(layer)] unsignedIntValue] : 0;
    return d;
}

// the layer's padding mask, or any buffer for an unmasked kernel to ignore
static id<MTLBuffer> pad_or(int layer, id<MTLBuffer> other)
{
    id<MTLBuffer> mask = g_metal.pad[@(layer)];
    return mask ? mask : other;
}

static id<MTLBuffer> param_buf(PLayer l, int p)
{
    return (__bridge id<MTLBuffer>)l->params[p].value->gpu_buf;
}

// a threadgroup per channel: the smallest power of two covering the channel's values, at most 256
static void encode_bn_channels(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                               id<MTLBuffer> __strong *bufs, int n_bufs, const BnDims *d)
{
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
    [enc setBytes:d length:sizeof(*d) atIndex:(NSUInteger)n_bufs];
    NSUInteger n = (NSUInteger)d->batch * d->area, tg = 1;
    while (tg < n && tg < 256 && tg * 2 <= pso.maxTotalThreadsPerThreadgroup)
        tg *= 2;
    [enc dispatchThreadgroups:MTLSizeMake(d->C, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static int encode_batchnorm_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                    id<MTLBuffer> X, id<MTLBuffer> Y, int training)
{
    PLayer l = &pnet->layers[layer];
    BnDims d = bn_dims(pnet, layer, batch_size);
    NSUInteger count = (NSUInteger)batch_size * l->node_count;

    if (!training) {
        id<MTLBuffer> bufs[6] = { X, Y, param_buf(l, 0), param_buf(l, 1), param_buf(l, 2), param_buf(l, 3) };
        encode_grid(cmd, g_metal.pso_bn_forward_infer, bufs, 6, &d, sizeof(d), MTLSizeMake(count, 1, 1));
        return 1;
    }

    id<MTLBuffer> x_hat = layer_scratch(layer, SCRATCH_COLS, count * sizeof(float));
    id<MTLBuffer> stats = layer_scratch(layer, SCRATCH_PARTIAL, (size_t)4 * d.C * sizeof(float));
    if (!x_hat || !stats)
        return 0;
    id<MTLBuffer> sbufs[5] = { X, stats, param_buf(l, 2), param_buf(l, 3), pad_or(layer, X) };
    encode_bn_channels(cmd, g_metal.pso_bn_stats, sbufs, 5, &d);
    id<MTLBuffer> bufs[7] = { X, Y, x_hat, stats, param_buf(l, 0), param_buf(l, 1), pad_or(layer, X) };
    encode_grid(cmd, g_metal.pso_bn_forward_train, bufs, 7, &d, sizeof(d), MTLSizeMake(count, 1, 1));
    return 1;
}

// gamma and beta gradients, then dX unless it is nil (layer 1)
static int encode_batchnorm_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                     id<MTLBuffer> dZ, id<MTLBuffer> dX)
{
    PLayer l = &pnet->layers[layer];
    BnDims d = bn_dims(pnet, layer, batch_size);
    NSUInteger count = (NSUInteger)batch_size * l->node_count;
    id<MTLBuffer> x_hat = layer_scratch(layer, SCRATCH_COLS, count * sizeof(float));
    id<MTLBuffer> stats = layer_scratch(layer, SCRATCH_PARTIAL, (size_t)4 * d.C * sizeof(float));
    if (!x_hat || !stats)
        return 0;

    id<MTLBuffer> gbufs[5] = { dZ, x_hat, stats, (__bridge id<MTLBuffer>)l->params[0].grad->gpu_buf,
                               (__bridge id<MTLBuffer>)l->params[1].grad->gpu_buf };
    encode_bn_channels(cmd, g_metal.pso_bn_grad, gbufs, 5, &d);
    if (dX) {
        id<MTLBuffer> bufs[6] = { dZ, x_hat, stats, param_buf(l, 0), dX, pad_or(layer, dZ) };
        encode_grid(cmd, g_metal.pso_bn_input_grad, bufs, 6, &d, sizeof(d), MTLSizeMake(count, 1, 1));
    }
    return 1;
}

//-----------------------------------------------------------
// Embedding: a gather forward; the weight gradient sums each distinct id's
// positions in the step's id index (see ann_embedding_index), so no two
// threads add to the same weight
//-----------------------------------------------------------
typedef struct {
    uint32_t V, D, L;
    int32_t pad;
} EmbDims;

static EmbDims emb_dims(PNetwork pnet, int layer)
{
    PLayer l = &pnet->layers[layer];
    EmbDims d;
    int vocab = 0, stride = 0, padding = -1;
    ann_get_layer_settings(pnet, layer, &vocab, &stride, &padding);
    d.V = (uint32_t)vocab;
    d.D = (uint32_t)l->shape.c;
    d.L = (uint32_t)l->shape.w;
    d.pad = padding;
    return d;
}

static void encode_embedding_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                     id<MTLBuffer> X, id<MTLBuffer> Y)
{
    EmbDims d = emb_dims(pnet, layer);
    id<MTLBuffer> bufs[3] = { X, param_buf(&pnet->layers[layer], 0), Y };
    encode_grid(cmd, g_metal.pso_embedding_forward, bufs, 3, &d, sizeof(d),
                MTLSizeMake((NSUInteger)batch_size * d.D * d.L, 1, 1));
}

static void encode_embedding_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, id<MTLBuffer> dZ)
{
    EmbDims d = emb_dims(pnet, layer);
    if (g_metal.emb_count == 0)
        return;     // nothing but padding
    id<MTLBuffer> bufs[5] = { dZ, g_metal.emb_unique, g_metal.emb_offsets, g_metal.emb_positions,
                              (__bridge id<MTLBuffer>)pnet->layers[layer].params[0].grad->gpu_buf };
    encode_grid(cmd, g_metal.pso_embedding_weight_grad, bufs, 5, &d, sizeof(d), MTLSizeMake(d.D, g_metal.emb_count, 1));
}

//-----------------------------------------------------------
// Padding masks (see ann_add_embedding_masked). ann.c works them out on the
// CPU from the ids; set_pad_masks() copies each masked layer's into a
// shared buffer (reusing those in reuse, a slot's, when given) and makes
// them the pass's g_metal.pad. Masked layers' values are zeroed at the
// masked steps after the activation and dropout, and their gradients before
// the activation's derivative; batch norm and the global pools read them.
// Returns 0 if a buffer could not be allocated.
//-----------------------------------------------------------
static int set_pad_masks(PNetwork pnet, const real *inputs, int rows,
                         NSMutableDictionary<NSNumber *, id<MTLBuffer>> *reuse)
{
    g_metal.pad = nil;
    if (!ann_get_layer_masked(pnet, 1))
        return 1;
    if (ann_compute_masks(pnet, inputs, rows) != ERR_OK)
        return 0;

    NSMutableDictionary<NSNumber *, id<MTLBuffer>> *pad = reuse ? reuse : [NSMutableDictionary dictionary];
    if (!g_metal.pad_kept)
        g_metal.pad_kept = [NSMutableDictionary dictionary];
    for (int layer = 1; layer < pnet->layer_count; layer++) {
        PLayer l = &pnet->layers[layer];
        if (!l->masked) {
            [pad removeObjectForKey:@(layer)];
            continue;
        }
        size_t n = (size_t)rows * l->shape.w;
        id<MTLBuffer> buf = gpu_scratch(pad[@(layer)], n * sizeof(float));
        if (!buf)
            return 0;
        pad[@(layer)] = buf;
        memcpy([buf contents], l->t_mask->values, n * sizeof(float));
        uint32_t kept = 0;
        for (size_t i = 0; i < n; i++)
            kept += l->t_mask->values[i] != 0.0f;
        g_metal.pad_kept[@(layer)] = @(kept);
    }
    g_metal.pad = pad;
    return 1;
}

// zero values (or gradients) [batch x nodes] of a masked layer at its masked steps
static void encode_mask_steps(id<MTLCommandBuffer> cmd, PLayer l, id<MTLBuffer> v, id<MTLBuffer> mask, int batch_size)
{
    uint32_t steps_nodes[2] = { (uint32_t)l->shape.w, (uint32_t)l->node_count };
    id<MTLBuffer> bufs[2] = { v, mask };
    encode_grid(cmd, g_metal.pso_mask_steps, bufs, 2, steps_nodes, sizeof(steps_nodes),
                MTLSizeMake((NSUInteger)batch_size * l->node_count, 1, 1));
}

//-----------------------------------------------------------
// 1 if every parameter of the layer is on the GPU
//-----------------------------------------------------------
static int layer_on_gpu(const Layer *l)
{
    for (int p = 0; p < l->param_count; p++)
        if (!l->params[p].value->gpu_buf)
            return 0;
    return 1;
}

//-----------------------------------------------------------
// 1 if the network's parameters were uploaded (the check ann.c makes)
//-----------------------------------------------------------
static int network_on_gpu(const PNetwork pnet)
{
    for (int layer = 1; layer < pnet->layer_count; layer++)
        if (pnet->layers[layer].param_count > 0)
            return pnet->layers[layer].params[0].value->gpu_buf != NULL;
    return 0;
}

//-----------------------------------------------------------
// Encode a layer's forward pass for a batch: the kind's linear part (dense:
// Y = X * W^T + b), then (when Z is given) save the pre-activation, apply
// the activation, and apply dropout when a mask is given. training selects
// batch norm's batch statistics over its running ones.
//-----------------------------------------------------------
static int encode_layer_forward(id<MTLCommandBuffer> cmd, PNetwork pnet, int layer, int batch_size,
                                id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> Z, id<MTLBuffer> mask, int training)
{
    PLayer dst = &pnet->layers[layer];
    int out_nodes = dst->node_count;
    NSUInteger count = (NSUInteger)batch_size * out_nodes;

    if (!layer_on_gpu(dst)) {
        fprintf(stderr, "[Metal] Layer %d parameters not on GPU\n", layer);
        return 0;
    }

    // Only GELU's derivative needs the values before activation (ReLU's and
    // leaky ReLU's read the sign from the output), so Z is saved only for it
    if (dst->activation != ACTIVATION_GELU)
        Z = nil;

    // Dense and conv layers apply their activation (other than softmax) as
    // they add the bias
    int activated = 0;
    switch (dst->kind) {
    case LAYER_KIND_DENSE: {
        int in_nodes = pnet->layers[layer - 1].node_count;
        encode_gemm(cmd, X, batch_size, in_nodes, NO, (__bridge id<MTLBuffer>)dst->t_weights->gpu_buf,
                    out_nodes, in_nodes, YES, Y, 1.0, 0.0);
        dispatch_bias_add(cmd, Y, (__bridge id<MTLBuffer>)dst->t_bias->gpu_buf, (NSUInteger)batch_size,
                          (NSUInteger)out_nodes, dst->activation, Z);
        activated = 1;
        break;
    }
    case LAYER_KIND_CONV2D:
        if (!encode_conv_forward(cmd, pnet, layer, batch_size, X, Y, Z)) return 0;
        activated = 1;
        break;
    case LAYER_KIND_MAXPOOL2D:
        if (!encode_maxpool_forward(cmd, pnet, layer, batch_size, X, Y)) return 0;
        break;
    case LAYER_KIND_UPSAMPLE2D:
        encode_upsample_forward(cmd, pnet, layer, batch_size, X, Y);
        break;
    case LAYER_KIND_GLOBAL_AVGPOOL2D:
        encode_global_avgpool_forward(cmd, pnet, layer, batch_size, X, Y);
        break;
    case LAYER_KIND_GLOBAL_MAXPOOL2D:
        if (!encode_global_maxpool(cmd, pnet, layer, batch_size, X, Y, 0)) return 0;
        break;
    case LAYER_KIND_BATCHNORM:
        if (!encode_batchnorm_forward(cmd, pnet, layer, batch_size, X, Y, training)) return 0;
        break;
    case LAYER_KIND_EMBEDDING:
        encode_embedding_forward(cmd, pnet, layer, batch_size, X, Y);
        break;
    case LAYER_KIND_FLATTEN:
    case LAYER_KIND_RESHAPE:
        encode_copy(cmd, X, Y, count);
        break;
    default:
        return 0;
    }

    if (dst->activation == ACTIVATION_SOFTMAX) {
        dispatch_softmax(cmd, Y, (uint32_t)out_nodes, (NSUInteger)batch_size);
    } else if (!activated && dst->activation != ACTIVATION_NULL) {
        id<MTLComputePipelineState> pso = pso_for_activation(dst->activation);
        if (Z)
            encode_copy(cmd, Y, Z, count);
        if (pso)
            dispatch_activation(cmd, pso, Y, count);
    }

    if (mask)
        dispatch_binary_elementwise(cmd, g_metal.pso_dropout_forward, Y, mask, count);
    if (g_metal.pad[@(layer)])
        encode_mask_steps(cmd, dst, Y, g_metal.pad[@(layer)], batch_size);
    return 1;
}

//-----------------------------------------------------------
// GPU batched inference: each layer's forward pass in turn, encoded in one
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

        // the last layer writes straight into the caller's buffer
        id<MTLBuffer> out = output_buf;
        if (layer < pnet->layer_count - 1) {
            out = [g_metal.device newBufferWithLength:(size_t)batch_size * dst->node_count * sizeof(float)
                                              options:MTLResourceStorageModePrivate];
            if (!out) return 0;
        }

        if (!encode_layer_forward(cmd, pnet, layer, batch_size, cur_buf, out, nil, nil, 0))
            return 0;
        cur_buf = out;
    }

    return commit_and_wait(cmd);
}

//-----------------------------------------------------------
// Internal Metal backend functions (registered in metal_backend vtable).
// These are not part of the public libann API.
//-----------------------------------------------------------

static int metal_upload_network(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized)
        return ERR_NULL_PTR;
    finish_all_slots();

    // Every layer kind in ann.h is implemented; reject any added later, and an
    // embedding anywhere but layer 1 (its ids, and so its masks and its
    // gradient's id index, come from the inputs on the CPU)
    for (int layer = 0; layer < pnet->layer_count; layer++) {
        Layer_kind kind = pnet->layers[layer].kind;
        if (kind != LAYER_KIND_DENSE && kind != LAYER_KIND_CONV2D && kind != LAYER_KIND_MAXPOOL2D &&
            kind != LAYER_KIND_FLATTEN && kind != LAYER_KIND_UPSAMPLE2D && kind != LAYER_KIND_RESHAPE &&
            kind != LAYER_KIND_GLOBAL_AVGPOOL2D && kind != LAYER_KIND_BATCHNORM &&
            kind != LAYER_KIND_GLOBAL_MAXPOOL2D && (kind != LAYER_KIND_EMBEDDING || layer != 1))
            return ERR_INVALID;
    }

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
    finish_all_slots();

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
    [g_metal.scratch removeAllObjects];
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
    finish_all_slots();     // inference shares the layers' work buffers with training

    // Check that weights are on GPU
    if (!network_on_gpu(pnet))
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

    if (!set_pad_masks(pnet, pnet->layers[0].t_values->values, 1, nil))
        return 0;
    int ok = metal_forward_pass(pnet, input_buf, output_buf, 1);
    g_metal.pad = nil;
    if (!ok)
        return 0;

    // Write GPU output back into the output layer's t_values
    memcpy(pnet->layers[pnet->layer_count - 1].t_values->values,
           [output_buf contents],
           out_bytes);

    return 1;
}

//-----------------------------------------------------------
// GPU memory one sample needs in a forward pass: each layer's output plus
// conv im2col columns and output rows (pool argmax fits in the output size)
static size_t forward_bytes_per_sample(const PNetwork pnet)
{
    size_t floats = 0;
    for (int layer = 0; layer < pnet->layer_count; layer++) {
        const Layer *l = &pnet->layers[layer];
        floats += (size_t)l->node_count;
        if (l->kind == LAYER_KIND_CONV2D)
            floats += (size_t)l->shape.h * l->shape.w * (l->params[0].value->cols + l->shape.c);
        else if (l->kind == LAYER_KIND_MAXPOOL2D || l->kind == LAYER_KIND_GLOBAL_MAXPOOL2D)
            floats += (size_t)l->node_count;
    }
    return floats * sizeof(float);
}

// Rows per forward pass are capped so a large batch (conv columns grow with
// it) is evaluated in pieces instead of failing to allocate
#define EVAL_BATCH_BYTES ((size_t)256 << 20)

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
    finish_all_slots();     // inference shares the layers' work buffers with training

    if (!g_metal.initialized)
        return ERR_FAIL;

    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[pnet->layer_count - 1].node_count;
    size_t chunk_rows = EVAL_BATCH_BYTES / forward_bytes_per_sample(pnet);
    int chunk = chunk_rows < 1 ? 1 : (chunk_rows < (size_t)batch_size ? (int)chunk_rows : batch_size);

    for (int start = 0; start < batch_size; start += chunk) {
        int rows = batch_size - start < chunk ? batch_size - start : chunk;
        size_t in_bytes  = (size_t)rows * input_nodes  * sizeof(float);
        size_t out_bytes = (size_t)rows * output_nodes * sizeof(float);

        // release each piece's buffers before the next
        @autoreleasepool {
            id<MTLBuffer> input_buf = [g_metal.device newBufferWithBytes:inputs + (size_t)start * input_nodes
                                                                  length:in_bytes
                                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buf = [g_metal.device newBufferWithLength:out_bytes
                                                                   options:MTLResourceStorageModeShared];
            if (!input_buf || !output_buf)
                return ERR_ALLOC;

            if (!set_pad_masks((PNetwork)pnet, inputs + (size_t)start * input_nodes, rows, nil))
                return ERR_ALLOC;
            int ok = metal_forward_pass((PNetwork)pnet, input_buf, output_buf, rows);
            g_metal.pad = nil;
            if (!ok)
                return ERR_FAIL;

            memcpy(outputs + (size_t)start * output_nodes, [output_buf contents], out_bytes);
        }
    }
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
// This step's dropout mask for a hidden layer, copied from the one ann.c drew
// into the slot's own buffer (a step still running may be reading the last
// one). Returns nil when the layer does not use dropout.
//-----------------------------------------------------------
static id<MTLBuffer> slot_dropout_mask(int s, PNetwork pnet, int layer, int batch_size)
{
    PLayer l = &pnet->layers[layer];
    PTensor m = l->t_dropout_mask;
    if (!pnet->is_training || l->dropout_rate <= 0.0f || !m ||
        m->rows != batch_size || m->cols != l->node_count)
        return nil;

    size_t bytes = (size_t)batch_size * l->node_count * sizeof(float);
    if (!g_metal.slots[s].masks)
        g_metal.slots[s].masks = [NSMutableDictionary dictionary];
    id<MTLBuffer> buf = gpu_scratch(g_metal.slots[s].masks[@(layer)], bytes);
    if (!buf)
        return nil;
    g_metal.slots[s].masks[@(layer)] = buf;
    memcpy([buf contents], m->values, bytes);
    return buf;
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
    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
    for (int i = 0; i < n_consts; i++)
        [enc setBytes:&consts[i] length:sizeof(float) atIndex:(NSUInteger)(n_bufs + i)];
    NSUInteger tg = MIN(pso.maxTotalThreadsPerThreadgroup, count);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
        id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
        uint32_t cols = (uint32_t)l->node_count;
        [enc setComputePipelineState:g_metal.pso_softmax_jacobian];
        [enc setBuffer:dl_dz offset:0 atIndex:0];
        [enc setBuffer:A offset:0 atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)batch_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                return;
    }

    id<MTLComputePipelineState> pso = pso_for_deriv(l->activation);
    if (!pso)
        return;
    // the forward pass saves Z only for GELU; ReLU's and leaky ReLU's
    // derivatives test z > 0, which is a > 0
    dispatch_binary_elementwise(cmd, pso, dl_dz, l->activation == ACTIVATION_GELU ? Z : A, count);
}

//-----------------------------------------------------------
// Conv2D layers apply their activation's derivative in encode_conv_backward
// (softmax keeps its own kernel)
//-----------------------------------------------------------
static int conv_fuses_derivative(PLayer l)
{
    return l->kind == LAYER_KIND_CONV2D && l->activation != ACTIVATION_SOFTMAX;
}

//-----------------------------------------------------------
// Encode the output layer's delta (t - y, then the activation derivative for
// MSE; cross-entropy with softmax/sigmoid needs none) and each row's loss
//-----------------------------------------------------------
static void encode_output_delta_loss(id<MTLCommandBuffer> cmd, PNetwork pnet, int batch_size,
                                     id<MTLBuffer> targets, id<MTLBuffer> row_loss)
{
    PLayer out = &pnet->layers[pnet->layer_count - 1];
    uint32_t cols = (uint32_t)out->node_count;
    uint32_t mode = pnet->loss_type == LOSS_MSE ? 0 : (out->activation == ACTIVATION_SIGMOID ? 2 : 1);

    id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
    [enc setComputePipelineState:g_metal.pso_output_delta_loss];
    [enc setBuffer:(__bridge id<MTLBuffer>)out->t_batch_values->gpu_buf offset:0 atIndex:0];
    [enc setBuffer:targets offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out->t_batch_dl_dz->gpu_buf offset:0 atIndex:2];
    [enc setBuffer:row_loss offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&mode length:sizeof(mode) atIndex:5];
    // a threadgroup per row: the smallest power of two covering the row, at most 256
    NSUInteger tg = 1;
    while (tg < cols && tg < 256 && tg * 2 <= g_metal.pso_output_delta_loss.maxTotalThreadsPerThreadgroup)
        tg *= 2;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

    if (pnet->loss_type == LOSS_MSE && !conv_fuses_derivative(out))
        encode_activation_derivative(cmd, out, batch_size);
}

//-----------------------------------------------------------
// Encode the backward pass. On entry the output layer's t_batch_dl_dz holds
// -dL/dz (-dL/da for a Conv2D output). For each layer, from the output down
// to layer 1:
//   1. hidden layers: undo dropout, then the activation derivative (Conv2D
//      applies it in its backward pass)
//   2. the kind's backward pass: accumulate parameter gradients (dense:
//      dW += dl_dz^T * A_prev, bias_grad += column sums of dl_dz) and write
//      dl_dz_prev (not for layer 1, whose input needs none)
// Returns 0 if a work buffer could not be allocated.
//-----------------------------------------------------------
static int encode_backward(id<MTLCommandBuffer> cmd, PNetwork pnet, int batch_size, id<MTLBuffer> input,
                           NSDictionary<NSNumber *, id<MTLBuffer>> *masks)
{
    for (int li = pnet->layer_count - 1; li >= 1; li--) {
        PLayer layer      = &pnet->layers[li];
        PLayer prev_layer = &pnet->layers[li - 1];
        int nodes      = layer->node_count;
        int prev_nodes = prev_layer->node_count;
        NSUInteger count = (NSUInteger)batch_size * nodes;

        id<MTLBuffer> dl_dz  = (__bridge id<MTLBuffer>)layer->t_batch_dl_dz->gpu_buf;
        id<MTLBuffer> A_prev = li == 1 ? input : (__bridge id<MTLBuffer>)prev_layer->t_batch_values->gpu_buf;
        id<MTLBuffer> dX = li > 1 ? (__bridge id<MTLBuffer>)prev_layer->t_batch_dl_dz->gpu_buf : nil;

        if (li < pnet->layer_count - 1) {
            id<MTLBuffer> mask = masks[@(li)];
            if (mask) {
                id<MTLBuffer> bufs[3] = { dl_dz, (__bridge id<MTLBuffer>)layer->t_batch_values->gpu_buf, mask };
                encode_1d(cmd, g_metal.pso_dropout_backward, bufs, 3, NULL, 0, count);
            }
            // masked steps' outputs are constant zeros
            if (g_metal.pad[@(li)])
                encode_mask_steps(cmd, layer, dl_dz, g_metal.pad[@(li)], batch_size);
            if (!conv_fuses_derivative(layer))
                encode_activation_derivative(cmd, layer, batch_size);
        } else if (g_metal.pad[@(li)])
            encode_mask_steps(cmd, layer, dl_dz, g_metal.pad[@(li)], batch_size);

        switch (layer->kind) {
        case LAYER_KIND_DENSE:
            // dW += dl_dz^T * A_prev  ([nodes x batch] * [batch x prev_nodes])
            encode_gemm(cmd, dl_dz, batch_size, nodes, YES, A_prev, batch_size, prev_nodes, NO,
                        (__bridge id<MTLBuffer>)layer->t_gradients->gpu_buf, 1.0, 1.0);

            // bias_grad += column sums of dl_dz
            {
                id<MTLComputeCommandEncoder> enc = compute_encoder(cmd);
                uint32_t bs = (uint32_t)batch_size, n = (uint32_t)nodes;
                [enc setComputePipelineState:g_metal.pso_bias_grad_sum];
                [enc setBuffer:dl_dz offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)layer->t_bias_grad->gpu_buf offset:0 atIndex:1];
                [enc setBytes:&bs length:sizeof(bs) atIndex:2];
                [enc setBytes:&n  length:sizeof(n)  atIndex:3];
                NSUInteger tg = MIN(g_metal.pso_bias_grad_sum.maxTotalThreadsPerThreadgroup, (NSUInteger)nodes);
                [enc dispatchThreads:MTLSizeMake((NSUInteger)nodes, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                            }

            // dl_dz_prev = dl_dz * W  ([batch x nodes] * [nodes x prev_nodes])
            if (dX)
                encode_gemm(cmd, dl_dz, batch_size, nodes, NO,
                            (__bridge id<MTLBuffer>)layer->t_weights->gpu_buf, nodes, prev_nodes, NO,
                            dX, 1.0, 0.0);
            break;
        case LAYER_KIND_CONV2D:
            // the output layer's delta already has its derivative, if any
            if (!encode_conv_backward(cmd, pnet, li, batch_size, A_prev, dl_dz, dX,
                                      li < pnet->layer_count - 1 || pnet->loss_type == LOSS_MSE ? layer->activation
                                                                                                 : ACTIVATION_NULL))
                return 0;
            break;
        case LAYER_KIND_MAXPOOL2D:
            if (dX && !encode_maxpool_backward(cmd, pnet, li, batch_size, dl_dz, dX)) return 0;
            break;
        case LAYER_KIND_UPSAMPLE2D:
            if (dX) encode_upsample_backward(cmd, pnet, li, batch_size, dl_dz, dX);
            break;
        case LAYER_KIND_GLOBAL_AVGPOOL2D:
            if (dX) encode_global_avgpool_backward(cmd, pnet, li, batch_size, dl_dz, dX);
            break;
        case LAYER_KIND_GLOBAL_MAXPOOL2D:
            if (dX && !encode_global_maxpool(cmd, pnet, li, batch_size, dl_dz, dX, 1)) return 0;
            break;
        case LAYER_KIND_BATCHNORM:
            if (!encode_batchnorm_backward(cmd, pnet, li, batch_size, dl_dz, dX)) return 0;
            break;
        case LAYER_KIND_EMBEDDING:
            encode_embedding_backward(cmd, pnet, li, dl_dz);    // the ids below get no gradient
            break;
        case LAYER_KIND_FLATTEN:
        case LAYER_KIND_RESHAPE:
            if (dX) encode_copy(cmd, dl_dz, dX, count);
            break;
        default:
            return 0;
        }
    }
    return 1;
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
        if (l->frozen)
            continue;	// ann_set_layer_trainable
        for (int p = 0; p < l->param_count; p++) {
            Layer_param *param = &l->params[p];
            if (param->state)
                continue;	// running statistics, updated by their own kernels
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
    finish_all_slots();

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
// Copies them into a free slot, encodes the step and commits it without
// waiting: the CPU assembles the next batch while the GPU runs this one,
// and metal_train_finish() collects the losses. Returns 1 when the GPU took
// the batch, 0 to fall back to the CPU.
//-----------------------------------------------------------
static int metal_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size)
{
    if (!pnet || !g_metal.initialized) return 0;
    if (!network_on_gpu(pnet)) return 0;
    if (!pnet->layers[0].t_batch_values) return 0;

    // The slot's previous step must be done before its buffers are reused
    int s = g_metal.next_slot;
    finish_slot(s);
    if (g_metal.train_failed) return 0;

    int output_layer_idx = pnet->layer_count - 1;
    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[output_layer_idx].node_count;

    // Lazy-alloc training buffers on first call, if the batch size changed,
    // or for stale backend state across network lifetimes; steps in flight
    // use them, so they finish first
    PLayer out = &pnet->layers[output_layer_idx];
    if (!g_metal.training_buffers_ready || g_metal.training_batch_size != batch_size ||
        !out->t_batch_values || !out->t_batch_values->gpu_buf || !out->t_batch_dl_dz || !out->t_batch_dl_dz->gpu_buf) {
        finish_all_slots();
        if (g_metal.train_failed) return 0;
        g_metal.training_buffers_ready = 0;
        if (!gpu_alloc_training_buffers(pnet, batch_size)) return 0;
    }

    // This step's inputs, targets and dropout masks, in the slot's buffers
    size_t input_bytes = (size_t)batch_size * input_nodes * sizeof(float);
    size_t target_bytes = (size_t)batch_size * output_nodes * sizeof(float);
    g_metal.slots[s].inputs   = gpu_scratch(g_metal.slots[s].inputs, input_bytes);
    g_metal.slots[s].targets  = gpu_scratch(g_metal.slots[s].targets, target_bytes);
    g_metal.slots[s].row_loss = gpu_scratch(g_metal.slots[s].row_loss, (size_t)batch_size * sizeof(float));
    if (!g_metal.slots[s].inputs || !g_metal.slots[s].targets || !g_metal.slots[s].row_loss) return 0;
    memcpy([g_metal.slots[s].inputs contents], pnet->layers[0].t_batch_values->values, input_bytes);
    memcpy([g_metal.slots[s].targets contents], batch_targets->values, target_bytes);

    NSMutableDictionary<NSNumber *, id<MTLBuffer>> *masks = [NSMutableDictionary dictionary];
    for (int layer = 1; layer < output_layer_idx; layer++) {
        id<MTLBuffer> mask = slot_dropout_mask(s, pnet, layer, batch_size);
        if (mask)
            masks[@(layer)] = mask;
    }

    // padding masks and the embedding's id index, from this step's ids
    const real *ids = pnet->layers[0].t_batch_values->values;
    if (!g_metal.slots[s].pad)
        g_metal.slots[s].pad = [NSMutableDictionary dictionary];
    if (!set_pad_masks(pnet, ids, batch_size, g_metal.slots[s].pad))
        return 0;
    g_metal.emb_count = 0;
    if (pnet->layers[1].kind == LAYER_KIND_EMBEDDING) {
        size_t n = (size_t)batch_size * input_nodes;
        g_metal.slots[s].emb_unique = gpu_scratch(g_metal.slots[s].emb_unique, (n + 1) * sizeof(uint32_t));
        g_metal.slots[s].emb_offsets = gpu_scratch(g_metal.slots[s].emb_offsets, (n + 1) * sizeof(uint32_t));
        g_metal.slots[s].emb_positions = gpu_scratch(g_metal.slots[s].emb_positions, n * sizeof(uint32_t));
        if (!g_metal.slots[s].emb_unique || !g_metal.slots[s].emb_offsets || !g_metal.slots[s].emb_positions)
            return 0;
        int count = ann_embedding_index(&pnet->layers[1], ids, batch_size, [g_metal.slots[s].emb_unique contents],
                                        [g_metal.slots[s].emb_offsets contents], [g_metal.slots[s].emb_positions contents]);
        if (count < 0)
            return 0;
        g_metal.emb_count = (uint32_t)count;
        g_metal.emb_unique = g_metal.slots[s].emb_unique;
        g_metal.emb_offsets = g_metal.slots[s].emb_offsets;
        g_metal.emb_positions = g_metal.slots[s].emb_positions;
    }

    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];

    // Zero the gradients
    {
        end_compute();
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
        id<MTLBuffer> X = layer == 1 ? g_metal.slots[s].inputs
                                     : (__bridge id<MTLBuffer>)pnet->layers[layer - 1].t_batch_values->gpu_buf;
        if (!encode_layer_forward(cmd, pnet, layer, batch_size, X, (__bridge id<MTLBuffer>)dst->t_batch_values->gpu_buf,
                                  (__bridge id<MTLBuffer>)dst->t_batch_z->gpu_buf, masks[@(layer)], 1)) {
            end_compute();
            return 0;
        }
    }

    encode_output_delta_loss(cmd, pnet, batch_size, g_metal.slots[s].targets, g_metal.slots[s].row_loss);
    if (!encode_backward(cmd, pnet, batch_size, g_metal.slots[s].inputs, masks)) {
        end_compute();
        return 0;
    }

    // Adam's bias correction counts this batch
    pnet->train_iteration++;
    encode_optimizer(cmd, pnet);

    end_compute();
    g_metal.pad = nil;
    [cmd commit];
    g_metal.slots[s].cmd = cmd;
    g_metal.slots[s].batch_size = batch_size;
    g_metal.slots[s].divisor = pnet->loss_type == LOSS_MSE ? batch_size * output_nodes : batch_size;
    g_metal.next_slot = (s + 1) % TRAIN_SLOTS;
    return 1;
}

//-----------------------------------------------------------
// Wait for the steps metal_train_batch() submitted; *loss_sum gets the sum
// of their batch losses. Returns 0 if any failed.
//-----------------------------------------------------------
static int metal_train_finish(PNetwork pnet, real *loss_sum)
{
    (void)pnet;
    finish_all_slots();
    if (loss_sum)
        *loss_sum = g_metal.loss_sum;
    int ok = !g_metal.train_failed;
    g_metal.loss_sum = 0.0f;
    g_metal.train_failed = 0;
    return ok;
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

static int pooled_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size)
{
    @autoreleasepool { return metal_train_batch(pnet, batch_targets, batch_size); }
}

static int pooled_train_finish(PNetwork pnet, real *loss_sum)
{
    @autoreleasepool { return metal_train_finish(pnet, loss_sum); }
}

GpuBackend metal_backend = {
    "Metal",                 /* name           */
    pooled_init,             /* init           */
    pooled_upload_network,   /* upload_network */
    pooled_free_network,     /* free_network   */
    pooled_sync_weights,     /* sync_weights   */
    pooled_release_buffer,   /* release_buffer */
    pooled_eval_single,      /* eval_single    */
    pooled_eval_batch,       /* eval_batch     */
    pooled_train_batch,      /* train_batch    */
    pooled_train_finish,     /* train_finish   */
};

#endif /* USE_METAL */
