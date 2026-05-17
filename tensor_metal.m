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

//-----------------------------------------------------------
// Internal GPU context (singleton)
//-----------------------------------------------------------
typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> pso_sigmoid;
    id<MTLComputePipelineState> pso_relu;
    id<MTLComputePipelineState> pso_leaky_relu;
    id<MTLComputePipelineState> pso_tanh;
    id<MTLComputePipelineState> pso_softsign;
    id<MTLComputePipelineState> pso_softmax;
    id<MTLComputePipelineState> pso_bias_add;
    int initialized;
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
        "        for (uint n = 0; n < cols; n++) buf[offset + n] /= s; }\n";

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
// Public C API (called from ann.c)
//-----------------------------------------------------------

int ann_gpu_init(void)
{
    return tensor_metal_init();
}

int ann_gpu_upload_network(PNetwork pnet)
{
    if (!pnet || !g_metal.initialized)
        return ERR_NULL_PTR;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];

        if (l->t_weights && !tensor_metal_upload(l->t_weights)) return ERR_FAIL;
        if (l->t_bias    && !tensor_metal_upload(l->t_bias))    return ERR_FAIL;
    }

    return ERR_OK;
}

void ann_gpu_free_network(PNetwork pnet)
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
}

//-----------------------------------------------------------
// GPU-accelerated single-sample inference.
// Called from eval_network() when Metal is available.
// Returns 1 on success, 0 if GPU not ready (fallback to CPU).
//-----------------------------------------------------------
int ann_gpu_eval_single(PNetwork pnet)
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
int ann_predict_batch_metal(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
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

#endif /* USE_METAL */
