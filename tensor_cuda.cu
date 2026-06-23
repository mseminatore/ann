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

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ann.h"
#include "ann_gpu_backend.h"
#include "shaders_cuda.h"

// NOTE: The CUDA backend is float-only (real must be float). cublasSgemm and the
// shaders.cu kernels operate on float; a double build is not supported. See docs/GPU_CUDA.md.

#define CUDA_THREADS 256
static inline int cuda_blocks(int n) { return (n + CUDA_THREADS - 1) / CUDA_THREADS; }

typedef struct {
    int initialized;
    int device_id;
    cublasHandle_t cublas;
    int training_buffers_ready;   // training-time GPU buffers allocated
    int training_batch_size;      // batch size the training buffers were sized for

    // Cached inference ping-pong buffers (avoid cudaMalloc/cudaFree per predict call).
    float *infer_bufA;
    float *infer_bufB;
    size_t infer_buf_bytes;       // current capacity of each inference buffer

    // Cached training loss/delta buffers (device-side targets + per-element loss).
    float *train_targets;         // [batch x out_nodes] uploaded targets
    float *train_lossbuf;         // [batch x out_nodes] per-element loss terms
    size_t train_loss_bytes;      // current capacity of each of the two buffers above
} CudaContext;

static CudaContext g_cuda = {0};

static int cuda_check(cudaError_t err, const char *where)
{
    if (err == cudaSuccess)
        return 1;
    fprintf(stderr, "[CUDA] %s: %s\n", where, cudaGetErrorString(err));
    return 0;
}

static int cublas_check(cublasStatus_t st, const char *where)
{
    if (st == CUBLAS_STATUS_SUCCESS)
        return 1;
    fprintf(stderr, "[CUDA] %s: cuBLAS status %d\n", where, (int)st);
    return 0;
}

// Check for errors from the most recent kernel launch(es).  cudaGetLastError() does not
// synchronize, so this is cheap and catches launch-configuration failures (e.g. an invalid
// launch on an unsupported architecture) that would otherwise be silent until the next sync.
#define CUDA_KERNEL_CHECK(where) cuda_check(cudaGetLastError(), (where))

static int cuda_backend_init(void)
{
    if (g_cuda.initialized)
        return 1;

    int device_count = 0;
    cudaError_t cnt_err = cudaGetDeviceCount(&device_count);
    if (cnt_err != cudaSuccess || device_count <= 0)
    {
        int driver_ver = 0, runtime_ver = 0;
        cudaDriverGetVersion(&driver_ver);
        cudaRuntimeGetVersion(&runtime_ver);
        fprintf(stderr,
            "[CUDA] GPU init failed: cudaGetDeviceCount -> %s (devices=%d)\n"
            "       CUDA driver supports up to %d.%d; runtime/toolkit is %d.%d.\n",
            cudaGetErrorString(cnt_err), device_count,
            driver_ver / 1000, (driver_ver % 1000) / 10,
            runtime_ver / 1000, (runtime_ver % 1000) / 10);
        if (driver_ver > 0 && driver_ver < runtime_ver)
            fprintf(stderr,
                "       Hint: the driver is older than the toolkit. Build with a CUDA toolkit\n"
                "       no newer than the driver (e.g. -T cuda=\"...vXX.Y\"), or update the driver.\n");
        else if (driver_ver > 0)
            fprintf(stderr,
                "       Hint: also ensure the build targets this GPU's compute capability via\n"
                "       -DCMAKE_CUDA_ARCHITECTURES=<arch> (e.g. 61 for Pascal, 75 Turing, 86 Ampere).\n");
        return 0;
    }

    g_cuda.device_id = 0;
    if (!cuda_check(cudaSetDevice(g_cuda.device_id), "cudaSetDevice"))
        return 0;

    if (!cublas_check(cublasCreate(&g_cuda.cublas), "cublasCreate"))
        return 0;

    g_cuda.initialized = 1;
    return 1;
}

static int cuda_upload_tensor(PTensor t)
{
    if (!t || !g_cuda.initialized)
        return 0;

    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);
    if (bytes == 0)
        return 1;

    void *dev_ptr = NULL;
    if (!cuda_check(cudaMalloc(&dev_ptr, bytes), "cudaMalloc"))
        return 0;
    if (!cuda_check(cudaMemcpy(dev_ptr, t->values, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D"))
    {
        cudaFree(dev_ptr);
        return 0;
    }

    if (t->gpu_buf)
        cudaFree(t->gpu_buf);
    t->gpu_buf = dev_ptr;
    return 1;
}

static int cuda_upload_network(PNetwork pnet)
{
    if (!pnet || !g_cuda.initialized)
        return ERR_NULL_PTR;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (l->t_weights && !cuda_upload_tensor(l->t_weights)) return ERR_FAIL;
        if (l->t_bias && !cuda_upload_tensor(l->t_bias)) return ERR_FAIL;
    }

    return ERR_OK;
}

//-----------------------------------------------------------
// Release one tensor's GPU buffer (device pointer) and NULL it.
//-----------------------------------------------------------
static void cuda_free_one(PTensor t)
{
    if (t && t->gpu_buf)
    {
        cudaFree(t->gpu_buf);
        t->gpu_buf = NULL;
    }
}

//-----------------------------------------------------------
// Free all training-time GPU buffers (everything except weights
// and biases, which are owned by upload/free_network).
//-----------------------------------------------------------
static void cuda_free_training_buffers(PNetwork pnet)
{
    if (!pnet)
        return;

    for (int layer = 0; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        cuda_free_one(l->t_batch_values);
        if (layer > 0)
        {
            cuda_free_one(l->t_batch_z);
            cuda_free_one(l->t_batch_dl_dz);
        }
        if (layer < pnet->layer_count - 1)
        {
            cuda_free_one(l->t_gradients);
            cuda_free_one(l->t_bias_grad);
            cuda_free_one(l->t_m);
            cuda_free_one(l->t_v);
            cuda_free_one(l->t_bias_m);
            cuda_free_one(l->t_bias_v);
        }
    }

    g_cuda.training_buffers_ready = 0;
    g_cuda.training_batch_size = 0;

    // Release cached on-device loss/delta buffers.
    if (g_cuda.train_targets) { cudaFree(g_cuda.train_targets); g_cuda.train_targets = NULL; }
    if (g_cuda.train_lossbuf) { cudaFree(g_cuda.train_lossbuf); g_cuda.train_lossbuf = NULL; }
    g_cuda.train_loss_bytes = 0;
}

static void cuda_free_network(PNetwork pnet)
{
    if (!pnet)
        return;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];
        cuda_free_one(l->t_weights);
        cuda_free_one(l->t_bias);
    }

    // Training buffers belong to this network too; reset readiness so a
    // subsequent training run reallocates from scratch.
    cuda_free_training_buffers(pnet);

    // Release cached inference scratch buffers (lazily reallocated on next predict).
    if (g_cuda.infer_bufA) { cudaFree(g_cuda.infer_bufA); g_cuda.infer_bufA = NULL; }
    if (g_cuda.infer_bufB) { cudaFree(g_cuda.infer_bufB); g_cuda.infer_bufB = NULL; }
    g_cuda.infer_buf_bytes = 0;
}

static void cuda_sync_weights(PNetwork pnet)
{
    if (!pnet || !g_cuda.initialized)
        return;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (l->t_weights && l->t_weights->gpu_buf)
        {
            size_t bytes = (size_t)(l->t_weights->rows * l->t_weights->cols) * sizeof(real);
            cudaMemcpy(l->t_weights->values, l->t_weights->gpu_buf, bytes, cudaMemcpyDeviceToHost);
        }
        if (l->t_bias && l->t_bias->gpu_buf)
        {
            size_t bytes = (size_t)l->t_bias->cols * sizeof(real);
            cudaMemcpy(l->t_bias->values, l->t_bias->gpu_buf, bytes, cudaMemcpyDeviceToHost);
        }
    }
}

static void cuda_release_buffer(void *gpu_buf)
{
    if (gpu_buf)
        cudaFree(gpu_buf);
}

//================================================================================================
// Compute helpers
//================================================================================================

//-----------------------------------------------------------
// Row-major GEMM:  C[M x N] = alpha * opA(A) * opB(B) + beta * C
//
// libann tensors are row-major; cuBLAS is column-major.  A row-major M x N matrix
// with leading dim N is, when viewed by cuBLAS as column-major, an N x M matrix
// (its transpose).  So we compute the column-major transpose C^T = opB(B)^T * opA(A)^T
// by swapping the operands (pass B first, then A) and the m/n dimensions.  The op
// flag for each stored matrix maps directly to its transA/transB request.
//-----------------------------------------------------------
static int cuda_gemm_rowmajor(int transA, int transB, int M, int N, int K,
                              float alpha, const float *A, const float *B,
                              float beta, float *C)
{
    int lda = transA ? M : K;   // physical row width (cols) of stored A
    int ldb = transB ? K : N;   // physical row width (cols) of stored B
    int ldc = N;                // physical row width (cols) of stored C

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t st = cublasSgemm(g_cuda.cublas,
                                    opB, opA,
                                    N, M, K,
                                    &alpha,
                                    B, ldb,
                                    A, lda,
                                    &beta,
                                    C, ldc);
    return cublas_check(st, "cublasSgemm");
}

//-----------------------------------------------------------
// Launch the forward activation kernel for a [batch x cols] buffer.
//-----------------------------------------------------------
static void cuda_launch_activation(Activation_type act, float *buf, int batch, int cols)
{
    int n = batch * cols;
    switch (act)
    {
        case ACTIVATION_SIGMOID:    cuda_activation_sigmoid<<<cuda_blocks(n), CUDA_THREADS>>>(buf, n); break;
        case ACTIVATION_RELU:       cuda_activation_relu<<<cuda_blocks(n), CUDA_THREADS>>>(buf, n); break;
        case ACTIVATION_LEAKY_RELU: cuda_activation_leaky_relu<<<cuda_blocks(n), CUDA_THREADS>>>(buf, n); break;
        case ACTIVATION_TANH:       cuda_activation_tanh<<<cuda_blocks(n), CUDA_THREADS>>>(buf, n); break;
        case ACTIVATION_SOFTSIGN:   cuda_activation_softsign<<<cuda_blocks(n), CUDA_THREADS>>>(buf, n); break;
        case ACTIVATION_SOFTMAX:    cuda_softmax_rows<<<cuda_blocks(batch), CUDA_THREADS>>>(buf, batch, cols); break;
        case ACTIVATION_NULL:
        default:                    break;
    }
}

//-----------------------------------------------------------
// Launch the activation-derivative kernel: dl_dz[i] *= f'(...).
// ReLU/LeakyReLU use the pre-activation Z; the others use the post-activation A.
//-----------------------------------------------------------
static void cuda_launch_deriv(Activation_type act, float *dl_dz,
                              const float *a, const float *z, int n)
{
    switch (act)
    {
        case ACTIVATION_SIGMOID:    cuda_deriv_sigmoid<<<cuda_blocks(n), CUDA_THREADS>>>(dl_dz, a, n); break;
        case ACTIVATION_RELU:       cuda_deriv_relu<<<cuda_blocks(n), CUDA_THREADS>>>(dl_dz, z, n); break;
        case ACTIVATION_LEAKY_RELU: cuda_deriv_leaky_relu<<<cuda_blocks(n), CUDA_THREADS>>>(dl_dz, z, n); break;
        case ACTIVATION_TANH:       cuda_deriv_tanh<<<cuda_blocks(n), CUDA_THREADS>>>(dl_dz, a, n); break;
        case ACTIVATION_SOFTSIGN:   cuda_deriv_softsign<<<cuda_blocks(n), CUDA_THREADS>>>(dl_dz, a, n); break;
        default:                    break;
    }
}

//================================================================================================
// Inference
//================================================================================================

//-----------------------------------------------------------
// Ensure the cached inference ping-pong buffers hold at least `bytes` each,
// growing (never shrinking) as needed.  Avoids cudaMalloc/cudaFree per predict call.
//-----------------------------------------------------------
static int cuda_ensure_infer_buffers(size_t bytes)
{
    if (bytes <= g_cuda.infer_buf_bytes && g_cuda.infer_bufA && g_cuda.infer_bufB)
        return 1;

    if (g_cuda.infer_bufA) { cudaFree(g_cuda.infer_bufA); g_cuda.infer_bufA = NULL; }
    if (g_cuda.infer_bufB) { cudaFree(g_cuda.infer_bufB); g_cuda.infer_bufB = NULL; }
    g_cuda.infer_buf_bytes = 0;

    if (!cuda_check(cudaMalloc((void **)&g_cuda.infer_bufA, bytes), "cudaMalloc(infer A)"))
        return 0;
    if (!cuda_check(cudaMalloc((void **)&g_cuda.infer_bufB, bytes), "cudaMalloc(infer B)"))
    {
        cudaFree(g_cuda.infer_bufA);
        g_cuda.infer_bufA = NULL;
        return 0;
    }
    g_cuda.infer_buf_bytes = bytes;
    return 1;
}

//-----------------------------------------------------------
// Ensure the cached on-device training loss/delta buffers (device targets and
// per-element loss) hold at least `bytes` each, growing as needed.
//-----------------------------------------------------------
static int cuda_ensure_loss_buffers(size_t bytes)
{
    if (bytes <= g_cuda.train_loss_bytes && g_cuda.train_targets && g_cuda.train_lossbuf)
        return 1;

    if (g_cuda.train_targets) { cudaFree(g_cuda.train_targets); g_cuda.train_targets = NULL; }
    if (g_cuda.train_lossbuf) { cudaFree(g_cuda.train_lossbuf); g_cuda.train_lossbuf = NULL; }
    g_cuda.train_loss_bytes = 0;

    if (!cuda_check(cudaMalloc((void **)&g_cuda.train_targets, bytes), "cudaMalloc(train targets)"))
        return 0;
    if (!cuda_check(cudaMalloc((void **)&g_cuda.train_lossbuf, bytes), "cudaMalloc(train lossbuf)"))
    {
        cudaFree(g_cuda.train_targets);
        g_cuda.train_targets = NULL;
        return 0;
    }
    g_cuda.train_loss_bytes = bytes;
    return 1;
}

//-----------------------------------------------------------
// Forward pass for inference using cached ping-pong device scratch buffers.
// Reads batch x input_nodes from host `inputs`, writes batch x output_nodes
// to host `outputs`.  Weights/biases must already be uploaded.
// Returns 1 on success, 0 on failure (caller falls back to CPU).
//-----------------------------------------------------------
static int cuda_forward_infer(PNetwork pnet, const real *inputs, real *outputs, int batch)
{
    int in_first  = pnet->layers[0].node_count;
    int out_final = pnet->layers[pnet->layer_count - 1].node_count;

    int max_nodes = 0;
    for (int i = 0; i < pnet->layer_count; i++)
        if (pnet->layers[i].node_count > max_nodes)
            max_nodes = pnet->layers[i].node_count;

    size_t buf_bytes = (size_t)batch * max_nodes * sizeof(float);
    if (!cuda_ensure_infer_buffers(buf_bytes))
        return 0;

    float *bufA = g_cuda.infer_bufA, *bufB = g_cuda.infer_bufB;

    int ok = 1;
    if (!cuda_check(cudaMemcpy(bufA, inputs, (size_t)batch * in_first * sizeof(float),
                               cudaMemcpyHostToDevice), "cudaMemcpy infer in"))
        ok = 0;

    float *cur = bufA, *nxt = bufB;
    for (int layer = 0; ok && layer < pnet->layer_count - 1; layer++)
    {
        int in_nodes  = pnet->layers[layer].node_count;
        int out_nodes = pnet->layers[layer + 1].node_count;
        const float *W = (const float *)pnet->layers[layer].t_weights->gpu_buf;
        const float *b = (const float *)pnet->layers[layer].t_bias->gpu_buf;

        // Y = X * W^T
        if (!cuda_gemm_rowmajor(0, 1, batch, out_nodes, in_nodes, 1.0f, cur, W, 0.0f, nxt))
        {
            ok = 0;
            break;
        }
        cuda_bias_add<<<cuda_blocks(batch * out_nodes), CUDA_THREADS>>>(nxt, b, batch, out_nodes);
        cuda_launch_activation(pnet->layers[layer + 1].activation, nxt, batch, out_nodes);

        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    if (ok && !CUDA_KERNEL_CHECK("infer kernels"))
        ok = 0;

    if (ok && !cuda_check(cudaMemcpy(outputs, cur, (size_t)batch * out_final * sizeof(float),
                                     cudaMemcpyDeviceToHost), "cudaMemcpy infer out"))
        ok = 0;

    return ok;
}

static int cuda_eval_single(PNetwork pnet)
{
    if (!pnet || !g_cuda.initialized)
        return 0;
    if (!pnet->layers[0].t_weights || !pnet->layers[0].t_weights->gpu_buf)
        return 0;

    const real *in = pnet->layers[0].t_values->values;
    real *out = pnet->layers[pnet->layer_count - 1].t_values->values;
    return cuda_forward_infer(pnet, in, out, 1);
}

static int cuda_eval_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
    if (!pnet || !inputs || !outputs || batch_size <= 0)
        return ERR_NULL_PTR;
    if (!g_cuda.initialized)
        return ERR_FAIL;
    if (!pnet->layers[0].t_weights || !pnet->layers[0].t_weights->gpu_buf)
        return ERR_FAIL;

    if (!cuda_forward_infer((PNetwork)pnet, inputs, outputs, batch_size))
        return ERR_FAIL;
    return ERR_OK;
}

//================================================================================================
// Training
//================================================================================================

//-----------------------------------------------------------
// Allocate (zero-initialized) a device buffer for a tensor if it
// doesn't already have one.  Assumes the tensor's rows/cols already
// reflect the current batch size; batch-size changes free + realloc.
//-----------------------------------------------------------
static int cuda_ensure_buf(PTensor t)
{
    if (!t)
        return 0;
    size_t bytes = (size_t)(t->rows * t->cols) * sizeof(real);
    if (bytes == 0)
        return 1;
    if (t->gpu_buf)
        return 1;
    void *p = NULL;
    if (!cuda_check(cudaMalloc(&p, bytes), "cudaMalloc(train buf)"))
        return 0;
    if (!cuda_check(cudaMemset(p, 0, bytes), "cudaMemset(train buf)"))
    {
        cudaFree(p);
        return 0;
    }
    t->gpu_buf = p;
    return 1;
}

//-----------------------------------------------------------
// Allocate device buffers for all training tensors.  Weights/biases
// already have gpu_buf from upload_network(); this adds buffers for the
// per-batch activations, pre-activations, gradients and optimizer state.
//-----------------------------------------------------------
static int cuda_alloc_training_buffers(PNetwork pnet, int batch_size)
{
    for (int layer = 0; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];

        if (!l->t_batch_values || !cuda_ensure_buf(l->t_batch_values)) return 0;

        if (layer > 0)
        {
            if (!l->t_batch_z     || !cuda_ensure_buf(l->t_batch_z))     return 0;
            if (!l->t_batch_dl_dz || !cuda_ensure_buf(l->t_batch_dl_dz)) return 0;
        }

        if (layer < pnet->layer_count - 1)
        {
            if (!l->t_gradients || !cuda_ensure_buf(l->t_gradients)) return 0;
            if (!l->t_bias_grad || !cuda_ensure_buf(l->t_bias_grad)) return 0;
            // Optimizer state is NULL for optimizers that don't use it (e.g. SGD).
            if (l->t_m      && !cuda_ensure_buf(l->t_m))      return 0;
            if (l->t_v      && !cuda_ensure_buf(l->t_v))      return 0;
            if (l->t_bias_m && !cuda_ensure_buf(l->t_bias_m)) return 0;
            if (l->t_bias_v && !cuda_ensure_buf(l->t_bias_v)) return 0;
        }
    }

    g_cuda.training_buffers_ready = 1;
    g_cuda.training_batch_size = batch_size;
    return 1;
}

//-----------------------------------------------------------
// Forward pass for training: same as inference but uses the per-layer
// t_batch_values buffers and saves pre-activation Z for backprop.
// On entry layer[0].t_batch_values->gpu_buf holds the batch inputs.
//-----------------------------------------------------------
static int cuda_forward_train(PNetwork pnet, int batch)
{
    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer src = &pnet->layers[layer];
        PLayer dst = &pnet->layers[layer + 1];

        const float *X = (const float *)src->t_batch_values->gpu_buf;
        const float *W = (const float *)src->t_weights->gpu_buf;
        const float *b = (const float *)src->t_bias->gpu_buf;
        float *Y = (float *)dst->t_batch_values->gpu_buf;
        float *Z = (float *)dst->t_batch_z->gpu_buf;

        int in_nodes  = src->node_count;
        int out_nodes = dst->node_count;

        // Y = X * W^T
        if (!cuda_gemm_rowmajor(0, 1, batch, out_nodes, in_nodes, 1.0f, X, W, 0.0f, Y))
            return 0;
        cuda_bias_add<<<cuda_blocks(batch * out_nodes), CUDA_THREADS>>>(Y, b, batch, out_nodes);

        // Save pre-activation Z = Y (device-to-device)
        if (!cuda_check(cudaMemcpy(Z, Y, (size_t)batch * out_nodes * sizeof(real),
                                   cudaMemcpyDeviceToDevice), "cudaMemcpy Z"))
            return 0;

        cuda_launch_activation(dst->activation, Y, batch, out_nodes);
    }
    return 1;
}

//-----------------------------------------------------------
// Backward pass.  Delta (= T - Y) must already be in the output layer's
// t_batch_dl_dz buffer.  Mirrors gpu_backward_pass() in the Metal backend.
//-----------------------------------------------------------
static int cuda_backward_pass(PNetwork pnet, int batch)
{
    int layer_count = pnet->layer_count;

    for (int li = layer_count - 1; li >= 1; li--)
    {
        PLayer layer      = &pnet->layers[li];
        PLayer prev_layer = &pnet->layers[li - 1];

        int nodes      = layer->node_count;
        int prev_nodes = prev_layer->node_count;

        float *dl_dz  = (float *)layer->t_batch_dl_dz->gpu_buf;
        float *A      = (float *)layer->t_batch_values->gpu_buf;
        float *Z      = (float *)layer->t_batch_z->gpu_buf;
        float *A_prev = (float *)prev_layer->t_batch_values->gpu_buf;
        float *W      = (float *)prev_layer->t_weights->gpu_buf;
        float *dW     = (float *)prev_layer->t_gradients->gpu_buf;
        float *db     = (float *)prev_layer->t_bias_grad->gpu_buf;

        int n = batch * nodes;

        // Activation derivative.  The output layer uses raw delta = T - Y for all
        // loss types (matching the CPU path), so skip its derivative.
        if (li != layer_count - 1)
            cuda_launch_deriv(layer->activation, dl_dz, A, Z, n);

        // dW += dl_dz^T * A_prev   (accumulate)
        if (!cuda_gemm_rowmajor(1, 0, nodes, prev_nodes, batch, 1.0f, dl_dz, A_prev, 1.0f, dW))
            return 0;

        // bias_grad += col_sum(dl_dz)
        cuda_bias_grad_sum<<<cuda_blocks(nodes), CUDA_THREADS>>>(dl_dz, db, batch, nodes);

        // Propagate delta to previous layer: dl_dz_prev = dl_dz * W
        if (li > 1)
        {
            float *dl_dz_prev = (float *)prev_layer->t_batch_dl_dz->gpu_buf;
            if (!cuda_gemm_rowmajor(0, 0, batch, prev_nodes, nodes, 1.0f, dl_dz, W, 0.0f, dl_dz_prev))
                return 0;
        }
    }
    return 1;
}

//-----------------------------------------------------------
// Optimizer step.  Applies optional gradient clipping, the per-optimizer
// update for weights and biases, then L2/L1 regularization.  Mirrors
// gpu_optimizer_step() in the Metal backend.
//-----------------------------------------------------------
static void cuda_optimizer_step(PNetwork pnet)
{
    Optimizer_type opt = pnet->optimizer;
    float lr    = (float)pnet->learning_rate;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;
    float beta_m   = 0.9f;   // momentum beta
    float beta_rms = 0.9f;   // rmsprop beta

    // Adam bias-correction scalars
    float bc1 = 0.0f, bc2 = 0.0f;
    if (opt == OPT_ADAM)
    {
        float t = (float)pnet->train_iteration;
        bc1 = 1.0f / (1.0f - powf(beta1, t));
        bc2 = 1.0f / (1.0f - powf(beta2, t));
    }

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];

        float *W  = (float *)l->t_weights->gpu_buf;
        float *g  = (float *)l->t_gradients->gpu_buf;
        float *b  = (float *)l->t_bias->gpu_buf;
        float *bg = (float *)l->t_bias_grad->gpu_buf;

        int w_count = l->t_weights->rows * l->t_weights->cols;
        int b_count = l->t_bias->cols;

        // Optional gradient clipping
        if (pnet->max_gradient > 0.0f)
        {
            float mg = (float)pnet->max_gradient;
            cuda_gradient_clip<<<cuda_blocks(w_count), CUDA_THREADS>>>(g,  mg, w_count);
            cuda_gradient_clip<<<cuda_blocks(b_count), CUDA_THREADS>>>(bg, mg, b_count);
        }

        switch (opt)
        {
            case OPT_SGD:
                cuda_sgd_update<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, g, lr, w_count);
                cuda_sgd_update<<<cuda_blocks(b_count), CUDA_THREADS>>>(b, bg, lr, b_count);
                break;

            case OPT_MOMENTUM:
            {
                float *m  = (float *)l->t_m->gpu_buf;
                float *bm = (float *)l->t_bias_m->gpu_buf;
                cuda_momentum_update<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, g, m, lr, beta_m, w_count);
                cuda_momentum_update<<<cuda_blocks(b_count), CUDA_THREADS>>>(b, bg, bm, lr, beta_m, b_count);
                break;
            }

            case OPT_ADAGRAD:
            {
                float *v  = (float *)l->t_v->gpu_buf;
                float *bv = (float *)l->t_bias_v->gpu_buf;
                cuda_adagrad_update<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, g, v, lr, eps, w_count);
                cuda_adagrad_update<<<cuda_blocks(b_count), CUDA_THREADS>>>(b, bg, bv, lr, eps, b_count);
                break;
            }

            case OPT_RMSPROP:
            {
                float *v  = (float *)l->t_v->gpu_buf;
                float *bv = (float *)l->t_bias_v->gpu_buf;
                cuda_rmsprop_update<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, g, v, lr, beta_rms, eps, w_count);
                cuda_rmsprop_update<<<cuda_blocks(b_count), CUDA_THREADS>>>(b, bg, bv, lr, beta_rms, eps, b_count);
                break;
            }

            case OPT_ADAM:
            {
                float *m  = (float *)l->t_m->gpu_buf;
                float *v  = (float *)l->t_v->gpu_buf;
                float *bm = (float *)l->t_bias_m->gpu_buf;
                float *bv = (float *)l->t_bias_v->gpu_buf;
                cuda_adam_update<<<cuda_blocks(w_count), CUDA_THREADS>>>(
                    W, g, m, v, lr, beta1, beta2, bc1, bc2, eps, w_count);
                cuda_adam_update<<<cuda_blocks(b_count), CUDA_THREADS>>>(
                    b, bg, bm, bv, lr, beta1, beta2, bc1, bc2, eps, b_count);
                break;
            }

            default:
                break;
        }

        // L2 regularization (weight decay) on weights
        if (pnet->l2_lambda > 0.0f)
        {
            float decay = 1.0f - (float)(pnet->learning_rate * pnet->l2_lambda);
            cuda_l2_regularize<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, decay, w_count);
        }
        // L1 regularization on weights
        if (pnet->l1_lambda > 0.0f)
        {
            float l1 = (float)(pnet->learning_rate * pnet->l1_lambda);
            cuda_l1_regularize<<<cuda_blocks(w_count), CUDA_THREADS>>>(W, l1, w_count);
        }
    }
}

//-----------------------------------------------------------
// One mini-batch: forward + backward + optimizer.  Mirrors
// metal_train_batch().  Returns 1 if handled, 0 to fall back to CPU.
//-----------------------------------------------------------
static int cuda_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out)
{
    if (!pnet || !g_cuda.initialized)
        return 0;
    if (!pnet->layers[0].t_weights || !pnet->layers[0].t_weights->gpu_buf)
        return 0;
    if (!pnet->layers[0].t_batch_values)
        return 0;

    int out_idx       = pnet->layer_count - 1;
    int input_nodes   = pnet->layers[0].node_count;
    int output_nodes  = pnet->layers[out_idx].node_count;

    // Lazy-alloc training buffers (or reallocate if the batch size changed).
    if (!g_cuda.training_buffers_ready || g_cuda.training_batch_size != batch_size)
    {
        cuda_free_training_buffers(pnet);
        if (!cuda_alloc_training_buffers(pnet, batch_size))
            return 0;
    }

    // Defensive re-allocation for stale state across network lifetimes.
    if (!pnet->layers[0].t_batch_values->gpu_buf ||
        !pnet->layers[out_idx].t_batch_values || !pnet->layers[out_idx].t_batch_values->gpu_buf ||
        !pnet->layers[out_idx].t_batch_dl_dz || !pnet->layers[out_idx].t_batch_dl_dz->gpu_buf)
    {
        cuda_free_training_buffers(pnet);
        if (!cuda_alloc_training_buffers(pnet, batch_size))
            return 0;
    }

    // --- Zero gradient buffers ---
    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (l->t_gradients && l->t_gradients->gpu_buf)
            cudaMemset(l->t_gradients->gpu_buf, 0,
                       (size_t)(l->t_gradients->rows * l->t_gradients->cols) * sizeof(real));
        if (l->t_bias_grad && l->t_bias_grad->gpu_buf)
            cudaMemset(l->t_bias_grad->gpu_buf, 0, (size_t)l->t_bias_grad->cols * sizeof(real));
    }

    // --- Copy batch inputs to GPU (host t_batch_values->values -> device) ---
    if (!cuda_check(cudaMemcpy(pnet->layers[0].t_batch_values->gpu_buf,
                               pnet->layers[0].t_batch_values->values,
                               (size_t)batch_size * input_nodes * sizeof(real),
                               cudaMemcpyHostToDevice), "cudaMemcpy train in"))
        return 0;

    // --- Forward pass (saves Z) ---
    if (!cuda_forward_train(pnet, batch_size))
        return 0;

    // --- Delta = T - Y and loss, computed on-device ---
    // Upload targets once, then a single kernel writes delta into the output layer's
    // dl_dz buffer and the per-element loss into a scratch buffer, which cuBLAS reduces.
    {
        int n = batch_size * output_nodes;
        size_t obytes = (size_t)n * sizeof(real);

        if (!cuda_ensure_loss_buffers(obytes))
            return 0;

        // Upload this batch's targets to the device.
        if (!cuda_check(cudaMemcpy(g_cuda.train_targets, batch_targets->values, obytes,
                                   cudaMemcpyHostToDevice), "cudaMemcpy targets h2d"))
            return 0;

        const float *Y     = (const float *)pnet->layers[out_idx].t_batch_values->gpu_buf;
        float       *delta = (float *)pnet->layers[out_idx].t_batch_dl_dz->gpu_buf;

        if (pnet->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY)
            cuda_xent_delta_loss<<<cuda_blocks(n), CUDA_THREADS>>>(
                Y, g_cuda.train_targets, delta, g_cuda.train_lossbuf, n);
        else
            cuda_mse_delta_loss<<<cuda_blocks(n), CUDA_THREADS>>>(
                Y, g_cuda.train_targets, delta, g_cuda.train_lossbuf, n);

        if (!CUDA_KERNEL_CHECK("delta/loss kernel"))
            return 0;

        // Reduce the per-element loss.  All loss terms are >= 0, so the sum of absolute
        // values returned by cublasSasum equals the true sum.
        float loss_sum = 0.0f;
        if (!cublas_check(cublasSasum(g_cuda.cublas, n, g_cuda.train_lossbuf, 1, &loss_sum),
                          "cublasSasum(loss)"))
            return 0;

        // Match the CPU normalization exactly.
        real total_loss = (pnet->loss_type == LOSS_MSE)
            ? (real)loss_sum / (real)(batch_size * output_nodes)
            : (real)loss_sum / (real)batch_size;

        if (loss_out) *loss_out = total_loss;
    }

    // --- Backward pass ---
    if (!cuda_backward_pass(pnet, batch_size))
        return 0;

    // --- Increment iteration (Adam bias correction) then optimizer step ---
    pnet->train_iteration++;
    cuda_optimizer_step(pnet);

    // Surface any kernel launch/exec error from this batch.
    if (!cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(train_batch)"))
        return 0;

    return 1;
}

extern "C" GpuBackend cuda_backend = {
    cuda_backend_init,
    cuda_upload_network,
    cuda_free_network,
    cuda_sync_weights,
    cuda_release_buffer,
    cuda_eval_single,
    cuda_eval_batch,
    cuda_train_batch,
};

#endif /* USE_CUDA */
