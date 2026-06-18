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
#include <string.h>

#include "ann.h"
#include "ann_gpu_backend.h"

typedef struct {
    int initialized;
    int device_id;
    cublasHandle_t cublas;
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

static int cuda_backend_init(void)
{
    if (g_cuda.initialized)
        return 1;

    int device_count = 0;
    if (!cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
        return 0;
    if (device_count <= 0)
        return 0;

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

static void cuda_free_network(PNetwork pnet)
{
    if (!pnet)
        return;

    for (int layer = 0; layer < pnet->layer_count - 1; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (l->t_weights && l->t_weights->gpu_buf)
        {
            cudaFree(l->t_weights->gpu_buf);
            l->t_weights->gpu_buf = NULL;
        }
        if (l->t_bias && l->t_bias->gpu_buf)
        {
            cudaFree(l->t_bias->gpu_buf);
            l->t_bias->gpu_buf = NULL;
        }
    }
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

static int cuda_eval_single(PNetwork pnet)
{
    (void)pnet;
    return 0;
}

static int cuda_eval_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
    (void)pnet;
    (void)inputs;
    (void)outputs;
    (void)batch_size;
    return ERR_FAIL;
}

static int cuda_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out)
{
    (void)pnet;
    (void)batch_targets;
    (void)batch_size;
    (void)loss_out;
    return 0;
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
