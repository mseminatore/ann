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

#pragma once

#ifndef __ANN_GPU_BACKEND_H
#define __ANN_GPU_BACKEND_H

//================================================================================================
// GPU Backend Abstraction (internal)
//================================================================================================
// This header defines the GpuBackend vtable used by ann.c to dispatch GPU operations.
// It is an internal interface — not part of the public libann API.
//
// Each backend (Metal, CUDA, ...) implements all slots and registers a static GpuBackend instance.
// ann_gpu_init() selects the appropriate backend at startup and sets g_gpu_backend.
//
// Adding a new backend requires only:
//   1. A new implementation file (e.g., tensor_cuda.cu) that fills in all vtable slots
//   2. A new #ifdef block in ann_gpu_init() (in ann.c)
//   3. A tensor_gpu_release_buffer() implementation (called from tensor_free())
//
// ann.c never contains backend-specific code beyond the #ifdef in ann_gpu_init().
//================================================================================================

#include "ann.h"

/**
 * GPU backend vtable.
 *
 * All function pointers must be non-NULL when a backend is active.
 * If a backend does not support an operation it should return 0/ERR_FAIL gracefully.
 */
typedef struct {
    /**
     * Initialize the GPU device and compile/load kernels.
     * @return 1 on success, 0 on failure.
     */
    int (*init)(void);

    /**
     * Upload all network weights and biases to the GPU.
     * Sets gpu_buf on each layer's t_weights and t_bias tensors.
     * @return ERR_OK on success, ERR_NULL_PTR / ERR_FAIL on error.
     */
    int (*upload_network)(PNetwork pnet);

    /**
     * Release all GPU buffers associated with the network.
     * Sets gpu_buf to NULL on each tensor after freeing.
     */
    void (*free_network)(PNetwork pnet);

    /**
     * Download trained weights from GPU back to CPU.
     * Call after training to make weights available for save/predict.
     */
    void (*sync_weights)(PNetwork pnet);

    /**
     * Release a backend-specific tensor GPU buffer.
     * Called from tensor_free() when t->gpu_buf is set.
     */
    void (*release_buffer)(void *gpu_buf);

    /**
     * Single-sample GPU forward pass.
     * Reads input from pnet->layers[0].t_values; writes output to last layer t_values.
     * @return 1 if GPU handled it, 0 to fall through to CPU.
     */
    int (*eval_single)(PNetwork pnet);

    /**
     * Batched GPU forward pass.
     * @param inputs  Row-major [batch_size x input_nodes]
     * @param outputs Row-major [batch_size x output_nodes] (written here)
     * @return ERR_OK on success.
     */
    int (*eval_batch)(const PNetwork pnet, const real *inputs, real *outputs, int batch_size);

    /**
     * One mini-batch: forward pass + backward pass + optimizer update.
     * Increments pnet->train_iteration for bias correction (Adam).
     * @param batch_targets Tensor of target labels for this batch
     * @param batch_size    Number of samples in the batch
     * @param loss_out      Written with the mean loss for this batch
     * @return 1 if GPU handled the batch, 0 to fall back to CPU.
     */
    int (*train_batch)(PNetwork pnet, PTensor batch_targets, int batch_size, real *loss_out);
} GpuBackend;

/**
 * Active GPU backend.  NULL means no GPU is available or initialized.
 * Set by ann_gpu_init(); read by ann.c dispatch logic and ann_gpu_* public API.
 */
extern GpuBackend *g_gpu_backend;

#endif /* __ANN_GPU_BACKEND_H */
