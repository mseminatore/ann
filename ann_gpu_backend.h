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
// Each backend (Metal, Vulkan, CUDA) implements all slots and registers a GpuBackend instance.
// ann_gpu_init() tries the compiled-in backends in order (Metal, Vulkan, CUDA), or only the
// one the ANN_GPU environment variable names, and sets g_gpu_backend to the first that starts.
//
// Adding a new backend requires only:
//   1. A new implementation file (e.g., tensor_cuda.cu) that fills in all vtable slots
//   2. An entry in the backend list in ann_gpu_init() (in ann.c)
//   3. Its USE_xxx define in the USE_GPU condition in ann_config.h
//
// ann.c never contains backend-specific code beyond that list.
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
     * Backend name ("Metal", "Vulkan", "CUDA"), reported by ann_get_backend_info()
     * and matched (ignoring case) against the ANN_GPU environment variable.
     */
    const char *name;

    /**
     * Initialize the GPU device and compile/load kernels.
     * @return 1 on success, 0 on failure.
     */
    int (*init)(void);

    /**
     * Upload all network parameters to the GPU.
     * Sets gpu_buf on each layer's params[] values (ann.c checks the first
     * parameter of the first layer that has any to decide whether to dispatch).
     * @return ERR_OK on success, ERR_INVALID if the network has a layer kind the
     *         backend does not implement, ERR_NULL_PTR / ERR_FAIL on error.
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
     * Submit one mini-batch: forward pass + backward pass + optimizer update.
     * Increments pnet->train_iteration for bias correction (Adam). The batch
     * is read from layers[0].t_batch_values, batch_targets and the layers'
     * dropout masks before this returns, so the caller can refill them at
     * once; the GPU may still be running the step (it queues behind earlier
     * ones), and its loss is collected by train_finish().
     * @param batch_targets Tensor of target labels for this batch
     * @param batch_size    Number of samples in the batch
     * @return 1 if the GPU took the batch, 0 to fall back to the CPU (call
     *         train_finish() first, to collect the steps already submitted).
     */
    int (*train_batch)(PNetwork pnet, PTensor batch_targets, int batch_size);

    /**
     * Wait for every step train_batch() submitted.
     * @param loss_sum Written with the sum of their batch losses (each the
     *                 batch mean, as on the CPU) since the last train_finish()
     * @return 1 if they all completed, 0 if any failed (the weights then hold
     *         the completed steps' updates).
     */
    int (*train_finish)(PNetwork pnet, real *loss_sum);
} GpuBackend;

/**
 * Apply the output activation's derivative to an MSE output error, in place.
 * On entry delta = T - Y; on exit delta = -dL/dz for L = 1/2 * sum((t - y)^2).
 * Implemented in ann.c and shared by the CPU path and GPU backends.
 *
 * @param act        Output layer activation
 * @param delta      Row-major [batch_size x nodes], modified in place
 * @param Y          Output activations [batch_size x nodes]
 * @param Z          Output pre-activations [batch_size x nodes] (ReLU variants)
 */
void ann_mse_output_delta(Activation_type act, real *delta, const real *Y, const real *Z,
                          int batch_size, int nodes);

/**
 * One output's cross-entropy loss term: -t*log(y) for softmax outputs, binary
 * cross-entropy for sigmoid outputs, with y clamped away from 0 and 1.
 * Implemented in ann.c and shared by the CPU path and GPU backends.
 */
real ann_cross_entropy_term(Activation_type output_act, real t, real y);

/**
 * Active GPU backend.  NULL means no GPU is available or initialized.
 * Set by ann_gpu_init(); read by ann.c dispatch logic and ann_gpu_* public API.
 */
extern GpuBackend *g_gpu_backend;

#endif /* __ANN_GPU_BACKEND_H */
