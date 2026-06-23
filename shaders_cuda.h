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

#ifndef __SHADERS_CUDA_H
#define __SHADERS_CUDA_H

//================================================================================================
// CUDA kernel prototypes (internal)
//================================================================================================
// Declares the __global__ kernels defined in shaders.cu so they can be launched from
// tensor_cuda.cu.  Both files are compiled by nvcc, so the launch (<<<...>>>) lives in
// tensor_cuda.cu while the kernel bodies live in shaders.cu.
//
// All kernels are float-only (matching the Metal shaders).  A double (real) build of the
// GPU backend is not currently supported — see docs/GPU_CUDA.md.
//================================================================================================

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// Bias add: buf[b*cols + n] += bias[n]   (one thread per element)
__global__ void cuda_bias_add(float *buf, const float *bias, int batch_size, int cols);

// Element-wise activations (one thread per element, in-place on buf[0..n))
__global__ void cuda_activation_sigmoid(float *buf, int n);
__global__ void cuda_activation_relu(float *buf, int n);
__global__ void cuda_activation_leaky_relu(float *buf, int n);
__global__ void cuda_activation_tanh(float *buf, int n);
__global__ void cuda_activation_softsign(float *buf, int n);

// Row-wise softmax (one thread per row)
__global__ void cuda_softmax_rows(float *buf, int rows, int cols);

// Activation derivatives: dl_dz[i] *= f'(...)   (one thread per element)
__global__ void cuda_deriv_sigmoid(float *dl_dz, const float *a, int n);
__global__ void cuda_deriv_relu(float *dl_dz, const float *z, int n);
__global__ void cuda_deriv_leaky_relu(float *dl_dz, const float *z, int n);
__global__ void cuda_deriv_tanh(float *dl_dz, const float *a, int n);
__global__ void cuda_deriv_softsign(float *dl_dz, const float *a, int n);

// Bias gradient reduction: bias_grad[n] += sum_b dl_dz[b*nodes + n]  (one thread per node)
__global__ void cuda_bias_grad_sum(const float *dl_dz, float *bias_grad, int batch_size, int nodes);

// Gradient clipping to [-max_norm, max_norm]  (one thread per element)
__global__ void cuda_gradient_clip(float *grads, float max_norm, int n);

// Optimizer updates (one thread per element)
__global__ void cuda_sgd_update(float *weights, const float *grads, float lr, int n);
__global__ void cuda_momentum_update(float *weights, const float *grads, float *momentum,
                                     float lr, float beta, int n);
__global__ void cuda_adagrad_update(float *weights, const float *grads, float *v,
                                    float lr, float eps, int n);
__global__ void cuda_rmsprop_update(float *weights, const float *grads, float *v,
                                    float lr, float beta, float eps, int n);
__global__ void cuda_adam_update(float *weights, const float *grads, float *m, float *v,
                                 float lr, float beta1, float beta2,
                                 float m_hat_scale, float v_hat_scale, float eps, int n);

// Regularization (one thread per element)
__global__ void cuda_l2_regularize(float *weights, float decay, int n);
__global__ void cuda_l1_regularize(float *weights, float l1_scale, int n);

#ifdef __cplusplus
}
#endif

#endif /* USE_CUDA */

#endif /* __SHADERS_CUDA_H */
