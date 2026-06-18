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
#include <math.h>

extern "C" __global__ void cuda_bias_add(
    float *buf, const float *bias, int batch_size, int cols)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * cols;
    if (gid < total)
    {
        int n = gid % cols;
        buf[gid] += bias[n];
    }
}

extern "C" __global__ void cuda_activation_sigmoid(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = buf[i];
        buf[i] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" __global__ void cuda_activation_relu(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = (buf[i] > 0.0f) ? buf[i] : 0.0f;
}

extern "C" __global__ void cuda_activation_leaky_relu(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = buf[i];
        buf[i] = (x > 0.0f) ? x : 0.01f * x;
    }
}

extern "C" __global__ void cuda_activation_tanh(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = tanhf(buf[i]);
}

extern "C" __global__ void cuda_activation_softsign(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = buf[i];
        buf[i] = x / (1.0f + fabsf(x));
    }
}

extern "C" __global__ void cuda_softmax_rows(float *buf, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows)
        return;

    int offset = r * cols;
    float mx = -INFINITY;
    for (int n = 0; n < cols; n++)
        mx = fmaxf(mx, buf[offset + n]);

    float s = 0.0f;
    for (int n = 0; n < cols; n++)
    {
        float e = expf(buf[offset + n] - mx);
        buf[offset + n] = e;
        s += e;
    }

    if (s > 0.0f)
    {
        for (int n = 0; n < cols; n++)
            buf[offset + n] /= s;
    }
}

extern "C" __global__ void cuda_deriv_sigmoid(float *dl_dz, const float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dl_dz[i] *= a[i] * (1.0f - a[i]);
}

extern "C" __global__ void cuda_deriv_relu(float *dl_dz, const float *z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dl_dz[i] *= (z[i] > 0.0f ? 1.0f : 0.0f);
}

extern "C" __global__ void cuda_deriv_leaky_relu(float *dl_dz, const float *z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dl_dz[i] *= (z[i] > 0.0f ? 1.0f : 0.01f);
}

extern "C" __global__ void cuda_deriv_tanh(float *dl_dz, const float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dl_dz[i] *= (1.0f - a[i] * a[i]);
}

extern "C" __global__ void cuda_deriv_softsign(float *dl_dz, const float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float d = 1.0f - fabsf(a[i]);
        dl_dz[i] *= d * d;
    }
}

extern "C" __global__ void cuda_bias_grad_sum(
    const float *dl_dz, float *bias_grad, int batch_size, int nodes)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= nodes)
        return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++)
        sum += dl_dz[b * nodes + nid];
    bias_grad[nid] += sum;
}

extern "C" __global__ void cuda_gradient_clip(float *grads, float max_norm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float g = grads[i];
        if (g > max_norm) g = max_norm;
        if (g < -max_norm) g = -max_norm;
        grads[i] = g;
    }
}

extern "C" __global__ void cuda_sgd_update(
    float *weights, const float *grads, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        weights[i] += lr * grads[i];
}

extern "C" __global__ void cuda_momentum_update(
    float *weights, const float *grads, float *momentum, float lr, float beta, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float m = beta * momentum[i] + (1.0f - beta) * grads[i];
        momentum[i] = m;
        weights[i] += lr * m;
    }
}

extern "C" __global__ void cuda_adagrad_update(
    float *weights, const float *grads, float *v, float lr, float eps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float g = grads[i];
        v[i] += g * g;
        weights[i] += (lr * g) / (sqrtf(v[i]) + eps);
    }
}

extern "C" __global__ void cuda_rmsprop_update(
    float *weights, const float *grads, float *v, float lr, float beta, float eps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float g = grads[i];
        v[i] = beta * v[i] + (1.0f - beta) * g * g;
        weights[i] += (lr * g) / (sqrtf(v[i]) + eps);
    }
}

extern "C" __global__ void cuda_adam_update(
    float *weights, const float *grads, float *m, float *v,
    float lr, float beta1, float beta2, float m_hat_scale, float v_hat_scale, float eps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float g = grads[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] * m_hat_scale;
        float v_hat = v[i] * v_hat_scale;
        weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void cuda_l2_regularize(float *weights, float decay, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        weights[i] *= (1.0f - decay);
}

extern "C" __global__ void cuda_l1_regularize(float *weights, float l1_scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float w = weights[i];
        float s = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        weights[i] -= l1_scale * s;
    }
}

#endif /* USE_CUDA */
