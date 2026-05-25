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

#include <metal_stdlib>
using namespace metal;

//-----------------------------------------------------------
// Bias broadcast: add bias vector to each row of a matrix
// buf[b * cols + n] += bias[n]
// Dispatched as (batch_size x node_count) 2D grid
//-----------------------------------------------------------
kernel void bias_add(
    device float *buf     [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    uint2 gid             [[thread_position_in_grid]],
    uint2 dims            [[threads_per_grid]])   // dims.x = cols (nodes)
{
    uint b = gid.y;   // batch index
    uint n = gid.x;   // node index
    buf[b * dims.x + n] += bias[n];
}

//-----------------------------------------------------------
// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
// Applied element-wise over a flat buffer of length `count`
//-----------------------------------------------------------
kernel void activation_sigmoid(
    device float *buf [[buffer(0)]],
    uint gid          [[thread_position_in_grid]])
{
    buf[gid] = 1.0f / (1.0f + exp(-buf[gid]));
}

//-----------------------------------------------------------
// ReLU activation: f(x) = max(0, x)
//-----------------------------------------------------------
kernel void activation_relu(
    device float *buf [[buffer(0)]],
    uint gid          [[thread_position_in_grid]])
{
    buf[gid] = max(0.0f, buf[gid]);
}

//-----------------------------------------------------------
// Leaky ReLU activation: f(x) = x > 0 ? x : 0.01 * x
//-----------------------------------------------------------
kernel void activation_leaky_relu(
    device float *buf [[buffer(0)]],
    uint gid          [[thread_position_in_grid]])
{
    float x = buf[gid];
    buf[gid] = x > 0.0f ? x : 0.01f * x;
}

//-----------------------------------------------------------
// Tanh activation: f(x) = tanh(x)
//-----------------------------------------------------------
kernel void activation_tanh(
    device float *buf [[buffer(0)]],
    uint gid          [[thread_position_in_grid]])
{
    buf[gid] = tanh(buf[gid]);
}

//-----------------------------------------------------------
// Softsign activation: f(x) = x / (1 + |x|)
//-----------------------------------------------------------
kernel void activation_softsign(
    device float *buf [[buffer(0)]],
    uint gid          [[thread_position_in_grid]])
{
    float x = buf[gid];
    buf[gid] = x / (1.0f + fabs(x));
}

//-----------------------------------------------------------
// Softmax: numerically stable per-row softmax
// Each row of length `cols` is processed by one thread.
// One thread per row is dispatched — no shared memory needed.
// Correct for any output size; suitable for small-to-medium layers.
//-----------------------------------------------------------
kernel void softmax_rows(
    device float *buf     [[buffer(0)]],
    constant uint &cols   [[buffer(1)]],
    uint bid              [[threadgroup_position_in_grid]])
{
    uint offset = bid * cols;

    float mx = -INFINITY;
    for (uint n = 0; n < cols; n++)
        mx = max(mx, buf[offset + n]);

    float s = 0.0f;
    for (uint n = 0; n < cols; n++) {
        float e = exp(buf[offset + n] - mx);
        buf[offset + n] = e;
        s += e;
    }

    if (s > 0.0f)
        for (uint n = 0; n < cols; n++)
            buf[offset + n] /= s;
}

//-----------------------------------------------------------
// Training: Activation Derivatives (backward pass)
//-----------------------------------------------------------

// Sigmoid derivative: dl_dz = dl_da * a * (1 - a)
kernel void deriv_sigmoid(
    device float *dl_dz   [[buffer(0)]],
    device const float *a [[buffer(1)]],
    uint gid              [[thread_position_in_grid]])
{
    dl_dz[gid] *= a[gid] * (1.0f - a[gid]);
}

// ReLU derivative: dl_dz = dl_da * (z > 0 ? 1 : 0)
kernel void deriv_relu(
    device float *dl_dz   [[buffer(0)]],
    device const float *z [[buffer(1)]],
    uint gid              [[thread_position_in_grid]])
{
    dl_dz[gid] *= (z[gid] > 0.0f ? 1.0f : 0.0f);
}

// LeakyReLU derivative: dl_dz = dl_da * (z > 0 ? 1 : 0.01)
kernel void deriv_leaky_relu(
    device float *dl_dz   [[buffer(0)]],
    device const float *z [[buffer(1)]],
    uint gid              [[thread_position_in_grid]])
{
    dl_dz[gid] *= (z[gid] > 0.0f ? 1.0f : 0.01f);
}

// Tanh derivative: dl_dz = dl_da * (1 - a^2)
kernel void deriv_tanh(
    device float *dl_dz   [[buffer(0)]],
    device const float *a [[buffer(1)]],
    uint gid              [[thread_position_in_grid]])
{
    float tanh_val = a[gid];
    dl_dz[gid] *= (1.0f - tanh_val * tanh_val);
}

// Softsign derivative: dl_dz = dl_da * (1 - |a|)^2
kernel void deriv_softsign(
    device float *dl_dz   [[buffer(0)]],
    device const float *a [[buffer(1)]],
    uint gid              [[thread_position_in_grid]])
{
    float abs_a = fabs(a[gid]);
    float denom = (1.0f - abs_a);
    dl_dz[gid] *= denom * denom;
}

//-----------------------------------------------------------
// Gradient aggregation: sum gradients across batch dimension
// dl_dz is [batch_size × nodes]
// bias_grad is [nodes] — accumulate
// One thread per output node, loop over batch
//-----------------------------------------------------------
kernel void bias_grad_sum(
    device const float *dl_dz  [[buffer(0)]],
    device float *bias_grad    [[buffer(1)]],
    constant uint &batch_size  [[buffer(2)]],
    constant uint &nodes       [[buffer(3)]],
    uint nid                   [[thread_position_in_grid]])
{
    if (nid >= nodes) return;
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++)
        sum += dl_dz[b * nodes + nid];
    bias_grad[nid] += sum;
}

//-----------------------------------------------------------
// Optimizer updates
//-----------------------------------------------------------

// SGD: w = w + lr * dw
kernel void sgd_update(
    device float *weights  [[buffer(0)]],
    device const float *grads [[buffer(1)]],
    constant float &lr     [[buffer(2)]],
    uint gid               [[thread_position_in_grid]])
{
    weights[gid] += lr * grads[gid];
}

// Momentum: m = beta*m + (1-beta)*g; w = w + lr*m
kernel void momentum_update(
    device float *weights      [[buffer(0)]],
    device const float *grads  [[buffer(1)]],
    device float *momentum     [[buffer(2)]],
    constant float &lr         [[buffer(3)]],
    constant float &beta       [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]])
{
    float g = grads[gid];
    float m = beta * momentum[gid] + (1.0f - beta) * g;
    momentum[gid] = m;
    weights[gid] += lr * m;
}

// AdaGrad: v = v + g^2; w = w + (lr * g) / (sqrt(v) + eps)
kernel void adagrad_update(
    device float *weights      [[buffer(0)]],
    device const float *grads  [[buffer(1)]],
    device float *v            [[buffer(2)]],
    constant float &lr         [[buffer(3)]],
    constant float &eps        [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]])
{
    float g = grads[gid];
    v[gid] += g * g;
    weights[gid] += (lr * g) / (sqrt(v[gid]) + eps);
}

// RMSProp: v = beta*v + (1-beta)*g^2; w = w + (lr * g) / (sqrt(v) + eps)
kernel void rmsprop_update(
    device float *weights      [[buffer(0)]],
    device const float *grads  [[buffer(1)]],
    device float *v            [[buffer(2)]],
    constant float &lr         [[buffer(3)]],
    constant float &beta       [[buffer(4)]],
    constant float &eps        [[buffer(5)]],
    uint gid                   [[thread_position_in_grid]])
{
    float g = grads[gid];
    v[gid] = beta * v[gid] + (1.0f - beta) * g * g;
    weights[gid] += (lr * g) / (sqrt(v[gid]) + eps);
}

// Adam: m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2;
//       m_hat = m / (1 - beta1^t); v_hat = v / (1 - beta2^t);
//       w = w - lr * m_hat / (sqrt(v_hat) + eps)
kernel void adam_update(
    device float *weights       [[buffer(0)]],
    device const float *grads   [[buffer(1)]],
    device float *m             [[buffer(2)]],
    device float *v             [[buffer(3)]],
    constant float &lr          [[buffer(4)]],
    constant float &beta1       [[buffer(5)]],
    constant float &beta2       [[buffer(6)]],
    constant float &m_hat_scale [[buffer(7)]],
    constant float &v_hat_scale [[buffer(8)]],
    constant float &eps         [[buffer(9)]],
    uint gid                    [[thread_position_in_grid]])
{
    float g = grads[gid];
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * g;
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * g * g;
    float m_hat = m[gid] * m_hat_scale;
    float v_hat = v[gid] * v_hat_scale;
    weights[gid] -= lr * m_hat / (sqrt(v_hat) + eps);
}

//-----------------------------------------------------------
// Regularization
//-----------------------------------------------------------

// Gradient clipping: g = clamp(g, -max_norm, max_norm)
kernel void gradient_clip(
    device float *grads    [[buffer(0)]],
    constant float &max_norm [[buffer(1)]],
    uint gid               [[thread_position_in_grid]])
{
    float g = grads[gid];
    grads[gid] = clamp(g, -max_norm, max_norm);
}

// L2 regularization (weight decay): w = w * (1 - decay)
kernel void l2_regularize(
    device float *weights     [[buffer(0)]],
    constant float &decay     [[buffer(1)]],
    uint gid                  [[thread_position_in_grid]])
{
    weights[gid] *= (1.0f - decay);
}

// L1 regularization: w = w - lr * l1 * sign(w)
kernel void l1_regularize(
    device float *weights     [[buffer(0)]],
    constant float &l1_scale  [[buffer(1)]],
    uint gid                  [[thread_position_in_grid]])
{
    float w = weights[gid];
    weights[gid] -= l1_scale * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
}
