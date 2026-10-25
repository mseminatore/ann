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
// ann_rt.h: a small, header-only inference runtime for models exported by ann_export_c_header()
//
// Copy this file next to the generated model header. The int8 path needs only <stddef.h> and
// <stdint.h>, so it builds freestanding; it never allocates and never recurses: the caller
// passes a workspace buffer whose size the model header defines.
//
// Two model forms share the layer description:
//   int8:    weights are int8 with one scale per output channel, activations are int8 with one
//            scale and zero point per tensor, and sums are int32. Results are rescaled with a
//            fixed-point multiplier and shift, so the int8 path uses integer math only.
//   float32: the trained weights as floats, for targets with an FPU.
//
// Define ANN_RT_NO_FLOAT to compile out every function that uses float math (the float model,
// and the int8 model's float input/output helpers). The int8 path does not use <math.h>.
//
// libann compiles this same file for ann_quantized_predict(), so a model behaves the same in
// the library and on the device.
//================================================================================================

#ifndef ANN_RT_H
#define ANN_RT_H

#include <stddef.h>
#include <stdint.h>

#ifndef ANN_RT_NO_FLOAT
#	include <math.h>
#endif

// 2 added ANN_RT_UPSAMPLE2D, 3 ANN_RT_GLOBAL_AVGPOOL2D; generated headers check
// for the version they need
#define ANN_RT_VERSION 3

#if defined(_MSC_VER) && !defined(__cplusplus)
#	define ANN_RT_FUNC static __inline
#else
#	define ANN_RT_FUNC static inline
#endif

//------------------------------
// layer kinds and activations (values are part of the generated headers)
//------------------------------
enum {
	ANN_RT_DENSE,			// y = act(W x + b)
	ANN_RT_CONV2D,			// 2-D convolution, (C, H, W) layout
	ANN_RT_MAXPOOL2D,		// 2-D max pooling, per channel
	ANN_RT_FLATTEN,			// no-op: values are already stored flat (also used for reshapes)
	ANN_RT_UPSAMPLE2D,		// nearest-neighbour upsampling by size, per channel
	ANN_RT_GLOBAL_AVGPOOL2D	// each channel's mean: (C, H, W) to (C, 1, 1)
};

enum {
	ANN_RT_ACT_NONE,
	ANN_RT_ACT_RELU,
	ANN_RT_ACT_LEAKY_RELU,
	ANN_RT_ACT_SIGMOID,
	ANN_RT_ACT_TANH,
	ANN_RT_ACT_SOFTSIGN,
	ANN_RT_ACT_GELU,
	ANN_RT_ACT_SOFTMAX		// output layer only; the int8 path outputs the logits
};

//------------------------------
// One layer. Dense weights are [out_c][in_c * in_h * in_w]; conv weights are
// [out_c][in_c][size][size]. Activations are stored channel by channel, row by row.
//------------------------------
typedef struct {
	int32_t kind;				// ANN_RT_DENSE, ...
	int32_t act;				// ANN_RT_ACT_NONE, ...
	int32_t in_c, in_h, in_w;	// input shape
	int32_t out_c, out_h, out_w;// output shape
	int32_t size, stride, pad;	// conv kernel or pool window (upsampling scale), stride, zero padding

	// int8 model (NULL in a float model)
	const int8_t *w_q;			// weights
	const int32_t *b_q;			// biases, scaled by input scale * weight scale
	const int32_t *mult;		// per output channel fixed-point multiplier (Q31)
	const int8_t *shift;		// per output channel right shift applied after mult
	int32_t in_zp;				// input zero point
	int32_t zp;					// zero point of the rescaled sum (the output, or the LUT input)
	int32_t act_min;			// lowest rescaled value (the output's zero point for ReLU)
	const int8_t *lut;			// 256-entry activation table indexed by value + 128, or NULL

	// float model (NULL in an int8 model)
	const float *w_f;
	const float *b_f;
} ann_rt_layer;

typedef struct {
	int32_t layer_count;
	const ann_rt_layer *layers;
	int32_t input_size;			// values per input sample
	int32_t output_size;		// values per output sample
	int32_t max_size;			// largest layer output; the workspace holds two of them
	int32_t softmax;			// 1 if the output layer is softmax

	// int8 model: input and output quantization (real = scale * (q - zero_point))
	float in_scale;
	int32_t in_zp;
	float out_scale;
	int32_t out_zp;
} ann_rt_model;

//================================================================================================
// int8 model
//================================================================================================

//------------------------------
// round(acc * mult * 2^-shift) + zp, clamped to [lo, 127]. mult is a Q31 value
// in [2^30, 2^31) and shift is 1..62, so the product fits in 64 bits. Halves
// round up. (Right shifts of negative values are arithmetic on every compiler
// this targets.)
//------------------------------
ANN_RT_FUNC int8_t ann_rt_requantize(int32_t acc, int32_t mult, int32_t shift, int32_t zp, int32_t lo)
{
	int64_t prod = (int64_t)acc * (int64_t)mult;
	int64_t v = (prod + ((int64_t)1 << (shift - 1))) >> shift;

	v += zp;
	if (v < lo)
		v = lo;
	if (v > 127)
		v = 127;
	return (int8_t)v;
}

//------------------------------
// rescale one output channel's sum and apply the activation
//------------------------------
ANN_RT_FUNC int8_t ann_rt_finish_q(const ann_rt_layer *l, int32_t acc, int ch)
{
	int8_t v = ann_rt_requantize(acc, l->mult[ch], l->shift[ch], l->zp, l->act_min);
	return l->lut ? l->lut[v + 128] : v;
}

ANN_RT_FUNC void ann_rt_dense_q(const ann_rt_layer *l, const int8_t *x, int8_t *y)
{
	int32_t n_in = l->in_c * l->in_h * l->in_w;
	int32_t o, i;

	for (o = 0; o < l->out_c; o++)
	{
		const int8_t *w = l->w_q + (int32_t)o * n_in;
		int32_t acc = l->b_q[o];

		for (i = 0; i < n_in; i++)
			acc += ((int32_t)x[i] - l->in_zp) * (int32_t)w[i];
		y[o] = ann_rt_finish_q(l, acc, o);
	}
}

ANN_RT_FUNC void ann_rt_conv_q(const ann_rt_layer *l, const int8_t *x, int8_t *y)
{
	int32_t k = l->size, s = l->stride, p = l->pad;
	int32_t f, oh, ow, c, i, j;

	for (f = 0; f < l->out_c; f++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
			{
				int32_t acc = l->b_q[f];

				for (c = 0; c < l->in_c; c++)
				{
					const int8_t *w = l->w_q + (f * l->in_c + c) * k * k;
					const int8_t *xc = x + c * l->in_h * l->in_w;

					for (i = 0; i < k; i++)
					{
						int32_t ih = oh * s - p + i;
						if (ih < 0 || ih >= l->in_h)
							continue;	// padding is real 0, which adds nothing
						for (j = 0; j < k; j++)
						{
							int32_t iw = ow * s - p + j;
							if (iw < 0 || iw >= l->in_w)
								continue;
							acc += ((int32_t)xc[ih * l->in_w + iw] - l->in_zp) * (int32_t)w[i * k + j];
						}
					}
				}
				y[(f * l->out_h + oh) * l->out_w + ow] = ann_rt_finish_q(l, acc, f);
			}
}

ANN_RT_FUNC void ann_rt_maxpool_q(const ann_rt_layer *l, const int8_t *x, int8_t *y)
{
	int32_t c, oh, ow, i, j;

	for (c = 0; c < l->out_c; c++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
			{
				const int8_t *xc = x + (c * l->in_h + oh * l->stride) * l->in_w + ow * l->stride;
				int8_t best = xc[0];

				for (i = 0; i < l->size; i++)
					for (j = 0; j < l->size; j++)
						if (xc[i * l->in_w + j] > best)
							best = xc[i * l->in_w + j];
				y[(c * l->out_h + oh) * l->out_w + ow] = best;
			}
}

//------------------------------
// nearest-neighbour upsampling: each value fills a size x size block (int8
// values keep their quantization)
//------------------------------
ANN_RT_FUNC void ann_rt_upsample_q(const ann_rt_layer *l, const int8_t *x, int8_t *y)
{
	int32_t c, oh, ow;

	for (c = 0; c < l->out_c; c++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
				y[(c * l->out_h + oh) * l->out_w + ow] = x[(c * l->in_h + oh / l->size) * l->in_w + ow / l->size];
}

//------------------------------
// global average pooling: each channel's mean, rounded half away from the
// zero point (int8 values keep their quantization)
//------------------------------
ANN_RT_FUNC void ann_rt_global_avgpool_q(const ann_rt_layer *l, const int8_t *x, int8_t *y)
{
	int32_t c, i, area = l->in_h * l->in_w;

	for (c = 0; c < l->out_c; c++)
	{
		int32_t sum = 0;
		for (i = 0; i < area; i++)
			sum += (int32_t)x[c * area + i] - l->in_zp;
		sum = sum >= 0 ? (sum + area / 2) / area : -((-sum + area / 2) / area);
		y[c] = (int8_t)(sum + l->in_zp);
	}
}

//------------------------------
// int8 inference: input_size int8 values in, output_size int8 values out.
// workspace holds 2 * max_size int8 values. Returns 0, or -1 for a bad layer.
//------------------------------
ANN_RT_FUNC int ann_rt_predict_q(const ann_rt_model *m, const int8_t *input, int8_t *output, int8_t *workspace)
{
	const int8_t *src = input;
	int32_t n;

	for (n = 0; n < m->layer_count; n++)
	{
		const ann_rt_layer *l = &m->layers[n];
		int8_t *dst = (src == workspace) ? workspace + m->max_size : workspace;

		switch (l->kind)
		{
		case ANN_RT_DENSE:		ann_rt_dense_q(l, src, dst); break;
		case ANN_RT_CONV2D:		ann_rt_conv_q(l, src, dst); break;
		case ANN_RT_MAXPOOL2D:	ann_rt_maxpool_q(l, src, dst); break;
		case ANN_RT_UPSAMPLE2D:	ann_rt_upsample_q(l, src, dst); break;
		case ANN_RT_GLOBAL_AVGPOOL2D: ann_rt_global_avgpool_q(l, src, dst); break;
		case ANN_RT_FLATTEN:	continue;
		default:				return -1;
		}
		src = dst;
	}

	for (n = 0; n < m->output_size; n++)
		output[n] = src[n];
	return 0;
}

//------------------------------
// index of the largest output (the first one on ties)
//------------------------------
ANN_RT_FUNC int ann_rt_argmax_q(const int8_t *values, int32_t count)
{
	int32_t i, best = 0;
	for (i = 1; i < count; i++)
		if (values[i] > values[best])
			best = i;
	return (int)best;
}

#ifndef ANN_RT_NO_FLOAT

//------------------------------
// round(x / scale) + zp, clamped to int8
//------------------------------
ANN_RT_FUNC int8_t ann_rt_quantize(float x, float scale, int32_t zp)
{
	float v = x / scale + (float)zp;

	if (v <= -128.0f)
		return -128;
	if (v >= 127.0f)
		return 127;
	return (int8_t)(v >= 0.0f ? (int32_t)(v + 0.5f) : -(int32_t)(0.5f - v));
}

ANN_RT_FUNC float ann_rt_dequantize(int8_t q, float scale, int32_t zp)
{
	return scale * (float)((int32_t)q - zp);
}

ANN_RT_FUNC void ann_rt_softmax_f(float *y, int32_t n)
{
	float max = y[0], sum = 0.0f;
	int32_t i;

	for (i = 1; i < n; i++)
		if (y[i] > max)
			max = y[i];
	for (i = 0; i < n; i++)
	{
		y[i] = expf(y[i] - max);
		sum += y[i];
	}
	for (i = 0; i < n; i++)
		y[i] /= sum;
}

//------------------------------
// int8 inference with float input and output: quantizes the input, runs the
// int8 model and dequantizes the output (applying softmax if the model has it).
// workspace holds 2 * max_size + input_size + output_size int8 values.
//------------------------------
ANN_RT_FUNC int ann_rt_predict_q_float(const ann_rt_model *m, const float *input, float *output, int8_t *workspace)
{
	int8_t *in_q = workspace + 2 * m->max_size;
	int8_t *out_q = in_q + m->input_size;
	int32_t i;

	for (i = 0; i < m->input_size; i++)
		in_q[i] = ann_rt_quantize(input[i], m->in_scale, m->in_zp);

	if (ann_rt_predict_q(m, in_q, out_q, workspace) != 0)
		return -1;

	for (i = 0; i < m->output_size; i++)
		output[i] = ann_rt_dequantize(out_q[i], m->out_scale, m->out_zp);
	if (m->softmax)
		ann_rt_softmax_f(output, m->output_size);
	return 0;
}

//================================================================================================
// float model
//================================================================================================

ANN_RT_FUNC float ann_rt_activate_f(int32_t act, float x)
{
	switch (act)
	{
	case ANN_RT_ACT_RELU:		return x > 0.0f ? x : 0.0f;
	case ANN_RT_ACT_LEAKY_RELU:	return x > 0.0f ? x : 0.01f * x;
	case ANN_RT_ACT_SIGMOID:	return 1.0f / (1.0f + expf(-x));
	case ANN_RT_ACT_TANH:		return tanhf(x);
	case ANN_RT_ACT_SOFTSIGN:	return x / (1.0f + fabsf(x));
	case ANN_RT_ACT_GELU:
		return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
	default:					return x;
	}
}

//------------------------------
// the linear part of a dense or conv layer: z = W x + b
//------------------------------
ANN_RT_FUNC void ann_rt_linear_f(const ann_rt_layer *l, const float *x, float *z)
{
	int32_t k = l->size, s = l->stride, p = l->pad;
	int32_t n_in = l->in_c * l->in_h * l->in_w;
	int32_t o, f, oh, ow, c, i, j;

	if (l->kind == ANN_RT_DENSE)
	{
		for (o = 0; o < l->out_c; o++)
		{
			const float *w = l->w_f + (int32_t)o * n_in;
			float acc = l->b_f[o];
			for (i = 0; i < n_in; i++)
				acc += x[i] * w[i];
			z[o] = acc;
		}
		return;
	}

	for (f = 0; f < l->out_c; f++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
			{
				float acc = l->b_f[f];
				for (c = 0; c < l->in_c; c++)
				{
					const float *w = l->w_f + (f * l->in_c + c) * k * k;
					const float *xc = x + c * l->in_h * l->in_w;
					for (i = 0; i < k; i++)
					{
						int32_t ih = oh * s - p + i;
						if (ih < 0 || ih >= l->in_h)
							continue;
						for (j = 0; j < k; j++)
						{
							int32_t iw = ow * s - p + j;
							if (iw >= 0 && iw < l->in_w)
								acc += xc[ih * l->in_w + iw] * w[i * k + j];
						}
					}
				}
				z[(f * l->out_h + oh) * l->out_w + ow] = acc;
			}
}

ANN_RT_FUNC void ann_rt_maxpool_f(const ann_rt_layer *l, const float *x, float *y)
{
	int32_t c, oh, ow, i, j;

	for (c = 0; c < l->out_c; c++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
			{
				const float *xc = x + (c * l->in_h + oh * l->stride) * l->in_w + ow * l->stride;
				float best = xc[0];
				for (i = 0; i < l->size; i++)
					for (j = 0; j < l->size; j++)
						if (xc[i * l->in_w + j] > best)
							best = xc[i * l->in_w + j];
				y[(c * l->out_h + oh) * l->out_w + ow] = best;
			}
}

ANN_RT_FUNC void ann_rt_upsample_f(const ann_rt_layer *l, const float *x, float *y)
{
	int32_t c, oh, ow;

	for (c = 0; c < l->out_c; c++)
		for (oh = 0; oh < l->out_h; oh++)
			for (ow = 0; ow < l->out_w; ow++)
				y[(c * l->out_h + oh) * l->out_w + ow] = x[(c * l->in_h + oh / l->size) * l->in_w + ow / l->size];
}

ANN_RT_FUNC void ann_rt_global_avgpool_f(const ann_rt_layer *l, const float *x, float *y)
{
	int32_t c, i, area = l->in_h * l->in_w;

	for (c = 0; c < l->out_c; c++)
	{
		float sum = 0.0f;
		for (i = 0; i < area; i++)
			sum += x[c * area + i];
		y[c] = sum / (float)area;
	}
}

//------------------------------
// float inference: workspace holds 2 * max_size floats. Returns 0, or -1 for a bad layer.
//------------------------------
ANN_RT_FUNC int ann_rt_predict_f(const ann_rt_model *m, const float *input, float *output, float *workspace)
{
	const float *src = input;
	int32_t n, i, count;

	for (n = 0; n < m->layer_count; n++)
	{
		const ann_rt_layer *l = &m->layers[n];
		float *dst = (src == workspace) ? workspace + m->max_size : workspace;

		switch (l->kind)
		{
		case ANN_RT_DENSE:
		case ANN_RT_CONV2D:
			ann_rt_linear_f(l, src, dst);
			count = l->out_c * l->out_h * l->out_w;
			if (l->act != ANN_RT_ACT_SOFTMAX)
				for (i = 0; i < count; i++)
					dst[i] = ann_rt_activate_f(l->act, dst[i]);
			break;
		case ANN_RT_MAXPOOL2D:	ann_rt_maxpool_f(l, src, dst); break;
		case ANN_RT_UPSAMPLE2D:	ann_rt_upsample_f(l, src, dst); break;
		case ANN_RT_GLOBAL_AVGPOOL2D: ann_rt_global_avgpool_f(l, src, dst); break;
		case ANN_RT_FLATTEN:	continue;
		default:				return -1;
		}
		src = dst;
	}

	for (i = 0; i < m->output_size; i++)
		output[i] = src[i];
	if (m->softmax)
		ann_rt_softmax_f(output, m->output_size);
	return 0;
}

ANN_RT_FUNC int ann_rt_argmax_f(const float *values, int32_t count)
{
	int32_t i, best = 0;
	for (i = 1; i < count; i++)
		if (values[i] > values[best])
			best = i;
	return (int)best;
}

#endif /* ANN_RT_NO_FLOAT */

#endif /* ANN_RT_H */
