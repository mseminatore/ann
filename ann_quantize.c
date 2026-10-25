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
// int8 quantization and C header export
//
// ann_quantize() turns a trained network into an int8 model:
//   weights      symmetric int8, one scale per output channel: s_w = max|W_row| / 127
//   biases       int32 at scale s_in * s_w
//   activations  asymmetric int8, one scale and zero point per tensor, from the value ranges
//                seen on a calibration set (each range is widened to include 0, so ReLU and
//                zero padding are exact)
// Dense and conv layers sum (x - x_zp) * w in int32 and rescale the sum with a fixed-point
// multiplier per output channel. ReLU and no activation are applied by clamping; the other
// activations rescale to an int8 pre-activation and look the result up in a 256-entry table.
// Max pooling and flatten work on int8 values directly. Softmax outputs stay as int8 logits;
// the float helpers apply softmax after dequantizing.
//
// The int8 and float models run on ann_rt.h, the same runtime the exported headers use.
//================================================================================================

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#	define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "ann.h"
#include "ann_rt.h"

struct QuantizedNetwork
{
	ann_rt_model model;
	ann_rt_layer *layers;		// owns every array the layers point to
	int8_t *workspace;			// 2 * max_size + input_size + output_size
	float *in_f, *out_f;		// float copies of one sample's inputs and outputs
};

//------------------------------
// report an error through the installed callback, or to stderr
//------------------------------
static void report_error(int error_code, const char *function_name)
{
	ErrorLogCallback callback = ann_get_error_log_callback();

	if (callback)
		callback(error_code, ann_strerror(error_code), function_name);
	else
		fprintf(stderr, "[%s] Error %d: %s\n", function_name, error_code, ann_strerror(error_code));
}

static int32_t rt_activation(Activation_type act)
{
	switch (act)
	{
	case ACTIVATION_RELU:		return ANN_RT_ACT_RELU;
	case ACTIVATION_LEAKY_RELU:	return ANN_RT_ACT_LEAKY_RELU;
	case ACTIVATION_SIGMOID:	return ANN_RT_ACT_SIGMOID;
	case ACTIVATION_TANH:		return ANN_RT_ACT_TANH;
	case ACTIVATION_SOFTSIGN:	return ANN_RT_ACT_SOFTSIGN;
	case ACTIVATION_GELU:		return ANN_RT_ACT_GELU;
	case ACTIVATION_SOFTMAX:	return ANN_RT_ACT_SOFTMAX;
	default:					return ANN_RT_ACT_NONE;
	}
}

//------------------------------
// activations in double precision, for building lookup tables
//------------------------------
static double activate(int32_t act, double x)
{
	switch (act)
	{
	case ANN_RT_ACT_RELU:		return x > 0.0 ? x : 0.0;
	case ANN_RT_ACT_LEAKY_RELU:	return x > 0.0 ? x : 0.01 * x;
	case ANN_RT_ACT_SIGMOID:	return 1.0 / (1.0 + exp(-x));
	case ANN_RT_ACT_TANH:		return tanh(x);
	case ANN_RT_ACT_SOFTSIGN:	return x / (1.0 + fabs(x));
	case ANN_RT_ACT_GELU:		return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
	default:					return x;
	}
}

static int is_weighted(const ann_rt_layer *l)
{
	return l->kind == ANN_RT_DENSE || l->kind == ANN_RT_CONV2D;
}

static int weight_count(const ann_rt_layer *l)
{
	if (l->kind == ANN_RT_DENSE)
		return l->out_c * l->in_c * l->in_h * l->in_w;
	if (l->kind == ANN_RT_CONV2D)
		return l->out_c * ann_rt_group_channels(l) * l->size * l->size;
	return 0;
}

//------------------------------
// free the float weights made by make_float_layers()
//------------------------------
static void free_float_layers(ann_rt_layer *layers, int count)
{
	if (!layers)
		return;
	for (int n = 0; n < count; n++)
	{
		free((void *)layers[n].w_f);
		free((void *)layers[n].b_f);
	}
	free(layers);
}

//------------------------------
// fold a batch norm layer into the dense or conv layer l before it:
// y = s * (W x + b - mean) + beta with s = gamma / sqrt(var + epsilon), so
// W' = s * W and b' = s * (b - mean) + beta per output channel. l takes the
// batch norm's activation. Returns ERR_OK, or ERR_INVALID when l is not a
// weighted layer without an activation.
//------------------------------
static int fold_batchnorm(ann_rt_layer *l, const Layer *bn)
{
	const real *gamma = bn->params[0].value->values, *beta = bn->params[1].value->values;
	const real *mean = bn->params[2].value->values, *var = bn->params[3].value->values;

	if (!is_weighted(l) || l->act != ANN_RT_ACT_NONE || l->out_c != bn->shape.c ||
		l->out_h != bn->shape.h || l->out_w != bn->shape.w)
		return ERR_INVALID;

	int per_channel = weight_count(l) / l->out_c;
	float *w = (float *)l->w_f, *b = (float *)l->b_f;
	int finite = 1;
	for (int c = 0; c < l->out_c; c++)
	{
		double s = (double)gamma[c] / sqrt((double)var[c] + ANN_BATCHNORM_EPSILON);
		for (int i = 0; i < per_channel; i++)
		{
			w[c * per_channel + i] = (float)(s * w[c * per_channel + i]);
			finite &= isfinite(w[c * per_channel + i]) != 0;
		}
		b[c] = (float)(s * ((double)b[c] - mean[c]) + beta[c]);
		finite &= isfinite(b[c]) != 0;
	}
	l->act = rt_activation(bn->activation);
	return finite ? ERR_OK : ERR_INVALID;
}

//------------------------------
// describe the network's layers 1..n-1 for ann_rt.h with float copies of the
// weights, batch norm layers folded into the layer before them; fills the
// model's sizes. Returns ERR_OK, ERR_INVALID or ERR_ALLOC.
//------------------------------
static int make_float_layers(const PNetwork pnet, ann_rt_model *model, ann_rt_layer **out)
{
	int count = pnet->layer_count - 1, n = 0;
	ann_rt_layer *layers;

	*out = NULL;
	if (count < 1)
		return ERR_INVALID;

	// softmax belongs on the output layer only
	for (int n = 1; n < count; n++)
		if (pnet->layers[n].activation == ACTIVATION_SOFTMAX)
			return ERR_INVALID;

	layers = calloc(count, sizeof(ann_rt_layer));
	if (!layers)
		return ERR_ALLOC;

	memset(model, 0, sizeof(*model));
	model->layers = layers;
	model->input_size = pnet->layers[0].node_count;
	model->output_size = pnet->layers[count].node_count;
	model->softmax = pnet->layers[count].activation == ACTIVATION_SOFTMAX;

	for (int k = 1; k <= count; k++)
	{
		const Layer *src = &pnet->layers[k];
		int size, stride, pad;

		if (src->kind == LAYER_KIND_BATCHNORM)
		{
			if (n == 0 || fold_batchnorm(&layers[n - 1], src) != ERR_OK)
			{
				free_float_layers(layers, count);
				return ERR_INVALID;
			}
			continue;
		}

		ann_rt_layer *l = &layers[n++];
		ann_get_layer_settings(pnet, k, &size, &stride, &pad);
		l->act = rt_activation(src->activation);
		l->in_c = src->in_shape.c;
		l->in_h = src->in_shape.h;
		l->in_w = src->in_shape.w;
		l->out_c = src->shape.c;
		l->out_h = src->shape.h;
		l->out_w = src->shape.w;
		l->size = size;
		l->stride = stride;
		l->pad = pad;
		l->groups = ann_get_layer_groups(pnet, k);

		switch (src->kind)
		{
		case LAYER_KIND_DENSE:		l->kind = ANN_RT_DENSE; break;
		case LAYER_KIND_CONV2D:		l->kind = ANN_RT_CONV2D; break;
		case LAYER_KIND_MAXPOOL2D:	l->kind = ANN_RT_MAXPOOL2D; break;
		case LAYER_KIND_FLATTEN:
		case LAYER_KIND_RESHAPE:	l->kind = ANN_RT_FLATTEN; break;
		case LAYER_KIND_UPSAMPLE2D:	l->kind = ANN_RT_UPSAMPLE2D; break;
		case LAYER_KIND_GLOBAL_AVGPOOL2D: l->kind = ANN_RT_GLOBAL_AVGPOOL2D; break;
		default:
			free_float_layers(layers, count);
			return ERR_INVALID;
		}

		if (src->node_count > model->max_size)
			model->max_size = src->node_count;

		if (!is_weighted(l))
			continue;

		const PTensor W = src->params[0].value, b = src->params[1].value;
		int wn = W->rows * W->cols;
		float *w_f = malloc(wn * sizeof(float));
		float *b_f = malloc(l->out_c * sizeof(float));

		l->w_f = w_f;
		l->b_f = b_f;
		if (!w_f || !b_f)
		{
			free_float_layers(layers, count);
			return ERR_ALLOC;
		}

		int finite = 1;
		for (int i = 0; i < wn; i++)
		{
			w_f[i] = (float)W->values[i];
			finite &= isfinite(w_f[i]) != 0;
		}
		for (int i = 0; i < l->out_c; i++)
		{
			b_f[i] = (float)b->values[i];
			finite &= isfinite(b_f[i]) != 0;
		}
		if (!finite)
		{
			free_float_layers(layers, count);
			return ERR_INVALID;
		}
	}

	model->layer_count = n;
	*out = layers;
	return ERR_OK;
}

//------------------------------
// value ranges seen during calibration
//------------------------------
typedef struct {
	float lo, hi;
} Range;

static void range_add(Range *r, const float *values, int count)
{
	for (int i = 0; i < count; i++)
	{
		if (values[i] < r->lo) r->lo = values[i];
		if (values[i] > r->hi) r->hi = values[i];
	}
}

//------------------------------
// int8 scale and zero point covering [lo, hi] widened to include 0
//------------------------------
static void range_params(Range r, float *scale, int32_t *zp)
{
	double lo = r.lo < 0.0f ? r.lo : 0.0, hi = r.hi > 0.0f ? r.hi : 0.0;
	double s = (hi - lo) / 255.0;

	if (!(s > 1e-30))
	{
		*scale = 1.0f;
		*zp = 0;
		return;
	}

	*scale = (float)s;
	double z = floor(-128.0 - lo / *scale + 0.5);
	*zp = (int32_t)(z < -128.0 ? -128.0 : z > 127.0 ? 127.0 : z);
}

static int8_t quantize_value(double x, float scale, int32_t zp)
{
	double q = floor(x / scale + 0.5) + zp;
	return (int8_t)(q < -128.0 ? -128.0 : q > 127.0 ? 127.0 : q);
}

//------------------------------
// split a positive real multiplier into a Q31 value and a right shift (see
// ann_rt_requantize); multipliers too small to matter become 0
//------------------------------
static void quantize_multiplier(double m, int32_t *mult, int8_t *shift)
{
	int e;
	double frac = frexp(m, &e);		// m = frac * 2^e, frac in [0.5, 1)
	long long q = (long long)floor(frac * 2147483648.0 + 0.5);

	if (!(m > 0.0))
	{
		*mult = 0;
		*shift = 1;
		return;
	}

	if (q == 2147483648LL)
	{
		q /= 2;
		e++;
	}

	int s = 31 - e;
	if (s > 62)
	{
		*mult = 0;
		*shift = 1;
	}
	else if (s < 1)
	{
		*mult = 2147483647;		// multipliers >= 2^30 do not occur in practice
		*shift = 1;
	}
	else
	{
		*mult = (int32_t)q;
		*shift = (int8_t)s;
	}
}

//------------------------------
// run the float model over the calibration rows, recording each layer's
// pre-activation (z) and output (y) ranges
//------------------------------
static int calibrate(const ann_rt_model *model, const PTensor inputs, Range *in_range, Range *z, Range *y)
{
	float *buf = malloc((2 * model->max_size + model->input_size) * sizeof(float));
	float *x = buf ? buf + 2 * model->max_size : NULL;

	if (!buf)
		return ERR_ALLOC;

	in_range->lo = in_range->hi = 0.0f;
	for (int n = 0; n < model->layer_count; n++)
	{
		z[n].lo = z[n].hi = 0.0f;
		y[n].lo = y[n].hi = 0.0f;
	}

	for (int r = 0; r < inputs->rows; r++)
	{
		const float *src = x;

		for (int i = 0; i < model->input_size; i++)
			x[i] = (float)inputs->values[(size_t)r * inputs->cols + i];
		range_add(in_range, x, model->input_size);

		for (int n = 0; n < model->layer_count; n++)
		{
			const ann_rt_layer *l = &model->layers[n];
			float *dst = (src == buf) ? buf + model->max_size : buf;
			int count = l->out_c * l->out_h * l->out_w;

			if (l->kind == ANN_RT_FLATTEN)
				continue;
			if (l->kind == ANN_RT_MAXPOOL2D)
				ann_rt_maxpool_f(l, src, dst);
			else if (l->kind == ANN_RT_UPSAMPLE2D)
				ann_rt_upsample_f(l, src, dst);
			else if (l->kind == ANN_RT_GLOBAL_AVGPOOL2D)
				ann_rt_global_avgpool_f(l, src, dst);
			else
			{
				ann_rt_linear_f(l, src, dst);
				range_add(&z[n], dst, count);
				if (l->act != ANN_RT_ACT_SOFTMAX)
					for (int i = 0; i < count; i++)
						dst[i] = ann_rt_activate_f(l->act, dst[i]);
				range_add(&y[n], dst, count);
			}
			src = dst;
		}
	}

	free(buf);
	return ERR_OK;
}

//------------------------------
// free a quantized network
//------------------------------
void ann_free_quantized(PQuantizedNetwork q)
{
	if (!q)
		return;

	if (q->layers)
	{
		for (int n = 0; n < q->model.layer_count; n++)
		{
			free((void *)q->layers[n].w_q);
			free((void *)q->layers[n].b_q);
			free((void *)q->layers[n].mult);
			free((void *)q->layers[n].shift);
			free((void *)q->layers[n].lut);
		}
		free(q->layers);
	}
	free(q->workspace);
	free(q->in_f);
	free(q->out_f);
	free(q);
}

//------------------------------
// quantize one dense or conv layer from its float form. x_scale/x_zp
// describe its input; z and y are its calibrated ranges. Sets *y_scale and
// *y_zp to its output's quantization.
//------------------------------
static int quantize_layer(ann_rt_layer *l, const ann_rt_layer *f, float x_scale, int32_t x_zp,
                          Range z, Range y, float *y_scale, int32_t *y_zp)
{
	int n_in = weight_count(f) / f->out_c;
	float t_scale;				// scale and zero point the sums are rescaled to
	int32_t t_zp;
	int8_t *w_q = malloc((size_t)f->out_c * n_in);
	int32_t *b_q = malloc(f->out_c * sizeof(int32_t));
	int32_t *mult = malloc(f->out_c * sizeof(int32_t));
	int8_t *shift = malloc(f->out_c);

	l->w_q = w_q;
	l->b_q = b_q;
	l->mult = mult;
	l->shift = shift;
	if (!w_q || !b_q || !mult || !shift)
		return ERR_ALLOC;

	l->in_zp = x_zp;
	l->act_min = -128;

	if (f->act == ANN_RT_ACT_NONE || f->act == ANN_RT_ACT_SOFTMAX)
	{
		range_params(z, &t_scale, &t_zp);
		*y_scale = t_scale;
		*y_zp = t_zp;
	}
	else if (f->act == ANN_RT_ACT_RELU)
	{
		range_params(y, &t_scale, &t_zp);
		*y_scale = t_scale;
		*y_zp = t_zp;
		l->act_min = t_zp;
	}
	else
	{
		int8_t *lut = malloc(256);
		l->lut = lut;
		if (!lut)
			return ERR_ALLOC;

		range_params(z, &t_scale, &t_zp);
		range_params(y, y_scale, y_zp);
		for (int i = 0; i < 256; i++)
		{
			double zv = (double)t_scale * (i - 128 - t_zp);
			lut[i] = quantize_value(activate(f->act, zv), *y_scale, *y_zp);
		}
	}
	l->zp = t_zp;

	for (int c = 0; c < f->out_c; c++)
	{
		const float *w = f->w_f + (size_t)c * n_in;
		double max_abs = 0.0;

		for (int i = 0; i < n_in; i++)
			if (fabs(w[i]) > max_abs)
				max_abs = fabs(w[i]);

		double w_scale = max_abs > 0.0 ? max_abs / 127.0 : 1.0;
		double acc_scale = (double)x_scale * w_scale;

		for (int i = 0; i < n_in; i++)
			w_q[(size_t)c * n_in + i] = (int8_t)floor(w[i] / w_scale + 0.5);

		double bq = floor(f->b_f[c] / acc_scale + 0.5);
		b_q[c] = (int32_t)(bq < -2147483647.0 ? -2147483647.0 : bq > 2147483647.0 ? 2147483647.0 : bq);

		quantize_multiplier(acc_scale / t_scale, &mult[c], &shift[c]);
	}
	return ERR_OK;
}

//------------------------------
// quantize a trained network to int8, calibrating activation ranges on the
// rows of calibration_inputs
//------------------------------
PQuantizedNetwork ann_quantize(const PNetwork pnet, const PTensor calibration_inputs)
{
	ann_rt_model fmodel;
	ann_rt_layer *flayers = NULL;
	Range in_range, *z = NULL, *y = NULL;
	PQuantizedNetwork q = NULL;
	float scale;
	int32_t zp;
	int count, err;

	if (!pnet || !calibration_inputs || !calibration_inputs->values)
	{
		report_error(ERR_NULL_PTR, "ann_quantize");
		return NULL;
	}

	if (pnet->layer_count < 2 || calibration_inputs->rows < 1 ||
		calibration_inputs->cols != pnet->layers[0].node_count)
	{
		report_error(ERR_INVALID, "ann_quantize");
		return NULL;
	}

	err = make_float_layers(pnet, &fmodel, &flayers);
	if (err != ERR_OK)
	{
		report_error(err, "ann_quantize");
		return NULL;
	}

	count = fmodel.layer_count;
	err = ERR_ALLOC;
	z = malloc(count * sizeof(Range));
	y = malloc(count * sizeof(Range));
	q = calloc(1, sizeof(*q));
	if (!z || !y || !q)
		goto done;

	err = calibrate(&fmodel, calibration_inputs, &in_range, z, y);
	if (err != ERR_OK)
		goto done;

	q->model = fmodel;
	q->model.layers = NULL;
	q->layers = calloc(count, sizeof(ann_rt_layer));
	q->workspace = malloc(2 * fmodel.max_size + fmodel.input_size + fmodel.output_size);
	q->in_f = malloc(fmodel.input_size * sizeof(float));
	q->out_f = malloc(fmodel.output_size * sizeof(float));
	if (!q->layers || !q->workspace || !q->in_f || !q->out_f)
		goto done;
	q->model.layers = q->layers;

	range_params(in_range, &scale, &zp);
	q->model.in_scale = scale;
	q->model.in_zp = zp;

	for (int n = 0; n < count; n++)
	{
		ann_rt_layer *l = &q->layers[n];
		const ann_rt_layer *f = &flayers[n];

		*l = *f;
		l->w_f = l->b_f = NULL;
		if (is_weighted(f))
		{
			err = quantize_layer(l, f, scale, zp, z[n], y[n], &scale, &zp);
			if (err != ERR_OK)
				goto done;
		}
		else
			l->in_zp = l->zp = zp;		// pooling, upsampling and flatten keep the input's quantization
	}
	q->model.out_scale = scale;
	q->model.out_zp = zp;
	err = ERR_OK;

done:
	free_float_layers(flayers, count);
	free(z);
	free(y);
	if (err != ERR_OK)
	{
		ann_free_quantized(q);
		report_error(err, "ann_quantize");
		return NULL;
	}
	return q;
}

//------------------------------
// int8 inference on int8 inputs
//------------------------------
int ann_quantized_predict_int8(const PQuantizedNetwork q, const signed char *inputs, signed char *outputs)
{
	if (!q || !inputs || !outputs)
	{
		report_error(ERR_NULL_PTR, "ann_quantized_predict_int8");
		return ERR_NULL_PTR;
	}

	if (ann_rt_predict_q(&q->model, (const int8_t *)inputs, (int8_t *)outputs, q->workspace) != 0)
		return ERR_INVALID;
	return ERR_OK;
}

//------------------------------
// int8 inference with real inputs and outputs
//------------------------------
int ann_quantized_predict(const PQuantizedNetwork q, const real *inputs, real *outputs)
{
	if (!q || !inputs || !outputs)
	{
		report_error(ERR_NULL_PTR, "ann_quantized_predict");
		return ERR_NULL_PTR;
	}

	for (int i = 0; i < q->model.input_size; i++)
		q->in_f[i] = (float)inputs[i];
	if (ann_rt_predict_q_float(&q->model, q->in_f, q->out_f, q->workspace) != 0)
		return ERR_INVALID;
	for (int i = 0; i < q->model.output_size; i++)
		outputs[i] = (real)q->out_f[i];
	return ERR_OK;
}

//------------------------------
// the int8 input and output quantization: real = scale * (q - zero_point)
//------------------------------
int ann_quantized_get_io_params(const PQuantizedNetwork q, real *in_scale, int *in_zero_point,
                                real *out_scale, int *out_zero_point)
{
	if (!q)
		return ERR_NULL_PTR;
	if (in_scale) *in_scale = (real)q->model.in_scale;
	if (in_zero_point) *in_zero_point = q->model.in_zp;
	if (out_scale) *out_scale = (real)q->model.out_scale;
	if (out_zero_point) *out_zero_point = q->model.out_zp;
	return ERR_OK;
}

//------------------------------
// classification accuracy of the int8 model
//------------------------------
real ann_quantized_evaluate_accuracy(const PQuantizedNetwork q, const PTensor inputs, const PTensor outputs)
{
	int correct = 0;

	if (!q || !inputs || !outputs || inputs->rows <= 0 || inputs->rows != outputs->rows ||
		inputs->cols != q->model.input_size || outputs->cols != q->model.output_size)
		return (real)-1.0;

	int classes = outputs->cols;
	real *pred = malloc(classes * sizeof(real));
	if (!pred)
		return (real)-1.0;

	for (int r = 0; r < inputs->rows; r++)
	{
		ann_quantized_predict(q, &inputs->values[(size_t)r * inputs->cols], pred);
		if (ann_class_prediction(pred, classes) ==
			ann_class_prediction(&outputs->values[(size_t)r * classes], classes))
			correct++;
	}

	free(pred);
	return (real)correct / inputs->rows;
}

//================================================================================================
// C header export
//================================================================================================

// a float as a C literal that reads back as the same float
static void write_float(FILE *f, float v)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%.9g", v);
	fputs(buf, f);
	if (!strpbrk(buf, ".e"))
		fputs(".0", f);
	fputc('f', f);
}

static void write_int8_array(FILE *f, const char *prefix, const char *name, int n, const int8_t *v, int count)
{
	fprintf(f, "static const int8_t %s_%s%d[%d] = {", prefix, name, n, count);
	for (int i = 0; i < count; i++)
		fprintf(f, "%s%d%s", i % 24 ? "" : "\n\t", v[i], i + 1 < count ? "," : "");
	fprintf(f, "\n};\n");
}

static void write_int32_array(FILE *f, const char *prefix, const char *name, int n, const int32_t *v, int count)
{
	fprintf(f, "static const int32_t %s_%s%d[%d] = {", prefix, name, n, count);
	for (int i = 0; i < count; i++)
		fprintf(f, "%s%ld%s", i % 8 ? "" : "\n\t", (long)v[i], i + 1 < count ? "," : "");
	fprintf(f, "\n};\n");
}

static void write_float_array(FILE *f, const char *prefix, const char *name, int n, const float *v, int count)
{
	fprintf(f, "static const float %s_%s%d[%d] = {", prefix, name, n, count);
	for (int i = 0; i < count; i++)
	{
		fputs(i % 8 ? "" : "\n\t", f);
		write_float(f, v[i]);
		if (i + 1 < count)
			fputc(',', f);
	}
	fprintf(f, "\n};\n");
}

static const char *kind_name(int32_t kind)
{
	static const char *names[] = { "ANN_RT_DENSE", "ANN_RT_CONV2D", "ANN_RT_MAXPOOL2D", "ANN_RT_FLATTEN",
		"ANN_RT_UPSAMPLE2D", "ANN_RT_GLOBAL_AVGPOOL2D" };
	return names[kind];
}

static const char *act_name(int32_t act)
{
	static const char *names[] = { "ANN_RT_ACT_NONE", "ANN_RT_ACT_RELU", "ANN_RT_ACT_LEAKY_RELU",
		"ANN_RT_ACT_SIGMOID", "ANN_RT_ACT_TANH", "ANN_RT_ACT_SOFTSIGN", "ANN_RT_ACT_GELU", "ANN_RT_ACT_SOFTMAX" };
	return names[act];
}

//------------------------------
// write the header for an int8 model (quantized) or a float model
//------------------------------
static int write_header(FILE *f, const ann_rt_model *m, int quantized, const char *prefix, const char *upper)
{
	long param_bytes = 0, float_bytes = 0;
	int ws = quantized ? 2 * m->max_size + m->input_size + m->output_size : 2 * m->max_size;

	for (int n = 0; n < m->layer_count; n++)
	{
		const ann_rt_layer *l = &m->layers[n];
		if (!is_weighted(l))
			continue;
		param_bytes += quantized ? weight_count(l) + l->out_c * 9L + (l->lut ? 256 : 0)
		                         : (weight_count(l) + l->out_c) * 4L;
		float_bytes += (weight_count(l) + l->out_c) * 4L;
	}

	fprintf(f, "/* %s.h: generated by libann ann_export_c_header(); do not edit.\n", prefix);
	fprintf(f, " *\n * %s model, %d inputs, %d outputs. Parameters: %ld bytes", quantized ? "int8" : "float32",
	        (int)m->input_size, (int)m->output_size, param_bytes);
	if (quantized)
		fprintf(f, " (%ld as float32)", float_bytes);
	fprintf(f, ".\n * Needs ann_rt.h from libann on the include path. See docs/C_EXPORT.md.\n *\n");
	fprintf(f, " *   static %s ws[%s_WORKSPACE_SIZE];\n", quantized ? "int8_t" : "float", upper);
	if (quantized)
	{
		fprintf(f, " *   %s_predict(input_q, output_q, ws);         int8 in and out\n", prefix);
		fprintf(f, " *   %s_predict_float(input, output, ws);       float in and out (not with ANN_RT_NO_FLOAT)\n", prefix);
	}
	else
		fprintf(f, " *   %s_predict(input, output, ws);\n", prefix);
	fprintf(f, " *   class = %s_argmax(output);\n */\n\n", prefix);

	fprintf(f, "#ifndef %s_H\n#define %s_H\n\n#include \"ann_rt.h\"\n\n", upper, upper);
	// the header initializes every ann_rt_layer field (so it builds without
	// missing-initializer warnings), so it needs the runtime that has them all
	fprintf(f, "#if !defined(ANN_RT_VERSION) || ANN_RT_VERSION < %d\n#error \"%s.h needs ann_rt.h version %d or later\"\n#endif\n\n",
	        ANN_RT_VERSION, prefix, ANN_RT_VERSION);
	fprintf(f, "#define %s_INPUT_SIZE %d\n", upper, (int)m->input_size);
	fprintf(f, "#define %s_OUTPUT_SIZE %d\n", upper, (int)m->output_size);
	fprintf(f, "#define %s_WORKSPACE_SIZE %d\n", upper, ws);
	if (quantized)
	{
		fprintf(f, "#define %s_INPUT_SCALE ", upper);
		write_float(f, m->in_scale);
		fprintf(f, "\n#define %s_INPUT_ZERO_POINT %d\n", upper, (int)m->in_zp);
		fprintf(f, "#define %s_OUTPUT_SCALE ", upper);
		write_float(f, m->out_scale);
		fprintf(f, "\n#define %s_OUTPUT_ZERO_POINT %d\n", upper, (int)m->out_zp);
	}
	fputc('\n', f);

	for (int n = 0; n < m->layer_count; n++)
	{
		const ann_rt_layer *l = &m->layers[n];
		if (!is_weighted(l))
			continue;
		if (quantized)
		{
			write_int8_array(f, prefix, "w", n, l->w_q, weight_count(l));
			write_int32_array(f, prefix, "b", n, l->b_q, l->out_c);
			write_int32_array(f, prefix, "mult", n, l->mult, l->out_c);
			write_int8_array(f, prefix, "shift", n, l->shift, l->out_c);
			if (l->lut)
				write_int8_array(f, prefix, "lut", n, l->lut, 256);
		}
		else
		{
			write_float_array(f, prefix, "w", n, l->w_f, weight_count(l));
			write_float_array(f, prefix, "b", n, l->b_f, l->out_c);
		}
		fputc('\n', f);
	}

	fprintf(f, "static const ann_rt_layer %s_layers[%d] = {\n", prefix, (int)m->layer_count);
	for (int n = 0; n < m->layer_count; n++)
	{
		const ann_rt_layer *l = &m->layers[n];
		int w = is_weighted(l);

		fprintf(f, "\t{ %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d,\n", kind_name(l->kind), act_name(l->act),
		        (int)l->in_c, (int)l->in_h, (int)l->in_w, (int)l->out_c, (int)l->out_h, (int)l->out_w,
		        (int)l->size, (int)l->stride, (int)l->pad);
		if (quantized && w)
		{
			fprintf(f, "\t  %s_w%d, %s_b%d, %s_mult%d, %s_shift%d, %d, %d, %d, ", prefix, n, prefix, n, prefix, n,
			        prefix, n, (int)l->in_zp, (int)l->zp, (int)l->act_min);
			if (l->lut)
				fprintf(f, "%s_lut%d,\n", prefix, n);
			else
				fprintf(f, "NULL,\n");
		}
		else
			fprintf(f, "\t  NULL, NULL, NULL, NULL, %d, %d, %d, NULL,\n", (int)l->in_zp, (int)l->zp, (int)l->act_min);
		if (!quantized && w)
			fprintf(f, "\t  %s_w%d, %s_b%d, %d },\n", prefix, n, prefix, n, (int)l->groups);
		else
			fprintf(f, "\t  NULL, NULL, %d },\n", (int)l->groups);
	}
	fprintf(f, "};\n\n");

	fprintf(f, "static const ann_rt_model %s_model = {\n", prefix);
	fprintf(f, "\t%d, %s_layers, %d, %d, %d, %d,\n\t", (int)m->layer_count, prefix, (int)m->input_size,
	        (int)m->output_size, (int)m->max_size, (int)m->softmax);
	write_float(f, m->in_scale);
	fprintf(f, ", %d, ", (int)m->in_zp);
	write_float(f, m->out_scale);
	fprintf(f, ", %d\n};\n\n", (int)m->out_zp);

	if (quantized)
	{
		fprintf(f, "ANN_RT_FUNC int %s_predict(const int8_t *input, int8_t *output, int8_t *workspace)\n"
		           "{\n\treturn ann_rt_predict_q(&%s_model, input, output, workspace);\n}\n\n", prefix, prefix);
		fprintf(f, "#ifndef ANN_RT_NO_FLOAT\n"
		           "ANN_RT_FUNC int %s_predict_float(const float *input, float *output, int8_t *workspace)\n"
		           "{\n\treturn ann_rt_predict_q_float(&%s_model, input, output, workspace);\n}\n#endif\n\n",
		        prefix, prefix);
		fprintf(f, "ANN_RT_FUNC int %s_argmax(const int8_t *output)\n"
		           "{\n\treturn ann_rt_argmax_q(output, %s_OUTPUT_SIZE);\n}\n\n", prefix, upper);
	}
	else
	{
		fprintf(f, "ANN_RT_FUNC int %s_predict(const float *input, float *output, float *workspace)\n"
		           "{\n\treturn ann_rt_predict_f(&%s_model, input, output, workspace);\n}\n\n", prefix, prefix);
		fprintf(f, "ANN_RT_FUNC int %s_argmax(const float *output)\n"
		           "{\n\treturn ann_rt_argmax_f(output, %s_OUTPUT_SIZE);\n}\n\n", prefix, upper);
	}

	fprintf(f, "#endif /* %s_H */\n", upper);
	return ferror(f) ? ERR_IO : ERR_OK;
}

//------------------------------
// export a model as a C header: the int8 model q, or the float network pnet
// when q is NULL
//------------------------------
int ann_export_c_header(const PNetwork pnet, const PQuantizedNetwork q, const char *filename, const char *prefix)
{
	char upper[64];
	ann_rt_model fmodel;
	ann_rt_layer *flayers = NULL;
	int err;

	if ((!pnet && !q) || !filename || !prefix)
	{
		report_error(ERR_NULL_PTR, "ann_export_c_header");
		return ERR_NULL_PTR;
	}

	// prefix must be a C identifier
	size_t len = strlen(prefix);
	int ok = len > 0 && len < sizeof(upper) && (isalpha((unsigned char)prefix[0]) || prefix[0] == '_');
	for (size_t i = 0; ok && i < len; i++)
	{
		if (!isalnum((unsigned char)prefix[i]) && prefix[i] != '_')
			ok = 0;
		upper[i] = (char)toupper((unsigned char)prefix[i]);
	}
	if (!ok)
	{
		report_error(ERR_INVALID, "ann_export_c_header");
		return ERR_INVALID;
	}
	upper[len] = '\0';

	if (!q)
	{
		err = make_float_layers(pnet, &fmodel, &flayers);
		if (err != ERR_OK)
		{
			report_error(err, "ann_export_c_header");
			return err;
		}
	}

	FILE *f = fopen(filename, "w");
	if (!f)
	{
		if (flayers)
			free_float_layers(flayers, fmodel.layer_count);
		report_error(ERR_IO, "ann_export_c_header");
		return ERR_IO;
	}

	err = write_header(f, q ? &q->model : &fmodel, q != NULL, prefix, upper);
	if (fclose(f) != 0)
		err = ERR_IO;
	if (flayers)
		free_float_layers(flayers, fmodel.layer_count);
	if (err != ERR_OK)
		report_error(err, "ann_export_c_header");
	return err;
}
