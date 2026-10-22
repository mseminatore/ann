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

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#	define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include "ann.h"
#include "json.h"
#include "ann_parallel.h"

// Element-wise loops run on several threads once they have at least twice this
// many elements (see ann_parallel.h). Each element is computed independently,
// so results do not depend on the thread count.
#define ANN_ELEMENT_GRAIN 16384

#include "ann_gpu_backend.h"

// Active GPU backend. NULL when no GPU is initialized.
// Set by ann_gpu_init(); read by dispatch logic throughout this file.
GpuBackend *g_gpu_backend = NULL;

//-----------------------------------------------------------
// Public GPU API — delegate to the active backend vtable.
// Backends register themselves by providing a GpuBackend instance.
//-----------------------------------------------------------

ANN_API int ann_gpu_init(void)
{
#ifdef USE_METAL
    extern GpuBackend metal_backend;
    if (metal_backend.init()) { g_gpu_backend = &metal_backend; printf("[ann] Using Metal GPU backend\n"); return 1; }
#endif
#ifdef USE_CUDA
    extern GpuBackend cuda_backend;
    if (cuda_backend.init()) { g_gpu_backend = &cuda_backend; printf("[ann] Using CUDA GPU backend\n"); return 1; }
#endif
    printf("[ann] Using CPU backend\n");
    return 0;
}

static void invoke_error_callback(int error_code, const char *function_name);

ANN_API int ann_gpu_upload_network(PNetwork pnet)
{
    if (!g_gpu_backend)
        return ERR_FAIL;

    if (!pnet)
    {
        invoke_error_callback(ERR_NULL_PTR, "ann_gpu_upload_network");
        return ERR_NULL_PTR;
    }

    // The backend returns ERR_INVALID for layer kinds it does not implement;
    // those networks stay on the CPU
    int err = g_gpu_backend->upload_network(pnet);
    if (err == ERR_OK)
        pnet->gpu_stale = 0;
    else
        invoke_error_callback(err, "ann_gpu_upload_network");
    return err;
}

ANN_API void ann_gpu_free_network(PNetwork pnet)
{
    if (g_gpu_backend) g_gpu_backend->free_network(pnet);
    if (pnet) pnet->gpu_stale = 0;
}

ANN_API void ann_gpu_sync_weights(PNetwork pnet)
{
    // A stale GPU copy is older than the CPU weights; never copy it back
    if (g_gpu_backend && pnet && !pnet->gpu_stale)
        g_gpu_backend->sync_weights(pnet);
}

#ifdef USE_GPU
// Weights are on the GPU (dispatch checks the first parameter of the first
// layer that has any, which every backend uploads)
static int gpu_uploaded(const PNetwork pnet)
{
    if (!g_gpu_backend)
        return 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        if (pnet->layers[layer].param_count > 0)
            return pnet->layers[layer].params[0].value->gpu_buf != NULL;
    }
    return 0;
}

// The GPU copy may be used for inference: uploaded and as new as the CPU weights
static int gpu_ready(const PNetwork pnet)
{
    return gpu_uploaded(pnet) && !pnet->gpu_stale;
}

// Re-upload after CPU-side weight changes so the GPU copy is current again
static void gpu_refresh(PNetwork pnet)
{
    if (gpu_uploaded(pnet))
        pnet->gpu_stale = (g_gpu_backend->upload_network(pnet) != ERR_OK);
}

#endif

//================================================================================================
// NEURAL NETWORK IMPLEMENTATION - Architecture and Design Overview
//================================================================================================
// This file implements a feedforward neural network (multilayer perceptron) with the following:
//
// NETWORK STRUCTURE:
//   - Flexible layer architecture: input layer → hidden layers → output layer
//   - Each layer contains: node_count, weights matrix, bias vector, activation function
//   - Forward propagation: a[i] = activation(W[i] * a[i-1] + b[i])
//   - Gradients stored in each layer for backpropagation
//
// KEY ALGORITHMS:
//   1. Forward Propagation (eval_network):
//      Input → Conv to Matrix → Layer1 compute → Layer2 compute → ... → Output
//
//   2. Backpropagation (back_propagate_batched):
//      Computes gradients layer-by-layer from output back to input
//      Using chain rule: ∂L/∂W = (∂L/∂a) * (∂a/∂z) * (∂z/∂W)
//      Where z = W*a + b, a = activation(z)
//
//   3. Optimization:
//      Updates weights based on gradients using SGD, momentum, adaptive rates, etc.
//      W_new = W_old - learning_rate * ∂L/∂W
//
// LOSS FUNCTIONS:
//   - MSE (Mean Squared Error): For regression tasks
//   - Cross-Entropy: For classification tasks
//   - Computed as: L = (1/batch_size) * sum of per-sample losses
//
// SUPPORTED ACTIVATIONS:
//   - Sigmoid: f(x) = 1/(1+e^(-x)), range (0,1)
//   - ReLU: f(x) = max(0,x), efficient and addresses vanishing gradient
//   - Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x)), range (-1,1), zero-centered
//   - Softmax: Multi-class probability distribution, range (0,1) summing to 1
//   - Leaky ReLU, Softsign: Specialized variants
//
// TRAINING LOOP (ann_train_network):
//   For each epoch:
//     Shuffle training data indices
//     For each mini-batch:
//       1. Load sample(s)
//       2. Forward propagation → compute loss
//       3. Backpropagation → compute gradients
//       4. Optimization step → update weights
//       5. Check convergence criteria
//
// ERROR HANDLING:
//   - NULL pointer validation for all public functions (ERR_NULL_PTR)
//   - Memory allocation failures with automatic rollback (ERR_ALLOC)
//   - Invalid dimensions/parameters (ERR_INVALID)
//   - File I/O errors (ERR_IO)
//
// MEMORY MANAGEMENT:
//   - All tensors allocated with aligned memory for BLAS performance
//   - Gradients allocated on-demand during training
//   - Automatic cleanup: ann_free_network releases all layer tensors
//
// PERSISTENCE:
//   - Text format: Human-readable network parameters
//   - Binary format: Compact storage with checksums
//   - Both formats support version control for format evolution
//
//================================================================================================

//================================================================================================================
// GLOBAL ERROR LOGGING CALLBACK
//================================================================================================================

/**
 * Global callback function pointer for error logging.
 * Can be set via ann_set_error_log_callback() to enable error notifications.
 * NULL when no callback is installed (default).
 */
static ErrorLogCallback g_error_log_callback = NULL;

//------------------------------------------------
// define the text and binary file format versions
//------------------------------------------------
// ONNX producer identity; version 1.0 files hold untransposed weights
#define ONNX_PRODUCER_NAME		"ann-library"
#define ONNX_PRODUCER_VERSION	"1.1"

// model file format versions (see save_network)
static const int ANN_FORMAT_V1 = 1;
static const int ANN_FORMAT_V2 = 2;

#if defined(_WIN32) && !defined(_WIN64)
#	define R_MIN -1
#	define R_MAX 1
#else
#	define R_MIN -1.0
#	define R_MAX 1.0
#endif

//-----------------------------------------------
// optimizer printable names
//-----------------------------------------------
static const char *optimizers[] = {
	"Stochastic Gradient Descent",
	"SGD + Momentum",
	"RMSProp",
	"Adagrad",
	"Adam",
	"Adam"	// default
};

//-----------------------------------------------
// loss function printable anmes
//-----------------------------------------------
static const char *loss_types[] = {
	"Mean Squared Error",
	"Categorical Cross-Entropy",
	"Mean Squared Error"
};

//-----------------------------------------------
// ANSI color codes for terminal output
// Disabled if ANN_NO_COLOR environment variable is set
//-----------------------------------------------
static int g_colors_enabled = -1;  // -1 = not checked yet

static int colors_enabled(void)
{
	if (g_colors_enabled == -1)
	{
		const char *no_color = getenv("ANN_NO_COLOR");
		g_colors_enabled = (no_color == NULL || no_color[0] == '\0') ? 1 : 0;
	}
	return g_colors_enabled;
}

// ANSI color codes
#define ANSI_RESET      "\033[0m"
#define ANSI_BOLD       "\033[1m"
#define ANSI_DIM        "\033[2m"
#define ANSI_RED        "\033[31m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_YELLOW     "\033[33m"
#define ANSI_BLUE       "\033[34m"
#define ANSI_MAGENTA    "\033[35m"
#define ANSI_CYAN       "\033[36m"
#define ANSI_WHITE      "\033[37m"
#define ANSI_BOLD_GREEN "\033[1;32m"
#define ANSI_BOLD_CYAN  "\033[1;36m"
#define ANSI_BOLD_WHITE "\033[1;37m"
#define ANSI_BOLD_RED   "\033[1;31m"

// Helper macros for conditional color output
#define COLOR(code) (colors_enabled() ? (code) : "")
#define RESET()     (colors_enabled() ? ANSI_RESET : "")

//-----------------------------------------------
// default lib output function
//-----------------------------------------------
static void ann_puts(const char *s)
{
	fputs(s, stdout);
}

//-----------------------------------------------
// formatted output
//-----------------------------------------------
static void ann_printf(PNetwork pnet, const char *format, ...)
{
	char buf[DEFAULT_SMALL_BUF_SIZE];
	va_list valist;

	// a network with no print function is silent
	if (!pnet->print_func)
		return;

	va_start(valist, format);
		vsnprintf(buf, sizeof(buf), format, valist);
	va_end(valist);

	pnet->print_func(buf);
}

//------------------------------
// null activation
//------------------------------
static real no_activation(real x) { return x; }

//------------------------------
// compute the sigmoid activation
// sigmoid(x) = 1 / (1 + e^(-x))
// Range: (0, 1), S-shaped curve
// Good for: binary classification, helps vanishing gradient issues less than tanh
//------------------------------
static real sigmoid(real x)
{
	return (real)(1.0 / (1.0 + exp(-x)));
}

//------------------------------
// compute the ReLU activation
// relu(x) = max(0, x)
// Ranges: [0, inf), linear in positive domain
// Advantages: Computationally efficient, mitigates vanishing gradient
// Issues: Dead neurons (neurons that always output 0)
//------------------------------
static real relu(real x)
{
	return (real)fmax(0.0, x);
}

//------------------------------
// compute the leaky ReLU activation
// leaky_relu(x) = max(0.01*x, x)
// Solves the "dead neuron" problem of ReLU by allowing small negative slopes
// 0.01 is the slope coefficient for negative inputs
//------------------------------
static real leaky_relu(real x)
{
	return (real)fmax(0.01 * x, x);
}

//------------------------------
// compute the tanh activation
// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
// Range: (-1, 1), zero-centered which aids learning
// Good for: hidden layers, less prone to vanishing gradient than sigmoid
//------------------------------
static real ann_tanh(real x)
{
	return (real)tanh(x);
}

//---------------------------------
// compute the softsign activation
// softsign(x) = x / (1 + |x|)
// Similar to tanh but with polynomial decay instead of exponential
// Smoother derivative than tanh in high magnitude regions
//---------------------------------
static real softsign(real x)
{
	return (real)(x / (1.0 + fabs(x)));
}

//---------------------------------
// compute the GELU activation (tanh approximation, as in GPT/BERT)
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The approximation is used everywhere (CPU, GPU, ONNX) so all paths agree.
//---------------------------------
#define GELU_C ((real)0.7978845608)		// sqrt(2/pi)
#define GELU_K ((real)0.044715)

static real gelu(real x)
{
	return (real)0.5 * x * ((real)1.0 + (real)tanh(GELU_C * (x + GELU_K * x * x * x)));
}

//---------------------------------
// derivative of gelu() with respect to its input z
//---------------------------------
static real gelu_derivative(real z)
{
	real t = (real)tanh(GELU_C * (z + GELU_K * z * z * z));
	return (real)0.5 * ((real)1.0 + t) +
		(real)0.5 * z * ((real)1.0 - t * t) * GELU_C * ((real)1.0 + (real)3.0 * GELU_K * z * z);
}

//------------------------------
// Forward declarations
//------------------------------
static void record_history(PNetwork pnet, real loss, real learning_rate, real val_loss);

//------------------------------
// compute the softmax
// Converts output layer into probability distribution (sums to 1)
// softmax(x_i) = e^x_i / sum(e^x_j for all j)
// Use with LOSS_CATEGORICAL_CROSS_ENTROPY for multi-class classification
//------------------------------
static void softmax(PNetwork pnet)
{
	real sum = 0.0;

	// find the sum of the output node values, excluding the bias node
	int output_layer = pnet->layer_count - 1;
	int node_count = pnet->layers[output_layer].node_count;
	real *values = pnet->layers[output_layer].t_values->values;

	// Find max for numerical stability
	real max_val = values[0];
	for (int node = 1; node < node_count; node++)
	{
		if (values[node] > max_val)
			max_val = values[node];
	}

	for (int node = 0; node < node_count; node++)
	{
		sum += (real)exp(values[node] - max_val);
	}

	for (int node = 0; node < node_count; node++)
	{
		values[node] = (real)(exp(values[node] - max_val) / sum);
	}
}

//================================================================================================
// Layer kinds
//
// Each Layer_kind supplies the operations below. Generic code (training loop, optimizers,
// regularization, early stopping, freeing) works through them and through each layer's
// params[] list, so a new kind only implements its own math.
//================================================================================================

//-------------------------------------------
// Accumulate column sums from a [batch × cols] matrix into a [1 × cols] vector.
// bias_grad[n] += sum over batch rows of src[b * cols + n]
//-------------------------------------------
static void accumulate_bias_gradient(PTensor bias_grad, const PTensor src, int batch_size, int cols)
{
	for (int b = 0; b < batch_size; b++)
	{
		const real *row = src->values + b * cols;
		real *bg = bias_grad->values;
		int n = 0;
		int unroll_limit = cols - 3;
		for (; n < unroll_limit; n += 4)
		{
			bg[n]     += row[n];
			bg[n + 1] += row[n + 1];
			bg[n + 2] += row[n + 2];
			bg[n + 3] += row[n + 3];
		}
		for (; n < cols; n++)
			bg[n] += row[n];
	}
}

//------------------------------
// free a layer's parameters, gradients and optimizer state
//------------------------------
static void free_layer_params(PLayer layer)
{
	for (int i = 0; i < layer->param_count; i++)
	{
		tensor_free(layer->params[i].value);
		tensor_free(layer->params[i].grad);
		tensor_free(layer->params[i].m);
		tensor_free(layer->params[i].v);
	}
	memset(layer->params, 0, sizeof(layer->params));
	layer->param_count = 0;

	for (int i = 0; i < 2; i++)
	{
		tensor_free(layer->t_scratch[i]);
		layer->t_scratch[i] = NULL;
	}

	// the dense aliases pointed at the tensors just freed
	layer->t_weights	= NULL;
	layer->t_gradients	= NULL;
	layer->t_m			= NULL;
	layer->t_v			= NULL;
	layer->t_bias		= NULL;
	layer->t_bias_grad	= NULL;
	layer->t_bias_m		= NULL;
	layer->t_bias_v		= NULL;
}

//------------------------------
// add a zeroed [rows x cols] parameter with its gradient and optimizer
// state; on failure the caller frees the layer's params
//------------------------------
static int add_param(PLayer layer, int rows, int cols, int regularize)
{
	if (layer->param_count >= ANN_LAYER_MAX_PARAMS)
		return ERR_INVALID;

	Layer_param *p = &layer->params[layer->param_count++];
	p->value		= tensor_zeros(rows, cols);
	p->grad			= tensor_zeros(rows, cols);
	p->m			= tensor_zeros(rows, cols);
	p->v			= tensor_zeros(rows, cols);
	p->regularize	= regularize;

	return (p->value && p->grad && p->m && p->v) ? ERR_OK : ERR_ALLOC;
}

//------------------------------
// Dense: allocate weights [nodes x in_nodes] and bias [1 x nodes]
//------------------------------
static int dense_setup(PNetwork pnet, int layer)
{
	PLayer cur = &pnet->layers[layer];
	int in_nodes = pnet->layers[layer - 1].node_count;

	if (add_param(cur, cur->node_count, in_nodes, 1) != ERR_OK ||
		add_param(cur, 1, cur->node_count, 0) != ERR_OK)
	{
		free_layer_params(cur);
		return ERR_ALLOC;
	}

	cur->t_weights		= cur->params[0].value;
	cur->t_gradients	= cur->params[0].grad;
	cur->t_m			= cur->params[0].m;
	cur->t_v			= cur->params[0].v;
	cur->t_bias			= cur->params[1].value;
	cur->t_bias_grad	= cur->params[1].grad;
	cur->t_bias_m		= cur->params[1].m;
	cur->t_bias_v		= cur->params[1].v;
	return ERR_OK;
}

//------------------------------
// initialize a weight tensor and its bias with the chosen scheme
//------------------------------
static void init_weight_bias(PNetwork pnet, PTensor weights, PTensor bias, int fan_in, int fan_out,
                             Weight_init_type init_type)
{
	switch (init_type)
	{
	case WEIGHT_INIT_HE:
		// He initialization: std = sqrt(2 / fan_in)
		{
			real std = (real)sqrt(2.0 / fan_in);
			tensor_random_normal(weights, (real)0.0, std);
			tensor_fill(bias, (real)0.0);
		}
		break;

	case WEIGHT_INIT_XAVIER:
		// Xavier/Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
		{
			real std = (real)sqrt(2.0 / (fan_in + fan_out));
			tensor_random_normal(weights, (real)0.0, std);
			tensor_fill(bias, (real)0.0);
		}
		break;

	default:
	case WEIGHT_INIT_UNIFORM:
		// Original uniform initialization
		tensor_random_uniform(bias, (real)-pnet->weight_limit, (real)pnet->weight_limit);
		tensor_random_uniform(weights, (real)-pnet->weight_limit, (real)pnet->weight_limit);
		break;
	}
}

//------------------------------
// Dense: initialize weights and bias with the chosen scheme
//------------------------------
static void dense_init_weights(PNetwork pnet, int layer, Weight_init_type init_type)
{
	PLayer cur = &pnet->layers[layer];
	init_weight_bias(pnet, cur->t_weights, cur->t_bias, pnet->layers[layer - 1].node_count,
	                 cur->node_count, init_type);
}

//------------------------------
// Dense forward, linear part: Y = X * W^T + b for X [rows x in_nodes]
//------------------------------
static void dense_forward(PLayer layer, const PTensor X, PTensor Y, int rows)
{
	int out_nodes = layer->node_count;

	// Y = X * W^T (gemm with B transposed)
	// X is [batch × in], W is [out × in], Y is [batch × out]
	tensor_gemm_transB((real)1.0, X, layer->t_weights, (real)0.0, Y);

	// Add bias to each row
	for (int b = 0; b < rows; b++)
	{
		int row_offset = b * out_nodes;
		for (int n = 0; n < out_nodes; n++)
		{
			Y->values[row_offset + n] += layer->t_bias->values[n];
		}
	}
}

//------------------------------
// Dense forward for one sample: y = W x + b
//------------------------------
static void dense_forward_single(PLayer layer, const PTensor x, PTensor y)
{
	tensor_matvec(Tensor_NoTranspose, (real)1.0, layer->t_weights, (real)0.0, x, y);
	tensor_add(y, layer->t_bias);
}

//------------------------------
// Dense backward: given dZ = -dL/dz [rows x nodes] and the layer's input X,
// accumulate dW += dZ^T X and db += column sums of dZ, and, when dX is not
// NULL, propagate dX = dZ W to the layer below
//------------------------------
static void dense_backward(PLayer layer, const PTensor X, const PTensor dZ, PTensor dX, int rows)
{
	// dZ is [batch × out], X is [batch × in]; dZ^T X is [out × in]
	tensor_gemm_transA((real)1.0, dZ, X, (real)1.0, layer->t_gradients);

	accumulate_bias_gradient(layer->t_bias_grad, dZ, rows, layer->node_count);

	// dX = dZ * W: [batch × out] * [out × in] = [batch × in]
	if (dX)
		tensor_gemm((real)1.0, dZ, layer->t_weights, (real)0.0, dX);
}

//------------------------------
// a Tensor header over existing memory (for GEMM on one sample's rows);
// it owns nothing and is never freed
//------------------------------
static Tensor tensor_view(real *values, int rows, int cols)
{
	Tensor t;
	memset(&t, 0, sizeof(t));
	t.rows = rows;
	t.cols = cols;
	t.stride = cols;
	t.values = values;
	return t;
}

//------------------------------
// Conv2D settings, stored in config[]
//------------------------------
enum { CONV_KERNEL, CONV_STRIDE, CONV_PADDING, CONV_CONFIG_COUNT };
enum { POOL_SIZE, POOL_STRIDE, POOL_CONFIG_COUNT };

//------------------------------
// output size of a window sliding over n inputs; 0 if it does not fit
//------------------------------
static int window_output_size(int n, int window, int stride, int padding)
{
	int span = n + 2 * padding - window;
	if (window < 1 || stride < 1 || padding < 0 || span < 0)
		return 0;
	return span / stride + 1;
}

//------------------------------
// make sure *t has at least rows rows of cols values (contents are not kept)
//------------------------------
static int ensure_rows(PTensor *t, int rows, int cols)
{
	if (*t && (*t)->rows >= rows && (*t)->cols == cols)
		return ERR_OK;

	tensor_free(*t);
	*t = tensor_zeros(rows, cols);
	return *t ? ERR_OK : ERR_ALLOC;
}

//------------------------------
// Conv2D: check settings and shape, allocate W [F x C*k*k], b [1 x F]
// and one sample's column gradient [C*k*k x OH*OW]. The forward pass keeps
// each sample's im2col columns in t_scratch[0] (one row per sample) for the
// backward pass; it grows to the batch size on first use.
//------------------------------
static int conv_setup(PNetwork pnet, int layer)
{
	PLayer cur = &pnet->layers[layer];
	Layer_shape in = cur->in_shape;
	int k = cur->config[CONV_KERNEL], s = cur->config[CONV_STRIDE], p = cur->config[CONV_PADDING];

	if (cur->activation == ACTIVATION_SOFTMAX ||
		window_output_size(in.h, k, s, p) != cur->shape.h ||
		window_output_size(in.w, k, s, p) != cur->shape.w || cur->shape.h < 1 || cur->shape.w < 1)
		return ERR_INVALID;

	int patch = in.c * k * k;
	int positions = cur->shape.h * cur->shape.w;

	if (add_param(cur, cur->shape.c, patch, 1) != ERR_OK ||
		add_param(cur, 1, cur->shape.c, 0) != ERR_OK ||
		!(cur->t_scratch[0] = tensor_zeros(1, patch * positions)) ||
		!(cur->t_scratch[1] = tensor_zeros(patch, positions)))
	{
		free_layer_params(cur);
		return ERR_ALLOC;
	}
	return ERR_OK;
}

static void conv_init_weights(PNetwork pnet, int layer, Weight_init_type init_type)
{
	PLayer cur = &pnet->layers[layer];
	int kk = cur->config[CONV_KERNEL] * cur->config[CONV_KERNEL];
	init_weight_bias(pnet, cur->params[0].value, cur->params[1].value,
	                 cur->in_shape.c * kk, cur->shape.c * kk, init_type);
}

//------------------------------
// the outputs ow in [*lo, *hi) whose input column ow*s - p + j lies inside
// [0, width); the others read padding
//------------------------------
static void conv_valid_range(int width, int out_width, int s, int p, int j, int *lo, int *hi)
{
	// ow*s - p + j >= 0  <=>  ow >= ceil((p - j) / s)
	int first = p - j > 0 ? (p - j + s - 1) / s : 0;
	// ow*s - p + j <= width - 1  <=>  ow <= floor((width - 1 + p - j) / s)
	int last = width - 1 + p - j >= 0 ? (width - 1 + p - j) / s : -1;

	*lo = first < out_width ? first : out_width;
	*hi = last + 1 < out_width ? last + 1 : out_width;
	if (*hi < *lo)
		*hi = *lo;
}

//------------------------------
// copy one sample's input patches into columns: col[(c*k + i)*k + j][oh*OW + ow]
// = x[c][oh*s - p + i][ow*s - p + j], or 0 in the padding
//------------------------------
static void conv_im2col(const Layer *layer, const real *x, real *col)
{
	Layer_shape in = layer->in_shape;
	int k = layer->config[CONV_KERNEL], s = layer->config[CONV_STRIDE], p = layer->config[CONV_PADDING];
	int oh_n = layer->shape.h, ow_n = layer->shape.w;

	for (int c = 0; c < in.c; c++)
		for (int i = 0; i < k; i++)
			for (int j = 0; j < k; j++)
			{
				real *row = col + ((c * k + i) * k + j) * oh_n * ow_n;
				int lo, hi;
				conv_valid_range(in.w, ow_n, s, p, j, &lo, &hi);

				for (int oh = 0; oh < oh_n; oh++)
				{
					real *out = row + oh * ow_n;
					int ih = oh * s - p + i;
					if (ih < 0 || ih >= in.h)
					{
						memset(out, 0, ow_n * sizeof(real));
						continue;
					}

					const real *src = x + (c * in.h + ih) * in.w + lo * s - p + j;
					for (int ow = 0; ow < lo; ow++)
						out[ow] = (real)0.0;
					if (s == 1)
						memcpy(out + lo, src, (hi - lo) * sizeof(real));
					else
						for (int ow = lo; ow < hi; ow++)
							out[ow] = src[(ow - lo) * s];
					for (int ow = hi; ow < ow_n; ow++)
						out[ow] = (real)0.0;
				}
			}
}

//------------------------------
// the reverse of conv_im2col: add each column entry back to its input position
//------------------------------
static void conv_col2im(const Layer *layer, const real *col, real *x)
{
	Layer_shape in = layer->in_shape;
	int k = layer->config[CONV_KERNEL], s = layer->config[CONV_STRIDE], p = layer->config[CONV_PADDING];
	int oh_n = layer->shape.h, ow_n = layer->shape.w;

	for (int c = 0; c < in.c; c++)
		for (int i = 0; i < k; i++)
			for (int j = 0; j < k; j++)
			{
				const real *row = col + ((c * k + i) * k + j) * oh_n * ow_n;
				int lo, hi;
				conv_valid_range(in.w, ow_n, s, p, j, &lo, &hi);

				for (int oh = 0; oh < oh_n; oh++)
				{
					int ih = oh * s - p + i;
					if (ih < 0 || ih >= in.h)
						continue;

					const real *src = row + oh * ow_n;
					real *dst = x + (c * in.h + ih) * in.w + lo * s - p + j;
					for (int ow = lo; ow < hi; ow++)
						dst[(ow - lo) * s] += src[ow];
				}
			}
}

//------------------------------
// Conv2D forward, per sample: Y_s [F x OH*OW] = W [F x C*k*k] * im2col(X_s) + b
// (rows of Y_s are channels, so the result is already in C, H, W order)
//------------------------------
// Conv2D work split across threads (see ann_parallel.h). Samples are
// independent in the forward pass and for dX; dW and db are split by filter
// and still summed over samples in order, so results match a serial run.
// Without BLAS the per-sample GEMMs run inside the threads; with BLAS they
// stay on the calling thread, which leaves BLAS's own threading alone.
//------------------------------
typedef struct {
	PLayer layer;
	const real *x;		// layer input, one row per sample
	real *y;			// layer output (forward) or dZ (backward)
	real *dx;			// gradient for the layer below (backward), or NULL
	int rows;
} conv_job;

// one sample's im2col columns, kept from the forward pass
static Tensor conv_columns(const Layer *layer, int b)
{
	int patch = layer->params[0].value->cols, positions = layer->shape.h * layer->shape.w;
	return tensor_view(layer->t_scratch[0]->values + (size_t)b * patch * positions, patch, positions);
}

// one sample's column gradient
static Tensor conv_column_grads(const Layer *layer, int b)
{
	int patch = layer->params[0].value->cols, positions = layer->shape.h * layer->shape.w;
	return tensor_view(layer->t_scratch[1]->values + (size_t)b * patch * positions, patch, positions);
}

// add each filter's bias to its row of a sample's output
static void conv_add_bias(const Layer *layer, real *y)
{
	const real *bias = layer->params[1].value->values;
	int positions = layer->shape.h * layer->shape.w;
	for (int f = 0; f < layer->shape.c; f++)
	{
		real *y_row = y + f * positions;
		for (int i = 0; i < positions; i++)
			y_row[i] += bias[f];
	}
}

static void conv_forward_samples(void *arg, int begin, int end)
{
	const conv_job *job = (const conv_job *)arg;
	PLayer layer = job->layer;
	int in_nodes = layer->in_shape.c * layer->in_shape.h * layer->in_shape.w;

	for (int b = begin; b < end; b++)
	{
		Tensor col = conv_columns(layer, b);
		conv_im2col(layer, job->x + b * in_nodes, col.values);
#ifndef USE_BLAS
		Tensor y = tensor_view(job->y + b * layer->node_count, layer->shape.c, col.cols);
		tensor_gemm((real)1.0, layer->params[0].value, &col, (real)0.0, &y);
		conv_add_bias(layer, y.values);
#endif
	}
}

//------------------------------
// Conv2D forward, per sample: Y_s [F x OH*OW] = W [F x C*k*k] * im2col(X_s) + b
// (rows of Y_s are channels, so the result is already in C, H, W order)
//------------------------------
static void conv_forward(PLayer layer, const PTensor X, PTensor Y, int rows)
{
	int patch = layer->params[0].value->cols, positions = layer->shape.h * layer->shape.w;
	conv_job job;

	if (ensure_rows(&layer->t_scratch[0], rows, patch * positions) != ERR_OK)
		return;

	job.layer = layer;
	job.x = X->values;
	job.y = Y->values;
	job.dx = NULL;
	job.rows = rows;
	ann_parallel_for(rows, 1, conv_forward_samples, &job);

#ifdef USE_BLAS
	for (int b = 0; b < rows; b++)
	{
		Tensor col = conv_columns(layer, b);
		Tensor y = tensor_view(Y->values + b * layer->node_count, layer->shape.c, positions);
		tensor_gemm((real)1.0, layer->params[0].value, &col, (real)0.0, &y);
		conv_add_bias(layer, y.values);
	}
#endif
}

// dW and db for filters [begin, end), summed over the samples in order
static void conv_filter_grads(void *arg, int begin, int end)
{
	const conv_job *job = (const conv_job *)arg;
	PLayer layer = job->layer;
	int positions = layer->shape.h * layer->shape.w;
	real *db = layer->params[1].grad->values;

	for (int b = 0; b < job->rows; b++)
	{
		const real *dz = job->y + b * layer->node_count;
#ifndef USE_BLAS
		PTensor dW = layer->params[0].grad;
		Tensor col = conv_columns(layer, b);
		Tensor dz_rows = tensor_view((real *)dz + begin * positions, end - begin, positions);
		Tensor dW_rows = tensor_view(dW->values + begin * dW->cols, end - begin, dW->cols);
		tensor_gemm_transB((real)1.0, &dz_rows, &col, (real)1.0, &dW_rows);
#endif
		for (int f = begin; f < end; f++)
		{
			const real *dz_row = dz + f * positions;
			for (int i = 0; i < positions; i++)
				db[f] += dz_row[i];
		}
	}
}

// dX for samples [begin, end): col2im of each sample's column gradient
static void conv_input_grads(void *arg, int begin, int end)
{
	const conv_job *job = (const conv_job *)arg;
	PLayer layer = job->layer;
	int in_nodes = layer->in_shape.c * layer->in_shape.h * layer->in_shape.w;

	for (int b = begin; b < end; b++)
	{
		Tensor dcol = conv_column_grads(layer, b);
#ifndef USE_BLAS
		Tensor dz = tensor_view(job->y + b * layer->node_count, layer->shape.c, dcol.cols);
		tensor_gemm_transA((real)1.0, layer->params[0].value, &dz, (real)0.0, &dcol);
#endif
		memset(job->dx + b * in_nodes, 0, in_nodes * sizeof(real));
		conv_col2im(layer, dcol.values, job->dx + b * in_nodes);
	}
}

//------------------------------
// Conv2D backward, per sample: dW += dZ_s * col_s^T, db += row sums of dZ_s,
// and dX_s = col2im(W^T * dZ_s), with col_s kept from the forward pass
//------------------------------
static void conv_backward(PLayer layer, const PTensor X, const PTensor dZ, PTensor dX, int rows)
{
	int patch = layer->params[0].value->cols, positions = layer->shape.h * layer->shape.w;
	conv_job job;
	(void)X;

	job.layer = layer;
	job.x = NULL;
	job.y = dZ->values;
	job.dx = dX ? dX->values : NULL;
	job.rows = rows;

#ifdef USE_BLAS
	for (int b = 0; b < rows; b++)
	{
		Tensor col = conv_columns(layer, b);
		Tensor dz = tensor_view(dZ->values + b * layer->node_count, layer->shape.c, positions);
		tensor_gemm_transB((real)1.0, &dz, &col, (real)1.0, layer->params[0].grad);
	}
#endif
	ann_parallel_for(layer->shape.c, 1, conv_filter_grads, &job);

	if (!dX || ensure_rows(&layer->t_scratch[1], rows, patch * positions) != ERR_OK)
		return;

#ifdef USE_BLAS
	for (int b = 0; b < rows; b++)
	{
		Tensor dcol = conv_column_grads(layer, b);
		Tensor dz = tensor_view(dZ->values + b * layer->node_count, layer->shape.c, positions);
		tensor_gemm_transA((real)1.0, layer->params[0].value, &dz, (real)0.0, &dcol);
	}
#endif
	ann_parallel_for(rows, 1, conv_input_grads, &job);
}

//------------------------------
// MaxPool2D: no parameters; check settings and shape. The forward pass
// records which input each output took in t_scratch[0] (one row per sample)
// so the backward pass can route gradients without searching again.
//------------------------------
static int maxpool_setup(PNetwork pnet, int layer)
{
	PLayer cur = &pnet->layers[layer];
	int size = cur->config[POOL_SIZE], stride = cur->config[POOL_STRIDE];

	if (cur->activation != ACTIVATION_NULL || cur->shape.c != cur->in_shape.c ||
		window_output_size(cur->in_shape.h, size, stride, 0) != cur->shape.h ||
		window_output_size(cur->in_shape.w, size, stride, 0) != cur->shape.w || cur->shape.h < 1 || cur->shape.w < 1)
		return ERR_INVALID;
	return ERR_OK;
}

//------------------------------
// MaxPool2D work split across threads by sample
//------------------------------
typedef struct {
	PLayer layer;
	const real *x;		// input (forward) or dZ (backward)
	real *y;			// output (forward) or dX (backward)
} pool_job;

static void maxpool_forward_samples(void *arg, int begin, int end)
{
	const pool_job *job = (const pool_job *)arg;
	PLayer layer = job->layer;
	Layer_shape in = layer->in_shape;
	int size = layer->config[POOL_SIZE], stride = layer->config[POOL_STRIDE];
	int in_nodes = in.c * in.h * in.w;
	int oh_n = layer->shape.h, ow_n = layer->shape.w;

	for (int b = begin; b < end; b++)
	{
		const real *x = job->x + b * in_nodes;
		real *y = job->y + b * layer->node_count;
		real *arg_max = layer->t_scratch[0]->values + (size_t)b * layer->node_count;

		for (int c = 0; c < layer->shape.c; c++)
			for (int oh = 0; oh < oh_n; oh++)
			{
				int out = (c * oh_n + oh) * ow_n;
				int row0 = (c * in.h + oh * stride) * in.w;

				for (int ow = 0; ow < ow_n; ow++, out++)
				{
					int best = row0 + ow * stride;
					if (size == 2)
					{
						// the common 2x2 window, unrolled in the same scan order
						int idx = best;
						if (x[idx + 1] > x[best]) best = idx + 1;
						if (x[idx + in.w] > x[best]) best = idx + in.w;
						if (x[idx + in.w + 1] > x[best]) best = idx + in.w + 1;
					}
					else
					{
						for (int i = 0; i < size; i++)
							for (int j = 0; j < size; j++)
							{
								int idx = row0 + ow * stride + i * in.w + j;
								if (x[idx] > x[best])
									best = idx;
							}
					}
					y[out] = x[best];
					arg_max[out] = (real)best;	// exact: indices are far below 2^24
				}
			}
	}
}

//------------------------------
// MaxPool2D forward: each output is its window's largest input (the first
// one on ties); its index within the sample is saved for backward
//------------------------------
static void maxpool_forward(PLayer layer, const PTensor X, PTensor Y, int rows)
{
	pool_job job;

	if (ensure_rows(&layer->t_scratch[0], rows, layer->node_count) != ERR_OK)
		return;

	job.layer = layer;
	job.x = X->values;
	job.y = Y->values;
	ann_parallel_for(rows, 1, maxpool_forward_samples, &job);
}

//------------------------------
// MaxPool2D backward: each output's gradient goes to the input it took
//------------------------------
static void maxpool_backward_samples(void *arg, int begin, int end)
{
	const pool_job *job = (const pool_job *)arg;
	PLayer layer = job->layer;
	int in_nodes = layer->in_shape.c * layer->in_shape.h * layer->in_shape.w;

	for (int b = begin; b < end; b++)
	{
		const real *dz = job->x + b * layer->node_count;
		const real *arg_max = layer->t_scratch[0]->values + (size_t)b * layer->node_count;
		real *dx = job->y + b * in_nodes;

		memset(dx, 0, in_nodes * sizeof(real));
		for (int out = 0; out < layer->node_count; out++)
			dx[(int)arg_max[out]] += dz[out];
	}
}

static void maxpool_backward(PLayer layer, const PTensor X, const PTensor dZ, PTensor dX, int rows)
{
	pool_job job;
	(void)X;
	if (!dX)
		return;

	job.layer = layer;
	job.x = dZ->values;
	job.y = dX->values;
	ann_parallel_for(rows, 1, maxpool_backward_samples, &job);
}

//------------------------------
// Flatten: a reshape; values are already stored flat, so it copies them
//------------------------------
static int flatten_setup(PNetwork pnet, int layer)
{
	PLayer cur = &pnet->layers[layer];
	int in_nodes = cur->in_shape.c * cur->in_shape.h * cur->in_shape.w;

	if (cur->activation != ACTIVATION_NULL || cur->shape.c != in_nodes || cur->shape.h != 1 || cur->shape.w != 1)
		return ERR_INVALID;
	return ERR_OK;
}

static void flatten_forward(PLayer layer, const PTensor X, PTensor Y, int rows)
{
	memcpy(Y->values, X->values, (size_t)rows * layer->node_count * sizeof(real));
}

static void flatten_backward(PLayer layer, const PTensor X, const PTensor dZ, PTensor dX, int rows)
{
	(void)X;
	if (dX)
		memcpy(dX->values, dZ->values, (size_t)rows * layer->node_count * sizeof(real));
}

//------------------------------
// Operations every layer kind provides
//------------------------------
typedef struct {
	const char *name;
	int config_count;		// how many config[] entries the kind uses (saved in v2 files)
	// check the layer's settings and shape, then allocate and register params[];
	// shape, in_shape, config and node_count are already set. Returns ERR_INVALID
	// for bad settings or ERR_ALLOC.
	int  (*setup)(PNetwork pnet, int layer);
	// NULL for kinds without parameters
	void (*init_weights)(PNetwork pnet, int layer, Weight_init_type init_type);
	// linear part of the forward pass for X [rows x input nodes] into Y
	void (*forward)(PLayer layer, const PTensor X, PTensor Y, int rows);
	// single-sample forward pass; NULL means use forward() with one row
	void (*forward_single)(PLayer layer, const PTensor x, PTensor y);
	// accumulate parameter gradients from dZ; write dX unless it is NULL
	void (*backward)(PLayer layer, const PTensor X, const PTensor dZ, PTensor dX, int rows);
	// ONNX export: the layer's initializers, then its nodes from input to output
	// (the activation node is added by the caller); *first tracks JSON commas.
	// NULL until the kind supports ONNX export.
	void (*onnx_initializers)(FILE *fptr, const PNetwork pnet, int layer, int *first);
	void (*onnx_nodes)(FILE *fptr, const PNetwork pnet, int layer, const char *input,
	                   char *output, size_t output_size, int *first);
} Layer_ops;

// ONNX hooks live with the rest of the ONNX export code
static void dense_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first);
static void dense_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                             char *output, size_t output_size, int *first);
static void conv_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first);
static void conv_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                            char *output, size_t output_size, int *first);
static void no_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first);
static void maxpool_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                               char *output, size_t output_size, int *first);
static void flatten_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                               char *output, size_t output_size, int *first);

// indexed by Layer_kind
static const Layer_ops g_layer_ops[] = {
	{ "Dense", 0, dense_setup, dense_init_weights, dense_forward, dense_forward_single, dense_backward,
	  dense_onnx_initializers, dense_onnx_nodes },
	{ "Conv2D", CONV_CONFIG_COUNT, conv_setup, conv_init_weights, conv_forward, NULL, conv_backward,
	  conv_onnx_initializers, conv_onnx_nodes },
	{ "MaxPool2D", POOL_CONFIG_COUNT, maxpool_setup, NULL, maxpool_forward, NULL, maxpool_backward,
	  no_onnx_initializers, maxpool_onnx_nodes },
	{ "Flatten", 0, flatten_setup, NULL, flatten_forward, NULL, flatten_backward,
	  no_onnx_initializers, flatten_onnx_nodes },
};

static const Layer_ops *layer_ops(const Layer *layer)
{
	return &g_layer_ops[layer->kind];
}

//------------------------------
// initialize the weights
//------------------------------
static void init_weights(PNetwork pnet)
{
	if (!pnet)
		return;
	
	// Don't reinitialize if weights already set
	if (pnet->weights_set)
		return;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		Activation_type activation = pnet->layers[layer].activation;

		// Determine initialization strategy
		Weight_init_type init_type = pnet->weight_init;

		if (init_type == WEIGHT_INIT_AUTO)
		{
			// Auto-select based on activation function
			switch (activation)
			{
			case ACTIVATION_RELU:
			case ACTIVATION_LEAKY_RELU:
			case ACTIVATION_GELU:
				init_type = WEIGHT_INIT_HE;
				break;
			case ACTIVATION_SIGMOID:
			case ACTIVATION_TANH:
			case ACTIVATION_SOFTSIGN:
			case ACTIVATION_SOFTMAX:
				init_type = WEIGHT_INIT_XAVIER;
				break;
			default:
				init_type = WEIGHT_INIT_UNIFORM;
				break;
			}
		}

		const Layer_ops *ops = layer_ops(&pnet->layers[layer]);
		if (ops->init_weights)
			ops->init_weights(pnet, layer, init_type);
	}

	pnet->weights_set = 1;
}

//--------------------------------
// print nodes in the output layer
//--------------------------------
void ann_print_outputs(const PNetwork pnet)
{
	if (!pnet)
		return;

	ann_printf(pnet, "\n");

	PLayer pLayer = &pnet->layers[0];

	pLayer = &pnet->layers[pnet->layer_count - 1];
	tensor_print(pLayer->t_values);
}

//================================================================================================
// ERROR MESSAGE HELPER
//================================================================================================

/**
 * Convert error code to human-readable string.
 * Maps numeric error codes to descriptive messages for debugging and logging.
 * Returns static strings (do not free).
 */
const char* ann_strerror(int error_code)
{
	switch (error_code) {
		case ERR_OK:
			return "Success (ERR_OK)";
		
		case ERR_NULL_PTR:
			return "NULL pointer provided (ERR_NULL_PTR)";
		
		case ERR_ALLOC:
			return "Memory allocation failed (ERR_ALLOC)";
		
		case ERR_INVALID:
			return "Invalid parameter or state (ERR_INVALID)";
		
		case ERR_IO:
			return "File I/O error (ERR_IO)";
		
		case ERR_FAIL:
			return "Generic failure (ERR_FAIL)";
		
		default:
			return "Unknown error code";
	}
}

//================================================================================================
// ERROR LOGGING CALLBACK MANAGEMENT
//================================================================================================

/**
 * Set the error logging callback.
 * Installs a callback function to be called on library errors.
 * Pass NULL to restore the default (print to stderr).
 */
void ann_set_error_log_callback(ErrorLogCallback callback)
{
	g_error_log_callback = callback;
}

/**
 * Get the current error logging callback.
 * Returns NULL if no callback is installed.
 */
ErrorLogCallback ann_get_error_log_callback(void)
{
	return g_error_log_callback;
}

/**
 * Clear the error logging callback, restoring the default stderr output.
 * Equivalent to calling ann_set_error_log_callback(NULL).
 */
void ann_clear_error_log_callback(void)
{
	g_error_log_callback = NULL;
}

/**
 * Internal helper function to invoke the error callback if one is installed.
 * Called internally whenever a library error occurs.
 * 
 * @param error_code The error code that occurred
 * @param function_name The name of the function where error occurred
 */
static void invoke_error_callback(int error_code, const char *function_name)
{
	const char *error_message = ann_strerror(error_code);
	
	if (g_error_log_callback != NULL) {
		g_error_log_callback(error_code, error_message, function_name);
	} else {
		// Default: print to stderr with color
		if (colors_enabled())
			fprintf(stderr, "%s[%s] Error %d: %s%s\n", ANSI_BOLD_RED, function_name, error_code, error_message, ANSI_RESET);
		else
			fprintf(stderr, "[%s] Error %d: %s\n", function_name, error_code, error_message);
	}
}

//---------------------------------------------
// forward evaluate the network
//---------------------------------------------
// Performs forward propagation through the network:
// For each hidden layer: z = Wx + b; a = activation(z)
// For output layer: apply final activation (e.g., softmax for multi-class)
//
// Input is stored in layers[0].t_values by ann_predict() or training code
// Output is in layers[layer_count-1].t_values
//---------------------------------------------
static void eval_network(PNetwork pnet)
{
	if (!pnet)
		return;

#ifdef USE_GPU
	// Dispatch to GPU if a backend is active and the uploaded weights are current
	if (gpu_ready(pnet))
	{
		if (g_gpu_backend->eval_single(pnet))
			return;
		// Fall through to CPU path if GPU eval fails
	}
#endif

	// loop over the non-input layers; each owns the weights that produce it
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		const Layer_ops *ops = layer_ops(cur);

		// linear part, e.g. y = Wx + b
		if (ops->forward_single)
			ops->forward_single(cur, pnet->layers[layer - 1].t_values, cur->t_values);
		else
			ops->forward(cur, pnet->layers[layer - 1].t_values, cur->t_values, 1);

		// apply activation function to values
		for (int i = 0; i < cur->node_count; i++)
		{
			cur->t_values->values[i] = cur->activation_func(cur->t_values->values[i]);
		}
	}

	// apply softmax on output, if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax(pnet);
}

//---------------------------------------------
// Batched softmax: apply softmax per row of batch matrix
// Input/output: t_batch_values[batch × nodes]
//---------------------------------------------
static void softmax_batched(PNetwork pnet, int batch_size)
{
	int output_layer = pnet->layer_count - 1;
	int node_count = pnet->layers[output_layer].node_count;
	PTensor batch = pnet->layers[output_layer].t_batch_values;

	for (int b = 0; b < batch_size; b++)
	{
		real *row = batch->values + b * node_count;

		// Find max for numerical stability
		real max_val = row[0];
		for (int n = 1; n < node_count; n++)
		{
			if (row[n] > max_val)
				max_val = row[n];
		}

		// Fused exp + sum + normalize in two passes (down from three)
		real sum = (real)0.0;
		for (int n = 0; n < node_count; n++)
		{
			row[n] = (real)exp(row[n] - max_val);
			sum += row[n];
		}

		real inv_sum = (real)1.0 / sum;
		for (int n = 0; n < node_count; n++)
			row[n] *= inv_sum;
	}
}

//---------------------------------------------
// Draw a hidden layer's dropout mask for this batch (inverted dropout):
// 0 for dropped neurons, 1/(1-rate) for kept ones. The mask is kept in
// t_dropout_mask for backprop. Returns NULL when dropout is off for the layer.
//---------------------------------------------
static PTensor make_dropout_mask(PNetwork pnet, int layer_idx, int batch_size)
{
	PLayer layer = &pnet->layers[layer_idx];
	real rate = layer->dropout_rate;
	
	// Skip if dropout disabled or not training
	if (rate <= (real)0.0 || !pnet->is_training)
		return NULL;
	
	int nodes = layer->node_count;
	
	// Allocate or reallocate dropout mask if needed
	if (!layer->t_dropout_mask || 
	    layer->t_dropout_mask->rows != batch_size ||
	    layer->t_dropout_mask->cols != nodes)
	{
		tensor_free(layer->t_dropout_mask);
		layer->t_dropout_mask = tensor_create(batch_size, nodes);
	}
	
	PTensor mask = layer->t_dropout_mask;
	if (!mask)
		return NULL;
	
	real scale = (real)1.0 / ((real)1.0 - rate);
	int total = batch_size * nodes;
	for (int i = 0; i < total; i++)
	{
		real r = (real)rand() / (real)RAND_MAX;
		mask->values[i] = (r < rate) ? (real)0.0 : scale;
	}
	return mask;
}

//---------------------------------------------
// Apply dropout to a layer's activations: draw the mask and multiply
// Only applied during training; mask is stored for backprop
//---------------------------------------------
static void apply_dropout_batched(PNetwork pnet, int layer_idx, int batch_size)
{
	PTensor mask = make_dropout_mask(pnet, layer_idx, batch_size);
	if (!mask)
		return;
	
	PTensor Y = pnet->layers[layer_idx].t_batch_values;
	int total = batch_size * pnet->layers[layer_idx].node_count;
	for (int i = 0; i < total; i++)
		Y->values[i] *= mask->values[i];
}

#ifdef USE_GPU
//---------------------------------------------
// Draw every hidden layer's dropout mask for a GPU batch. The GPU applies
// them, but drawing them here, in layer order, uses the same random sequence
// as CPU training, so both paths drop the same neurons.
//---------------------------------------------
static void make_dropout_masks(PNetwork pnet, int batch_size)
{
	for (int layer = 1; layer < pnet->layer_count - 1; layer++)
		make_dropout_mask(pnet, layer, batch_size);
}
#endif

//---------------------------------------------
// Apply an activation in place to values [begin, end) (inlined per activation
// so the loops vectorize)
//---------------------------------------------
typedef struct {
	Activation_type act;
	real *values;
} activation_job;

static void activation_range(void *arg, int begin, int end)
{
	const activation_job *job = (const activation_job *)arg;
	real *y = job->values;

	switch (job->act)
	{
	case ACTIVATION_SIGMOID:
		for (int i = begin; i < end; i++)
			y[i] = (real)1.0 / ((real)1.0 + (real)exp(-y[i]));
		break;
	case ACTIVATION_RELU:
		for (int i = begin; i < end; i++)
			y[i] = y[i] > (real)0.0 ? y[i] : (real)0.0;
		break;
	case ACTIVATION_LEAKY_RELU:
		for (int i = begin; i < end; i++)
			y[i] = y[i] > (real)0.0 ? y[i] : (real)0.01 * y[i];
		break;
	case ACTIVATION_TANH:
		for (int i = begin; i < end; i++)
			y[i] = (real)tanh(y[i]);
		break;
	case ACTIVATION_SOFTSIGN:
		for (int i = begin; i < end; i++)
			y[i] = y[i] / ((real)1.0 + (real)fabs(y[i]));
		break;
	case ACTIVATION_GELU:
		for (int i = begin; i < end; i++)
			y[i] = gelu(y[i]);
		break;
	default:
		break;
	}
}

//---------------------------------------------
// Batched forward propagation
// Input: layers[0].t_batch_values has input data [batch_size × input_nodes]
// Output: all layers have t_batch_values populated
// Also stores pre-activation values in t_batch_z for backprop derivatives
//---------------------------------------------
static void eval_network_batched(PNetwork pnet, int batch_size)
{
	if (!pnet)
		return;

	// Loop over non-input layers (each owns the parameters from the layer below)
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		PTensor X = pnet->layers[layer - 1].t_batch_values;  // [batch × in_nodes]
		PTensor Y = cur->t_batch_values;                     // [batch × out_nodes]
		PTensor Z = cur->t_batch_z;                          // [batch × out_nodes]

		// Linear part (e.g. Y = X * W^T + b), then save the pre-activation Z
		layer_ops(cur)->forward(cur, X, Y, batch_size);

		Activation_type act_type = cur->activation;
		int total = batch_size * cur->node_count;
		memcpy(Z->values, Y->values, total * sizeof(real));

		// Activation (softmax is applied to the output layer below)
		if (act_type != ACTIVATION_SOFTMAX && act_type != ACTIVATION_NULL)
		{
			activation_job job;
			job.act = act_type;
			job.values = Y->values;
			ann_parallel_for(total, ANN_ELEMENT_GRAIN, activation_range, &job);
		}
		
		// Apply dropout to hidden layers (not output layer)
		if (layer < pnet->layer_count - 1)
		{
			apply_dropout_batched(pnet, layer, batch_size);
		}
	}

	// Apply softmax on output if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax_batched(pnet, batch_size);
}

//-------------------------------------------
// One output's cross-entropy loss term. Softmax outputs use the categorical
// form -t*log(y). Sigmoid outputs use binary cross-entropy,
// -(t*log(y) + (1-t)*log(1-y)), so targets of 0 count too; T - Y is its exact
// gradient. y is clamped away from 0 and 1 so a confidently wrong prediction
// costs the most instead of nothing.
// Shared with the GPU backends (declared in ann_gpu_backend.h).
//-------------------------------------------
real ann_cross_entropy_term(Activation_type output_act, real t, real y)
{
	const real lo = (real)1e-7, hi = (real)1.0 - (real)1e-7;
	if (y < lo) y = lo;
	if (y > hi) y = hi;

	if (output_act == ACTIVATION_SIGMOID)
		return -(t * (real)log(y) + ((real)1.0 - t) * (real)log((real)1.0 - y));
	return -t * (real)log(y);
}

//-------------------------------------------
// Sum of one sample's loss terms over its outputs. Callers divide by
// rows * outputs for MSE, or by rows for cross-entropy.
//-------------------------------------------
static real sample_loss(const PNetwork pnet, const real *t, const real *y, int nodes)
{
	Activation_type output_act = pnet->layers[pnet->layer_count - 1].activation;
	real sum = (real)0.0;

	for (int n = 0; n < nodes; n++)
	{
		if (pnet->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY)
		{
			// Cross-entropy (binary for sigmoid outputs)
			sum += ann_cross_entropy_term(output_act, t[n], y[n]);
		}
		else
		{
			// MSE: sum((y - t)^2)
			real diff = y[n] - t[n];
			sum += diff * diff;
		}
	}

	return sum;
}

typedef struct {
	Activation_type act;
	real *delta;
	const real *Y, *Z;
	int nodes;
} derivative_job;

// delta *= f'(...) for elements [begin, end), or rows [begin, end) for softmax
static void derivative_range(void *arg, int begin, int end)
{
	const derivative_job *job = (const derivative_job *)arg;
	real *delta = job->delta;
	const real *Y = job->Y, *Z = job->Z;

	switch (job->act)
	{
	case ACTIVATION_SIGMOID:
		for (int i = begin; i < end; i++)
			delta[i] *= Y[i] * ((real)1.0 - Y[i]);
		break;

	case ACTIVATION_TANH:
		for (int i = begin; i < end; i++)
			delta[i] *= (real)1.0 - Y[i] * Y[i];
		break;

	case ACTIVATION_SOFTSIGN:
		for (int i = begin; i < end; i++)
		{
			real d = (real)1.0 - (real)fabs(Y[i]);
			delta[i] *= d * d;
		}
		break;

	// branch-free selects so the loops vectorize
	case ACTIVATION_RELU:
		for (int i = begin; i < end; i++)
			delta[i] = Z[i] <= (real)0.0 ? (real)0.0 : delta[i];
		break;

	case ACTIVATION_LEAKY_RELU:
		for (int i = begin; i < end; i++)
			delta[i] = Z[i] <= (real)0.0 ? delta[i] * (real)0.01 : delta[i];
		break;

	case ACTIVATION_GELU:
		for (int i = begin; i < end; i++)
			delta[i] *= gelu_derivative(Z[i]);
		break;

	case ACTIVATION_SOFTMAX:
		for (int b = begin; b < end; b++)
		{
			real *d = delta + b * job->nodes;
			const real *y = Y + b * job->nodes;
			real dot = (real)0.0;
			for (int n = 0; n < job->nodes; n++)
				dot += d[n] * y[n];
			for (int n = 0; n < job->nodes; n++)
				d[n] = y[n] * (d[n] - dot);
		}
		break;

	default:
		// linear output: delta is already -dL/dz
		break;
	}
}

//-------------------------------------------
// Turn the MSE output error delta = T - Y into the gradient with respect to
// the output pre-activations, -dL/dz, by applying the output activation's
// derivative. Hidden layers use it the same way to turn dL/da into dL/dz.
// Cross-entropy with softmax already has -dL/dz = T - Y. Softmax uses its
// full Jacobian per row: delta_j = y_j * (d_j - sum_i d_i * y_i).
// Shared with the GPU backends (declared in ann_gpu_backend.h) so every path
// computes the same output gradient.
//-------------------------------------------
void ann_mse_output_delta(Activation_type act, real *delta, const real *Y, const real *Z,
                          int batch_size, int nodes)
{
	derivative_job job;
	job.act = act;
	job.delta = delta;
	job.Y = Y;
	job.Z = Z;
	job.nodes = nodes;

	if (act == ACTIVATION_SOFTMAX)
		ann_parallel_for(batch_size, ANN_ELEMENT_GRAIN / (nodes > 0 ? nodes : 1), derivative_range, &job);
	else if (act != ACTIVATION_NULL)
		ann_parallel_for(batch_size * nodes, ANN_ELEMENT_GRAIN, derivative_range, &job);
}

//-------------------------------------------
// Batched output layer backpropagation
// Computes: delta = predicted - target for full batch
// Gradient: dW = delta^T * A_prev (using gemm)
// Bias gradient: column sums of delta
// Propagates: dl_dz_prev = delta * W
//-------------------------------------------
static real back_propagate_output_batched(PNetwork pnet, int batch_size, PTensor targets)
{
	int output_layer = pnet->layer_count - 1;
	PLayer layer = &pnet->layers[output_layer];
	PLayer prev_layer = &pnet->layers[output_layer - 1];
	
	int out_nodes = layer->node_count;
	
	PTensor Y = layer->t_batch_values;           // [batch × out] predictions
	PTensor A_prev = prev_layer->t_batch_values; // [batch × in] previous activations
	PTensor delta = layer->t_batch_dl_dz;        // [batch × out] output delta
	
	// Compute delta = T - Y (same convention as original: gradient direction for descent)
	real total_loss = (real)0.0;
	
	for (int b = 0; b < batch_size; b++)
	{
		int out_offset = b * out_nodes;
		int target_offset = b * out_nodes;
		
		for (int n = 0; n < out_nodes; n++)
		{
			// delta = t - y (matches original convention for gradient descent)
			delta->values[out_offset + n] = targets->values[target_offset + n] - Y->values[out_offset + n];
		}

		// Accumulate loss (cross-entropy or MSE based on loss function)
		total_loss += sample_loss(pnet, targets->values + target_offset, Y->values + out_offset, out_nodes);
	}
	
	// Average loss over batch
	if (pnet->loss_type == LOSS_MSE)
		total_loss /= (real)(batch_size * out_nodes);
	else
		total_loss /= (real)batch_size;

	// MSE needs the output activation's derivative; cross-entropy (with
	// softmax) does not
	if (pnet->loss_type == LOSS_MSE)
		ann_mse_output_delta(layer->activation, delta->values, Y->values,
		                     layer->t_batch_z->values, batch_size, out_nodes);
	
	// Parameter gradients, and the gradient for the layer below (the input
	// layer needs none)
	layer_ops(layer)->backward(layer, A_prev, delta, output_layer > 1 ? prev_layer->t_batch_dl_dz : NULL, batch_size);
	
	return total_loss;
}

//-------------------------------------------
// Undo dropout before a hidden layer's activation derivative:
// - mask the gradient (dropped neurons get none, kept ones are scaled)
// - restore kept activations to their pre-dropout values, A / scale, so the
//   sigmoid/tanh/softsign derivatives are evaluated at the right point
// Safe because this layer's dropped-out A was already used, as A_prev, by the
// layer above it, and nothing reads it after the derivative.
//-------------------------------------------
static void apply_dropout_mask_to_gradient(PLayer layer, int batch_size)
{
	if (!layer->t_dropout_mask || layer->dropout_rate <= (real)0.0)
		return;
	
	real *dl_dz = layer->t_batch_dl_dz->values;
	real *A = layer->t_batch_values->values;
	const real *mask = layer->t_dropout_mask->values;
	int total = batch_size * layer->node_count;
	
	for (int i = 0; i < total; i++)
	{
		dl_dz[i] *= mask[i];
		if (mask[i] != (real)0.0)
			A[i] /= mask[i];
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation. On entry t_batch_dl_dz holds the
// gradient with respect to the layer's activations; undo dropout, apply the
// activation's derivative (the same one the output layer uses with MSE), then
// let the layer kind accumulate its parameter gradients and propagate.
//-------------------------------------------
static void back_propagate_hidden_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	PTensor dl_dz = layer->t_batch_dl_dz;        // [batch × nodes]
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// dL/dz = dL/da * f'(z)
	ann_mse_output_delta(layer->activation, dl_dz->values, layer->t_batch_values->values,
	                     layer->t_batch_z->values, batch_size, layer->node_count);
	
	// Gradients, and propagation unless the layer below is the input
	layer_ops(layer)->backward(layer, prev_layer->t_batch_values, dl_dz,
	                           layer_idx > 1 ? prev_layer->t_batch_dl_dz : NULL, batch_size);
}

//-------------------------------------------
// Batched backpropagation through all layers
// Returns total loss for the batch
//-------------------------------------------
static real back_propagate_batched(PNetwork pnet, int batch_size, PTensor targets)
{
	int output_layer = pnet->layer_count - 1;
	
	// Output layer backprop (returns loss)
	real loss = back_propagate_output_batched(pnet, batch_size, targets);
	
	// Hidden layers backprop (from output-1 to layer 1)
	for (int layer = output_layer - 1; layer > 0; layer--)
		back_propagate_hidden_batched(pnet, batch_size, layer);
	
	return loss;
}

//-----------------------------------------------
// Allocate or reallocate batch tensors for all layers
// Called when batch size changes during training
// Returns ERR_OK on success, error code on failure
//-----------------------------------------------
static int ensure_batch_tensors(PNetwork pnet, unsigned batch_size)
{
	if (!pnet)
		return ERR_NULL_PTR;

	// Already allocated with correct size
	if (pnet->current_batch_size == batch_size)
		return ERR_OK;

	// Allocate or reallocate batch tensors for each layer
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		int nodes = pnet->layers[layer].node_count;

		// Free existing tensors if present
		tensor_free(pnet->layers[layer].t_batch_values);
		tensor_free(pnet->layers[layer].t_batch_dl_dz);
		tensor_free(pnet->layers[layer].t_batch_z);

		// Allocate new tensors [batch_size × nodes]
		pnet->layers[layer].t_batch_values = tensor_zeros(batch_size, nodes);
		if (!pnet->layers[layer].t_batch_values)
		{
			invoke_error_callback(ERR_ALLOC, "ensure_batch_tensors");
			return ERR_ALLOC;
		}

		// Only non-input layers need gradient and pre-activation tensors
		if (layer > 0)
		{
			pnet->layers[layer].t_batch_dl_dz = tensor_zeros(batch_size, nodes);
			pnet->layers[layer].t_batch_z = tensor_zeros(batch_size, nodes);
			if (!pnet->layers[layer].t_batch_dl_dz || !pnet->layers[layer].t_batch_z)
			{
				invoke_error_callback(ERR_ALLOC, "ensure_batch_tensors");
				return ERR_ALLOC;
			}
		}
		else
		{
			pnet->layers[layer].t_batch_dl_dz = NULL;
			pnet->layers[layer].t_batch_z = NULL;
		}
	}

	pnet->current_batch_size = batch_size;
	return ERR_OK;
}

//-----------------------------------------------
// shuffle the indices
// Implements Fisher-Yates shuffle algorithm
// Purpose: Randomize training data order each epoch
// Benefits: Reduces bias from data ordering, improves generalization
// Algorithm: For each position i, swap with random position j >= i
// https://en.wikipedia.org/wiki/Fisher–Yates_shuffle
// Time complexity: O(n) with single pass
//-----------------------------------------------
static void shuffle_indices(int *input_indices, int count)
{
	// Knuth's algorithm to shuffle an array a of n elements (indices 0..n-1):
	for (int i = 0; i < count - 1; i++)
	{
		int j = i + (rand() % (count - i));
		int val = input_indices[i];
		input_indices[i] = input_indices[j];
		input_indices[j] = val;
	}
}

//--------------------------------------------------------
// Clip gradients to prevent exploding gradients
// Called before applying gradients if max_gradient > 0
//--------------------------------------------------------
static void clip_gradients(PNetwork pnet)
{
	if (pnet->max_gradient <= (real)0.0)
		return;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
			tensor_clip(cur->params[p].grad, -pnet->max_gradient, pnet->max_gradient);
	}
}

//--------------------------------------------------------
// Apply L1/L2 weight regularization
// L2 (weight decay): W = W * (1 - lr * lambda)
// L1 (LASSO): W = W - lr * lambda * sign(W)
// Note: Only applied to weights, not biases (standard practice)
//--------------------------------------------------------
static void apply_regularization(PNetwork pnet)
{
	real lr = pnet->learning_rate;
	real l2 = pnet->l2_lambda;
	real l1 = pnet->l1_lambda;

	if (l2 <= (real)0.0 && l1 <= (real)0.0)
		return;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			if (!cur->params[p].regularize)
				continue;

			PTensor W = cur->params[p].value;
			int size = W->rows * W->cols;

			// L2 regularization: W = W * (1 - lr * l2)
			if (l2 > (real)0.0)
			{
				real decay = (real)1.0 - lr * l2;
				tensor_mul_scalar(W, decay);
			}

			// L1 regularization: W = W - lr * l1 * sign(W)
			if (l1 > (real)0.0)
			{
				real *w = W->values;
				real delta = lr * l1;
				for (int i = 0; i < size; i++)
				{
					if (w[i] > (real)0.0)
						w[i] -= delta;
					else if (w[i] < (real)0.0)
						w[i] += delta;
				}
			}
		}
	}
}

//--------------------------------------------------------
// Element-wise optimizer updates for one parameter, over elements
// [begin, end) (split across threads for large parameters)
//--------------------------------------------------------
typedef struct {
	Layer_param *param;
	real lr;
	real beta1;			// RMSProp: beta
	real beta2;			// RMSProp: 1 - beta
	real epsilon;
	real bias_correction1, bias_correction2;	// Adam
} optimizer_job;

// v = v + g^2, then W = W + lr * g / (sqrt(v) + epsilon)
static void adagrad_range(void *arg, int begin, int end)
{
	const optimizer_job *job = (const optimizer_job *)arg;
	const real *g = job->param->grad->values;
	real *v = job->param->v->values, *w = job->param->value->values;

	for (int i = begin; i < end; i++)
	{
		real grad = g[i];
		v[i] += grad * grad;
		w[i] += job->lr * grad / ((real)sqrt(v[i]) + job->epsilon);
	}
}

// v = beta * v + (1 - beta) * g^2, then W = W + lr * g / (sqrt(v) + epsilon)
static void rmsprop_range(void *arg, int begin, int end)
{
	const optimizer_job *job = (const optimizer_job *)arg;
	const real *g = job->param->grad->values;
	real *v = job->param->v->values, *w = job->param->value->values;

	for (int i = begin; i < end; i++)
	{
		real grad = g[i];
		v[i] = job->beta1 * v[i] + job->beta2 * grad * grad;
		w[i] += job->lr * grad / ((real)sqrt(v[i]) + job->epsilon);
	}
}

static void adam_range(void *arg, int begin, int end)
{
	const optimizer_job *job = (const optimizer_job *)arg;
	const real *g = job->param->grad->values;
	real *m = job->param->m->values, *v = job->param->v->values, *w = job->param->value->values;
	real beta1 = job->beta1, beta2 = job->beta2;

	for (int i = begin; i < end; i++)
	{
		real grad = g[i];

		// Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
		m[i] = beta1 * m[i] + ((real)1.0 - beta1) * grad;

		// Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
		v[i] = beta2 * v[i] + ((real)1.0 - beta2) * grad * grad;

		// Compute bias-corrected estimates
		real mhat = m[i] * job->bias_correction1;
		real vhat = v[i] * job->bias_correction2;

		// Update: W = W + lr * mhat / (sqrt(vhat) + epsilon)
		w[i] += job->lr * mhat / ((real)sqrt(vhat) + job->epsilon);
	}
}

//--------------------------------------------------------
// Stochastic Gradient Descent (SGD)
//
// update the weights based on gradients
// W = W + learning_rate * gradients
// 
// Simplest optimization: pure gradient descent with constant learning rate.
// Good baseline but can oscillate or get stuck in local minima.
//--------------------------------------------------------
static void optimize_sgd(PNetwork pnet)
{
	clip_gradients(pnet);

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			// W = W + n * gradients
			tensor_axpy(pnet->learning_rate, cur->params[p].grad, cur->params[p].value);
		}
	}

	apply_regularization(pnet);
}

//-----------------------------------------------
// Gradient descent with momentum
//-----------------------------------------------
// Momentum-based optimizer helps escape local minima and accelerates convergence.
// m = beta * m + (1-beta) * gradients  (exponential moving average of gradients)
// W = W + learning_rate * m
// 
// Default beta=0.9: favors recent gradient history to "build momentum"
// Usually faster convergence than plain SGD
//-----------------------------------------------
static void optimize_momentum(PNetwork pnet)
{
	clip_gradients(pnet);

	real beta = (real)0.9, one_minus_beta = (real)0.1;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			Layer_param *param = &cur->params[p];

			// momentum = beta * m + one_minus_beta * gradients
			tensor_axpby(one_minus_beta, param->grad, beta, param->m);

			// W = W + n * m
			tensor_axpy(pnet->learning_rate, param->m, param->value);
		}
	}

	apply_regularization(pnet);
}

//-----------------------------------------------
// Adaptive gradient descent
// Accumulates squared gradients and scales learning
// rate by inverse square root of accumulated sum.
// Good for sparse gradients but learning rate
// monotonically decreases (may stop learning).
//-----------------------------------------------
static void optimize_adagrad(PNetwork pnet)
{
	clip_gradients(pnet);

	real epsilon = (real)1e-8;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			optimizer_job job = { &cur->params[p], pnet->learning_rate, (real)0.0, (real)0.0, epsilon, (real)0.0, (real)0.0 };
			PTensor g = cur->params[p].grad;
			ann_parallel_for(g->rows * g->cols, ANN_ELEMENT_GRAIN, adagrad_range, &job);
		}
	}

	apply_regularization(pnet);
}

//-----------------------------------------------
// RMSProp - Root Mean Square Propagation
// Uses exponential moving average of squared gradients
// to scale learning rate. Fixes AdaGrad's diminishing
// learning rate by using decay factor beta.
//-----------------------------------------------
static void optimize_rmsprop(PNetwork pnet)
{
	clip_gradients(pnet);

	real beta = (real)0.9;
	real one_minus_beta = (real)0.1;
	real epsilon = (real)1e-8;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			optimizer_job job = { &cur->params[p], pnet->learning_rate, beta, one_minus_beta, epsilon, (real)0.0, (real)0.0 };
			PTensor g = cur->params[p].grad;
			ann_parallel_for(g->rows * g->cols, ANN_ELEMENT_GRAIN, rmsprop_range, &job);
		}
	}

	apply_regularization(pnet);
}

//-----------------------------------------------
// Adam - Adaptive Moment Estimation
// Combines momentum (first moment) and RMSProp (second moment)
// with bias correction for both. Generally the best default
// optimizer for most deep learning tasks.
//-----------------------------------------------
static void optimize_adam(PNetwork pnet)
{
	clip_gradients(pnet);

	real beta1 = (real)0.9;
	real beta2 = (real)0.999;
	real epsilon = (real)1e-8;

	// Bias correction factors (train_iteration is advanced by the caller,
	// once per batch, before the optimizer runs)
	real bias_correction1 = (real)1.0 / ((real)1.0 - (real)pow(beta1, pnet->train_iteration));
	real bias_correction2 = (real)1.0 / ((real)1.0 - (real)pow(beta2, pnet->train_iteration));

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			optimizer_job job = { &cur->params[p], pnet->learning_rate, beta1, beta2, epsilon,
			                      bias_correction1, bias_correction2 };
			PTensor g = cur->params[p].grad;
			ann_parallel_for(g->rows * g->cols, ANN_ELEMENT_GRAIN, adam_range, &job);
		}
	}

	apply_regularization(pnet);
}

//[]---------------------------------------------[]
//	Public ANN library interfaces
//[]---------------------------------------------[]

//------------------------------
// add a layer of any kind with the given output shape and kind settings
// (config may be NULL when config_count is 0)
//------------------------------
static int add_layer(PNetwork pnet, Layer_kind kind, Layer_shape shape, const int *config, int config_count,
                     Layer_type layer_type, Activation_type activation_type)
{
	if (!pnet) {
		invoke_error_callback(ERR_NULL_PTR, "ann_add_layer");
		return ERR_NULL_PTR;
	}

	int node_count = shape.c * shape.h * shape.w;
	if (shape.c <= 0 || shape.h <= 0 || shape.w <= 0 ||
		(int)kind < 0 || (size_t)kind >= sizeof(g_layer_ops) / sizeof(g_layer_ops[0]) ||
		config_count < 0 || config_count > ANN_LAYER_CONFIG_MAX ||
		(kind != LAYER_KIND_DENSE && pnet->layer_count == 0)) {
		invoke_error_callback(ERR_INVALID, "ann_add_layer");
		return ERR_INVALID;
	}

	// check whether we've run out of layers
	pnet->layer_count++;
	if (pnet->layer_count > pnet->layer_size)
	{
		// need to allocate more layers
		int old_size = pnet->layer_size;
		pnet->layer_size <<= 1;
		PLayer layer = realloc(pnet->layers, pnet->layer_size * (sizeof(Layer)));
		if (NULL == layer)
		{
			pnet->layer_count--;  // rollback count
			pnet->layer_size = old_size;  // rollback size
			invoke_error_callback(ERR_ALLOC, "ann_add_layer");
			return ERR_ALLOC;
		}

		pnet->layers = layer;
		
		// Initialize the newly allocated layers: every tensor NULL, dropout off
		memset(&pnet->layers[old_size], 0, (pnet->layer_size - old_size) * sizeof(Layer));
	}

	int cur_layer = pnet->layer_count - 1;

	// start from a clean slot (an earlier failed add may have used it)
	memset(&pnet->layers[cur_layer], 0, sizeof(Layer));
	pnet->layers[cur_layer].layer_type = layer_type;
	pnet->layers[cur_layer].activation = activation_type;

	switch (activation_type)
	{
	case ACTIVATION_SIGMOID:
		pnet->layers[cur_layer].activation_func = sigmoid;
		break;

	case ACTIVATION_RELU:
		pnet->layers[cur_layer].activation_func = relu;
		break;

	case ACTIVATION_LEAKY_RELU:
		pnet->layers[cur_layer].activation_func = leaky_relu;
		break;

	case ACTIVATION_TANH:
		pnet->layers[cur_layer].activation_func = ann_tanh;
		break;

	case ACTIVATION_SOFTSIGN:
		pnet->layers[cur_layer].activation_func = softsign;
		break;

	case ACTIVATION_GELU:
		pnet->layers[cur_layer].activation_func = gelu;
		break;

	case ACTIVATION_NULL:
		pnet->layers[cur_layer].activation_func = no_activation;
		break;

	case ACTIVATION_SOFTMAX:
		pnet->layers[cur_layer].activation_func = no_activation;
		// handled after full network is evaluated
		break;

	default:
		pnet->layer_count--;
		invoke_error_callback(ERR_INVALID, "ann_add_layer");
		return ERR_INVALID;
	}

	//--------------------
	// allocate the nodes
	//--------------------

	// create the node values tensor
	pnet->layers[cur_layer].t_values 	= tensor_zeros(1, node_count);
	if (pnet->layers[cur_layer].t_values == NULL)
	{
		pnet->layer_count--;
		invoke_error_callback(ERR_ALLOC, "ann_add_layer");
		return ERR_ALLOC;
	}
	pnet->layers[cur_layer].node_count	= node_count;
	pnet->layers[cur_layer].kind		= kind;
	pnet->layers[cur_layer].shape		= shape;
	if (cur_layer > 0)
		pnet->layers[cur_layer].in_shape = pnet->layers[cur_layer - 1].shape;
	for (int i = 0; i < config_count; i++)
		pnet->layers[cur_layer].config[i] = config[i];

	// check the kind's settings and create the parameters that produce this
	// layer from the one below
	int err = cur_layer > 0 ? layer_ops(&pnet->layers[cur_layer])->setup(pnet, cur_layer) : ERR_OK;
	if (err != ERR_OK)
	{
		tensor_free(pnet->layers[cur_layer].t_values);
		pnet->layers[cur_layer].t_values = NULL;
		pnet->layer_count--;
		invoke_error_callback(err, "ann_add_layer");
		return err;
	}

	return ERR_OK;
}

//------------------------------
// shape of the last layer added; NULL for an empty network
//------------------------------
static const Layer_shape *last_shape(const PNetwork pnet)
{
	return (pnet && pnet->layer_count > 0) ? &pnet->layers[pnet->layer_count - 1].shape : NULL;
}

//------------------------------
// add an input layer of shape (channels, height, width)
//------------------------------
int ann_add_input(PNetwork pnet, int channels, int height, int width)
{
	if (pnet && pnet->layer_count != 0)
	{
		invoke_error_callback(ERR_INVALID, "ann_add_input");
		return ERR_INVALID;
	}

	Layer_shape shape;
	shape.c = channels;
	shape.h = height;
	shape.w = width;
	return add_layer(pnet, LAYER_KIND_DENSE, shape, NULL, 0, LAYER_INPUT, ACTIVATION_NULL);
}

//------------------------------
// add a 2-D convolution layer
//------------------------------
int ann_add_conv2d(PNetwork pnet, int filters, int kernel_size, int stride, int padding, Activation_type activation)
{
	const Layer_shape *in = last_shape(pnet);
	Layer_shape shape;
	int config[CONV_CONFIG_COUNT];

	config[CONV_KERNEL] = kernel_size;
	config[CONV_STRIDE] = stride;
	config[CONV_PADDING] = padding;
	shape.c = filters;
	shape.h = in ? window_output_size(in->h, kernel_size, stride, padding) : 1;
	shape.w = in ? window_output_size(in->w, kernel_size, stride, padding) : 1;

	// a window that does not fit gives a zero size (add_layer rejects a conv
	// layer in an empty network, and filters < 1)
	if (in && (shape.h < 1 || shape.w < 1))
	{
		invoke_error_callback(ERR_INVALID, "ann_add_conv2d");
		return ERR_INVALID;
	}
	return add_layer(pnet, LAYER_KIND_CONV2D, shape, config, CONV_CONFIG_COUNT, LAYER_HIDDEN, activation);
}

//------------------------------
// add a 2-D max pooling layer
//------------------------------
int ann_add_maxpool2d(PNetwork pnet, int pool_size, int stride)
{
	const Layer_shape *in = last_shape(pnet);
	Layer_shape shape;
	int config[POOL_CONFIG_COUNT];

	config[POOL_SIZE] = pool_size;
	config[POOL_STRIDE] = stride;
	shape.c = in ? in->c : 1;
	shape.h = in ? window_output_size(in->h, pool_size, stride, 0) : 1;
	shape.w = in ? window_output_size(in->w, pool_size, stride, 0) : 1;

	if (in && (shape.h < 1 || shape.w < 1))
	{
		invoke_error_callback(ERR_INVALID, "ann_add_maxpool2d");
		return ERR_INVALID;
	}
	return add_layer(pnet, LAYER_KIND_MAXPOOL2D, shape, config, POOL_CONFIG_COUNT, LAYER_HIDDEN, ACTIVATION_NULL);
}

//------------------------------
// add a flatten layer
//------------------------------
int ann_add_flatten(PNetwork pnet)
{
	const Layer_shape *in = last_shape(pnet);
	Layer_shape shape;

	shape.c = in ? in->c * in->h * in->w : 1;
	shape.h = 1;
	shape.w = 1;
	return add_layer(pnet, LAYER_KIND_FLATTEN, shape, NULL, 0, LAYER_HIDDEN, ACTIVATION_NULL);
}

//------------------------------
// add a new dense layer to the network
//------------------------------
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type)
{
	Layer_shape shape;
	shape.c = node_count;
	shape.h = 1;
	shape.w = 1;
	return add_layer(pnet, LAYER_KIND_DENSE, shape, NULL, 0, layer_type, activation_type);
}

//------------------------------
// make a new network
//------------------------------
PNetwork ann_make_network(Optimizer_type opt, Loss_type loss_type)
{
	PNetwork pnet = malloc(sizeof(Network));
	if (NULL == pnet)
		return NULL;

	// set default values
	pnet->layer_size	= DEFAULT_LAYERS;
	pnet->layers		= malloc(pnet->layer_size * (sizeof(Layer)));
	if (NULL == pnet->layers)
	{
		free(pnet);
		return NULL;
	}

	pnet->layer_count	= 0;
	pnet->learning_rate = (real)DEFAULT_LEARNING_RATE;
	pnet->weights_set	= 0;
	pnet->convergence_epsilon = (real)DEFAULT_CONVERGENCE;
	pnet->mseCounter	= 0;
	pnet->dbg			= NULL;
	pnet->weight_limit	= R_MAX;
	pnet->init_bias		= (real)1.0;

	// every tensor NULL, dropout off
	memset(pnet->layers, 0, pnet->layer_size * sizeof(Layer));

	for (int i = 0; i < DEFAULT_MSE_AVG; i++)
	{
		pnet->lastMSE[i] = (real)0.0;
	}

	ann_set_loss_function(pnet, loss_type);

	pnet->epochLimit		= 10000;
	pnet->train_iteration 	= 0;
	pnet->batchSize			= DEFAULT_BATCH_SIZE;
	pnet->current_batch_size = 0;	// batch tensors not yet allocated
	pnet->print_func		= ann_puts;
	pnet->optimizer			= opt;
	pnet->max_gradient		= (real)0.0;		// gradient clipping disabled by default
	pnet->weight_init		= WEIGHT_INIT_AUTO;	// auto-select based on activation
	pnet->lr_scheduler		= NULL;				// no scheduler by default
	pnet->lr_scheduler_data	= NULL;
	pnet->base_learning_rate = (real)0.0;		// set when training starts
	pnet->default_dropout	= (real)0.0;		// dropout disabled by default
	pnet->is_training		= 0;				// inference mode by default
	pnet->gpu_stale			= 0;
	pnet->l2_lambda			= (real)0.0;		// L2 regularization disabled
	pnet->l1_lambda			= (real)0.0;		// L1 regularization disabled
	
	// Training history
	pnet->loss_history		= NULL;
	pnet->lr_history		= NULL;
	pnet->history_count		= 0;
	pnet->history_capacity	= 0;
	pnet->val_loss_history	= NULL;

	// Validation and early stopping (disabled by default)
	pnet->val_inputs		= NULL;
	pnet->val_outputs		= NULL;
	pnet->es_patience		= 0;
	pnet->es_min_delta		= (real)0.0;
	pnet->es_restore_best	= 0;
	pnet->best_epoch		= 0;

	switch(opt)
	{
	case OPT_MOMENTUM:
		pnet->optimize_func = optimize_momentum;
		pnet->learning_rate = (real)0.01;
		break;

	case OPT_RMSPROP:
		pnet->optimize_func = optimize_rmsprop;
		pnet->learning_rate = (real)0.001;
		break;

	case OPT_ADAGRAD:
		pnet->optimize_func = optimize_adagrad;
		pnet->learning_rate = (real)0.01;
		break;

	case OPT_SGD:
		pnet->optimize_func = optimize_sgd;
		break;

	default:
	case OPT_ADAM:
		pnet->optimize_func = optimize_adam;
		pnet->learning_rate = (real)0.001;
		break;
	}

	return pnet;
}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
//-----------------------------------------------
// Cross-entropy needs probabilities from the output layer: softmax (one class
// per sample) or sigmoid (binary cross-entropy per output). Other output
// activations have no meaningful log-loss, so training is refused.
// Returns 1 when the network can be trained; otherwise reports ERR_INVALID.
//-----------------------------------------------
static int check_loss_output(PNetwork pnet, const char *function_name)
{
	if (pnet->loss_type != LOSS_CATEGORICAL_CROSS_ENTROPY || pnet->layer_count < 1)
		return 1;

	Activation_type act = pnet->layers[pnet->layer_count - 1].activation;
	if (act == ACTIVATION_SOFTMAX || act == ACTIVATION_SIGMOID)
		return 1;

	invoke_error_callback(ERR_INVALID, function_name);
	return 0;
}

//-----------------------------------------------
// Zero every layer's parameter gradients before a batch
//-----------------------------------------------
static void zero_gradients(PNetwork pnet)
{
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
			tensor_fill(cur->params[p].grad, (real)0.0);
	}
}

//-----------------------------------------------
// One CPU training step on the batch already in layers[0].t_batch_values:
// forward, backward, then the optimizer update. Returns the batch loss.
//-----------------------------------------------
static real train_batch_cpu(PNetwork pnet, int batch_size, PTensor targets)
{
	eval_network_batched(pnet, batch_size);

	// Batched backward pass (also computes loss)
	real loss = back_propagate_batched(pnet, batch_size, targets);

	// Increment training iteration for Adam bias correction
	pnet->train_iteration++;

	// update weights based on batched gradients using the chosen optimizer
	pnet->optimize_func(pnet);

	return loss;
}

//-----------------------------------------------
// Copy every layer's parameters to or from one flat buffer, used to
// keep the best epoch's parameters for early stopping. Pass NULL as buf to
// get the number of reals needed.
//-----------------------------------------------
static size_t copy_parameters(PNetwork pnet, real *buf, int restore)
{
	size_t offset = 0;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		for (int p = 0; p < cur->param_count; p++)
		{
			PTensor t = cur->params[p].value;
			size_t count = (size_t)t->rows * t->cols;
			if (buf && restore)
				memcpy(t->values, buf + offset, count * sizeof(real));
			else if (buf)
				memcpy(buf + offset, t->values, count * sizeof(real));
			offset += count;
		}
	}

	return offset;
}

real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, int rows)
{
	if (!pnet)
		return 0.0;

	if (!inputs || !outputs)
		return 0.0;

	if (rows <= 0)
		return 0.0;

	if (inputs->rows != rows)
		return 0.0;

	if (outputs->rows != rows)
		return 0.0;

	// the data's columns must match the input and output layers
	if (pnet->layer_count < 2 || inputs->cols != pnet->layers[0].node_count ||
		outputs->cols != pnet->layers[pnet->layer_count - 1].node_count)
	{
		invoke_error_callback(ERR_INVALID, "ann_train_network");
		return 0.0;
	}

	if (!check_loss_output(pnet, "ann_train_network"))
		return 0.0;

	if (pnet->val_inputs &&
		(pnet->val_inputs->cols != pnet->layers[0].node_count ||
		 pnet->val_outputs->cols != pnet->layers[pnet->layer_count - 1].node_count))
	{
		invoke_error_callback(ERR_INVALID, "ann_train_network");
		return 0.0;
	}

	ann_printf(pnet,	"\n%sTraining ANN%s\n"
						"------------\n", COLOR(ANSI_BOLD_CYAN), RESET());
	ann_print_props(pnet);
	ann_printf(pnet, "%s  Training size:%s %s%u%s rows\n\n", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_WHITE), rows, RESET());

	time_t time_start = time(NULL);

	pnet->train_iteration = 0;
	
	// Enable training mode (for dropout)
	pnet->is_training = 1;
	
	// Save base learning rate for schedulers
	if (pnet->base_learning_rate == (real)0.0)
		pnet->base_learning_rate = pnet->learning_rate;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real loss = (real)0.0;
	unsigned epoch = 0;

	// create indices for shuffling the inputs and outputs
	int *input_indices = alloca(rows * sizeof(int));
	for (int i = 0; i < rows; i++)
	{
		input_indices[i] = i;
	}

	int input_node_count = pnet->layers[0].node_count;
	int output_node_count = pnet->layers[pnet->layer_count - 1].node_count;

	// If batch size is larger than dataset, use dataset size as batch
	unsigned actual_batch_size = pnet->batchSize;
	if (actual_batch_size > (unsigned)rows)
		actual_batch_size = (unsigned)rows;
	
	int batch_count = rows / actual_batch_size;
	if (batch_count == 0)
		batch_count = 1;  // At least one batch

	// Ensure batch tensors are allocated for the batch size
	if (ensure_batch_tensors(pnet, actual_batch_size) != ERR_OK)
	{
		invoke_error_callback(ERR_ALLOC, "ann_train_network");
		return 0.0;
	}

	// Allocate batch target tensor for loss computation
	PTensor batch_targets = tensor_create(actual_batch_size, output_node_count);
	if (!batch_targets)
	{
		invoke_error_callback(ERR_ALLOC, "ann_train_network");
		return 0.0;
	}

	// Early stopping state; best_params holds the best epoch's weights
	real best_loss = (real)0.0;
	unsigned epochs_since_best = 0;
	real *best_params = NULL;
	pnet->best_epoch = 0;
	if (pnet->es_patience > 0 && pnet->es_restore_best)
	{
		best_params = malloc(copy_parameters(pnet, NULL, 0) * sizeof(real));
		if (!best_params)
		{
			tensor_free(batch_targets);
			invoke_error_callback(ERR_ALLOC, "ann_train_network");
			return 0.0;
		}
	}

#ifdef USE_GPU
	// Check if GPU training is available (backend active + weights uploaded to GPU)
	int use_gpu_training = gpu_uploaded(pnet);
	// init_weights() may have just randomized CPU weights; re-upload once so GPU
	// training always starts from the same initialized parameters as CPU training.
	if (use_gpu_training && g_gpu_backend->upload_network(pnet) != ERR_OK)
		use_gpu_training = 0;
	// Training on the CPU while uploaded leaves the GPU copy behind until the end
	pnet->gpu_stale = gpu_uploaded(pnet) && !use_gpu_training;
#endif
	while (!converged)
	{
		// re-shuffle the indices for this epoch
		shuffle_indices(input_indices, rows);
		
		// Apply learning rate scheduler if set
		++epoch;
		if (pnet->lr_scheduler)
		{
			pnet->learning_rate = pnet->lr_scheduler(epoch, pnet->base_learning_rate, pnet->lr_scheduler_data);
		}
		
		// iterate over all sets of inputs in this epoch/minibatch
		loss = (real)0.0;
		real epoch_loss = (real)0.0;	// sum of batch losses; the epoch reports their mean

		// Progress bar buffer (avoid per-batch console I/O)
		char progress[64];
		int progress_len = 0;
		int progress_step = max(1, batch_count / 20);

		// iterate over all batches
		for (int batch = 0; batch < batch_count; batch++)
		{
			zero_gradients(pnet);

			// Assemble batch input matrix into layers[0].t_batch_values
			PTensor batch_input = pnet->layers[0].t_batch_values;
			for (unsigned b = 0; b < actual_batch_size; b++)
			{
				int row = batch * actual_batch_size + b;
				int input_offset = input_indices[row] * input_node_count;
				int output_offset = input_indices[row] * output_node_count;

				// Copy input row to batch matrix
				memcpy(batch_input->values + b * input_node_count,
				       inputs->values + input_offset,
				       input_node_count * sizeof(real));

				// Copy target row to batch targets
				memcpy(batch_targets->values + b * output_node_count,
				       outputs->values + output_offset,
				       output_node_count * sizeof(real));
			}

		// Batched forward + backward + optimize (CPU or GPU path)
#ifdef USE_GPU
			// GPU handles forward + backward + optimizer + train_iteration++
			int gpu_done = 0;
			if (use_gpu_training)
			{
				make_dropout_masks(pnet, actual_batch_size);
				gpu_done = g_gpu_backend->train_batch(pnet, batch_targets, actual_batch_size, &loss);
				if (!gpu_done)
				{
					// GPU failed: finish on the CPU, starting from the GPU's latest weights
					g_gpu_backend->sync_weights(pnet);
					use_gpu_training = 0;
					pnet->gpu_stale = 1;
				}
			}
			if (!gpu_done)
#endif
			{
				loss = train_batch_cpu(pnet, actual_batch_size, batch_targets);
			}

			epoch_loss += loss;

			// Progress indicator: buffer instead of per-batch I/O
			if (batch % progress_step == 0 && progress_len < 60)
				progress[progress_len++] = '=';
		}
		progress[progress_len] = '\0';
		loss = epoch_loss / (real)batch_count;

		// Validation loss (-1 when there is no validation set). ann_predict_batch
		// uses the GPU weights while GPU training keeps them current.
		real val_loss = (real)-1.0;
		if (pnet->val_inputs)
			val_loss = ann_evaluate_loss(pnet, pnet->val_inputs, pnet->val_outputs);

		char val_text[64] = "";
		if (val_loss >= (real)0.0)
			sprintf(val_text, " - val_loss: %s%3.2g%s", COLOR(ANSI_YELLOW), val_loss, RESET());

		ann_printf(pnet, "%sEpoch %u/%u%s\n[%s%s%s] - loss: %s%3.2g%s%s - LR: %s%3.2g%s\n",
			COLOR(ANSI_BOLD_WHITE), epoch, pnet->epochLimit, RESET(),
			COLOR(ANSI_GREEN), progress, RESET(),
			COLOR(ANSI_YELLOW), loss, RESET(), val_text,
			COLOR(ANSI_BLUE), pnet->learning_rate, RESET());
		
		// Record training history for learning curve
		record_history(pnet, loss, pnet->learning_rate, val_loss);

		// Early stopping monitors the validation loss, or the training loss without one
		if (pnet->es_patience > 0)
		{
			real monitored = val_loss >= (real)0.0 ? val_loss : loss;
			if (pnet->best_epoch == 0 || monitored < best_loss - pnet->es_min_delta)
			{
				best_loss = monitored;
				pnet->best_epoch = epoch;
				epochs_since_best = 0;

				if (best_params)
				{
#ifdef USE_GPU
					// GPU training updates only the GPU copy until it is synced
					if (use_gpu_training)
						g_gpu_backend->sync_weights(pnet);
#endif
					copy_parameters(pnet, best_params, 0);
				}
			}
			else if (++epochs_since_best >= pnet->es_patience)
			{
				ann_printf(pnet, "%sEarly stopping%s: no improvement for %u epochs, best epoch %u\n",
					COLOR(ANSI_BOLD_GREEN), RESET(), pnet->es_patience, pnet->best_epoch);
				converged = 1;
			}
		}

		if (loss < pnet->convergence_epsilon)
		{
			ann_printf(pnet, "%sNetwork converged%s with loss: %s%3.2g%s out of %3.2g\n", 
				COLOR(ANSI_BOLD_GREEN), RESET(),
				COLOR(ANSI_GREEN), loss, RESET(), 
				pnet->convergence_epsilon);
			converged = 1;
		}

		// check for no convergence
		if (epoch >= pnet->epochLimit)
		{
			converged = 1;
		}
	}

	// free up batch tensors
	tensor_free(batch_targets);

#ifdef USE_GPU
	// Leave the CPU and GPU copies of the weights identical
	if (use_gpu_training)
		g_gpu_backend->sync_weights(pnet);
	else if (pnet->gpu_stale)
		gpu_refresh(pnet);
#endif

	// Go back to the best epoch's weights if later epochs were worse
	if (best_params && pnet->best_epoch != epoch)
	{
		copy_parameters(pnet, best_params, 1);
		ann_printf(pnet, "Restored weights from epoch %u\n", pnet->best_epoch);
#ifdef USE_GPU
		gpu_refresh(pnet);
#endif
	}
	free(best_params);
	
	// Disable training mode (for dropout)
	pnet->is_training = 0;

	time_t time_end = time(NULL);
	double diff_t = (double)(time_end - time_start);
	double per_step = 1000.0 * diff_t / (rows * epoch);

	ann_printf(pnet, "\n%sTraining time:%s %s%.1f%s seconds, %s%.3f%s ms/step\n", 
		COLOR(ANSI_BOLD_WHITE), RESET(),
		COLOR(ANSI_GREEN), diff_t, RESET(),
		COLOR(ANSI_GREEN), per_step, RESET());

	return loss;
}

//-----------------------------------------------
// Begin an online/incremental training session
//-----------------------------------------------
int ann_train_begin(PNetwork pnet)
{
	if (!pnet)
		return ERR_NULL_PTR;

	if (pnet->layer_count <= 0 || !pnet->layers)
		return ERR_INVALID;

	if (!check_loss_output(pnet, "ann_train_begin"))
		return ERR_INVALID;

	// Enable training mode (for dropout)
	pnet->is_training = 1;

	// Save base learning rate for schedulers
	if (pnet->base_learning_rate == (real)0.0)
		pnet->base_learning_rate = pnet->learning_rate;

	// Initialize weights only if not already set (e.g. loaded model)
	init_weights(pnet);

	// Ensure batch tensors are allocated for the configured batch size
	if (ensure_batch_tensors(pnet, pnet->batchSize) != ERR_OK)
	{
		invoke_error_callback(ERR_ALLOC, "ann_train_begin");
		return ERR_ALLOC;
	}

#ifdef USE_GPU
	// Online steps run on the CPU; predictions use the CPU until ann_train_end()
	// re-uploads the weights
	if (gpu_uploaded(pnet))
		pnet->gpu_stale = 1;
#endif

	return ERR_OK;
}

//-----------------------------------------------
// Train one mini-batch step (online training)
//-----------------------------------------------
real ann_train_step(PNetwork pnet, const real *inputs, const real *targets, int batch_size)
{
	if (!pnet || !inputs || !targets)
		return (real)0.0;

	if (batch_size <= 0)
		return (real)0.0;

	if (!check_loss_output(pnet, "ann_train_step"))
		return (real)0.0;

	int input_node_count = pnet->layers[0].node_count;
	int output_node_count = pnet->layers[pnet->layer_count - 1].node_count;

	unsigned actual_batch_size = (unsigned)batch_size;

	// Reallocate batch tensors if batch size changed
	if (pnet->current_batch_size != actual_batch_size)
	{
		if (ensure_batch_tensors(pnet, actual_batch_size) != ERR_OK)
		{
			invoke_error_callback(ERR_ALLOC, "ann_train_step");
			return (real)0.0;
		}
	}

	// Allocate temporary batch target tensor
	PTensor batch_targets = tensor_create(actual_batch_size, output_node_count);
	if (!batch_targets)
	{
		invoke_error_callback(ERR_ALLOC, "ann_train_step");
		return (real)0.0;
	}

	zero_gradients(pnet);

	// Copy inputs into batch input tensor
	PTensor batch_input = pnet->layers[0].t_batch_values;
	memcpy(batch_input->values, inputs, actual_batch_size * input_node_count * sizeof(real));

	// Copy targets into batch target tensor
	memcpy(batch_targets->values, targets, actual_batch_size * output_node_count * sizeof(real));

	real loss = train_batch_cpu(pnet, actual_batch_size, batch_targets);

#ifdef USE_GPU
	if (gpu_uploaded(pnet))
		pnet->gpu_stale = 1;
#endif

	tensor_free(batch_targets);

	return loss;
}

//-----------------------------------------------
// End an online/incremental training session
//-----------------------------------------------
void ann_train_end(PNetwork pnet)
{
	if (!pnet)
		return;

	// Disable training mode (for dropout)
	pnet->is_training = 0;

#ifdef USE_GPU
	if (pnet->gpu_stale)
		gpu_refresh(pnet);
#endif
}

//------------------------------
// evaluate the accuracy 
//------------------------------
real ann_evaluate_accuracy(const PNetwork pnet, const PTensor inputs, const PTensor outputs)
{
	int correct = 0;

	if (!pnet || !inputs || !outputs)
	{
		return -1.0;
	}

	if (inputs->rows <= 0 || inputs->cols <= 0 || outputs->rows <= 0 || outputs->cols <= 0)
	{
		return -1.0;
	}

	if (inputs->rows != outputs->rows)
	{
		return -1.0;
	}

	int classes = outputs->cols;
	if (pnet->layer_count < 2 || inputs->cols != pnet->layers[0].node_count ||
		classes != pnet->layers[pnet->layer_count - 1].node_count)
	{
		return -1.0;
	}

	// Predict in chunks so a GPU backend can evaluate many rows per dispatch
	enum { EVAL_CHUNK = 256 };
	real *pred = malloc(EVAL_CHUNK * classes * sizeof(real));
	if (!pred)
	{
		invoke_error_callback(ERR_ALLOC, "ann_evaluate_accuracy");
		return -1.0;
	}

	for (int start = 0; start < inputs->rows; start += EVAL_CHUNK)
	{
		int count = inputs->rows - start < EVAL_CHUNK ? inputs->rows - start : EVAL_CHUNK;
		if (ann_predict_batch(pnet, &inputs->values[start * inputs->cols], pred, count) != ERR_OK)
		{
			free(pred);
			return -1.0;
		}

		for (int i = 0; i < count; i++)
		{
			int pred_class = ann_class_prediction(pred + i * classes, classes);
			int act_class = ann_class_prediction(&outputs->values[(start + i) * classes], classes);
			if (pred_class == act_class)
				correct++;
		}
	}
	free(pred);

	return (real)correct / inputs->rows;
}

//------------------------------------
// Average loss over a dataset
//------------------------------------
real ann_evaluate_loss(const PNetwork pnet, const PTensor inputs, const PTensor outputs)
{
	if (!pnet || !inputs || !outputs || pnet->layer_count < 2)
		return (real)-1.0;

	int out_nodes = pnet->layers[pnet->layer_count - 1].node_count;
	if (inputs->rows <= 0 || inputs->rows != outputs->rows ||
		inputs->cols != pnet->layers[0].node_count || outputs->cols != out_nodes)
	{
		return (real)-1.0;
	}

	// Predict in chunks so a GPU backend can evaluate many rows per dispatch
	enum { EVAL_CHUNK = 256 };
	real *pred = malloc(EVAL_CHUNK * out_nodes * sizeof(real));
	if (!pred)
	{
		invoke_error_callback(ERR_ALLOC, "ann_evaluate_loss");
		return (real)-1.0;
	}

	real total = (real)0.0;
	for (int start = 0; start < inputs->rows; start += EVAL_CHUNK)
	{
		int count = inputs->rows - start < EVAL_CHUNK ? inputs->rows - start : EVAL_CHUNK;
		if (ann_predict_batch(pnet, &inputs->values[start * inputs->cols], pred, count) != ERR_OK)
		{
			free(pred);
			return (real)-1.0;
		}

		for (int i = 0; i < count; i++)
			total += sample_loss(pnet, &outputs->values[(start + i) * out_nodes], pred + i * out_nodes, out_nodes);
	}
	free(pred);

	if (pnet->loss_type == LOSS_MSE)
		return total / (real)(inputs->rows * out_nodes);
	return total / (real)inputs->rows;
}

//------------------------------------
// Compute binary confusion matrix and MCC
//------------------------------------
real ann_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs,
                          int *tp, int *fp, int *tn, int *fn)
{
	if (!pnet || !inputs || !outputs)
		return (real)0.0;
	
	if (inputs->rows != outputs->rows || outputs->cols != 2)
		return (real)0.0;

	if (pnet->layer_count < 2 || inputs->cols != pnet->layers[0].node_count ||
		pnet->layers[pnet->layer_count - 1].node_count != 2)
		return (real)0.0;
	
	int true_pos = 0, false_pos = 0, true_neg = 0, false_neg = 0;

	// Predict in chunks, as ann_evaluate_accuracy() does
	enum { EVAL_CHUNK = 256 };
	real pred[EVAL_CHUNK * 2];

	for (int start = 0; start < inputs->rows; start += EVAL_CHUNK)
	{
		int count = inputs->rows - start < EVAL_CHUNK ? inputs->rows - start : EVAL_CHUNK;
		if (ann_predict_batch(pnet, &inputs->values[start * inputs->cols], pred, count) != ERR_OK)
			return (real)0.0;

		for (int i = 0; i < count; i++)
		{
			int row = start + i;
			int pred_class = pred[i * 2 + 1] > pred[i * 2] ? 1 : 0;
			int actual_class = outputs->values[row * 2 + 1] > outputs->values[row * 2] ? 1 : 0;
			
			if (actual_class == 1 && pred_class == 1)
				true_pos++;
			else if (actual_class == 0 && pred_class == 1)
				false_pos++;
			else if (actual_class == 0 && pred_class == 0)
				true_neg++;
			else
				false_neg++;
		}
	}
	
	if (tp) *tp = true_pos;
	if (fp) *fp = false_pos;
	if (tn) *tn = true_neg;
	if (fn) *fn = false_neg;
	
	// MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
	double num = (double)true_pos * true_neg - (double)false_pos * false_neg;
	double denom = sqrt(
		((double)true_pos + false_pos) *
		((double)true_pos + false_neg) *
		((double)true_neg + false_pos) *
		((double)true_neg + false_neg)
	);
	
	if (denom == 0.0)
		return (real)0.0;
	
	return (real)(num / denom);
}

//------------------------------------
// Print formatted confusion matrix with MCC
//------------------------------------
void ann_print_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	int tp = 0, fp = 0, tn = 0, fn = 0;
	real mcc = ann_confusion_matrix(pnet, inputs, outputs, &tp, &fp, &tn, &fn);
	
	ann_printf(pnet, "\n%sConfusion Matrix%s\n", COLOR(ANSI_BOLD_CYAN), RESET());
	ann_printf(pnet, "                %sPredicted%s\n", COLOR(ANSI_DIM), RESET());
	ann_printf(pnet, "              %sPos     Neg%s\n", COLOR(ANSI_WHITE), RESET());
	ann_printf(pnet, "%sActual Pos%s  %s%5d   %5d%s\n", 
		COLOR(ANSI_DIM), RESET(), COLOR(ANSI_GREEN), tp, fn, RESET());
	ann_printf(pnet, "%s       Neg%s  %s%5d   %5d%s\n", 
		COLOR(ANSI_DIM), RESET(), COLOR(ANSI_GREEN), fp, tn, RESET());
	ann_printf(pnet, "\n%sMCC:%s %s%.4f%s\n", 
		COLOR(ANSI_BOLD_WHITE), RESET(), COLOR(ANSI_CYAN), mcc, RESET());
}

//------------------------------------
// predict class from onehot vector
//------------------------------------
int ann_class_prediction(const real *outputs, int classes)
{
	int class = -1;
	real prob = -1.0;

	for (int i = 0; i < classes; i++)
	{
		if (outputs[i] > prob)
		{
			prob = outputs[i];
			class = i;
		}
	}

	return class;
}

//------------------------------
// set the network learning rate
//------------------------------
void ann_set_learning_rate(PNetwork pnet, real rate)
{
	if (!pnet)
		return;

	pnet->learning_rate = rate;
}

//------------------------------
// set the convergence limit
//------------------------------
void ann_set_convergence(PNetwork pnet, real limit)
{
	if (!pnet)
		return;

	pnet->convergence_epsilon = limit;
}

//------------------------------
// set gradient clipping threshold
//------------------------------
void ann_set_gradient_clip(PNetwork pnet, real max_grad)
{
	if (!pnet)
		return;

	pnet->max_gradient = max_grad;
}

//------------------------------
// set L2 regularization (weight decay)
//------------------------------
void ann_set_weight_decay(PNetwork pnet, real lambda)
{
	if (!pnet)
		return;

	pnet->l2_lambda = lambda;
}

//------------------------------
// set L1 regularization (LASSO)
//------------------------------
void ann_set_l1_regularization(PNetwork pnet, real lambda)
{
	if (!pnet)
		return;

	pnet->l1_lambda = lambda;
}

//------------------------------
// set weight initialization strategy
//------------------------------
void ann_set_weight_init(PNetwork pnet, Weight_init_type init_type)
{
	if (!pnet)
		return;

	pnet->weight_init = init_type;
}

//------------------------------
// set the mini-batch size
//------------------------------
void ann_set_batch_size(PNetwork pnet, unsigned batch_size)
{
	if (!pnet || batch_size == 0)
		return;

	pnet->batchSize = batch_size;
}

//------------------------------
// set the epoch limit
//------------------------------
void ann_set_epoch_limit(PNetwork pnet, unsigned limit)
{
	if (!pnet)
		return;

	pnet->epochLimit = limit;
}

//------------------------------
// set the validation dataset
//------------------------------
int ann_set_validation_data(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	if (!pnet || (!inputs != !outputs))
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_set_validation_data");
		return ERR_NULL_PTR;
	}

	if (inputs && (inputs->rows <= 0 || inputs->rows != outputs->rows))
	{
		invoke_error_callback(ERR_INVALID, "ann_set_validation_data");
		return ERR_INVALID;
	}

	pnet->val_inputs = inputs;
	pnet->val_outputs = outputs;
	return ERR_OK;
}

//------------------------------
// configure early stopping
//------------------------------
void ann_set_early_stopping(PNetwork pnet, unsigned patience, real min_delta, int restore_best)
{
	if (!pnet)
		return;

	pnet->es_patience = patience;
	pnet->es_min_delta = min_delta < (real)0.0 ? (real)0.0 : min_delta;
	pnet->es_restore_best = restore_best;
}

//------------------------------
// best epoch of the last training run
//------------------------------
unsigned ann_get_best_epoch(const PNetwork pnet)
{
	return pnet ? pnet->best_epoch : 0;
}

//------------------------------
// set learning rate scheduler
//------------------------------
void ann_set_lr_scheduler(PNetwork pnet, LRSchedulerFunc scheduler, void *user_data)
{
	if (!pnet)
		return;

	pnet->lr_scheduler = scheduler;
	pnet->lr_scheduler_data = user_data;
}

//------------------------------
// Step decay LR scheduler
// LR = base_lr * (gamma ^ (epoch / step_size))
//------------------------------
real ann_lr_scheduler_step(unsigned epoch, real base_lr, void *user_data)
{
	if (!user_data)
		return base_lr;

	LRStepParams *params = (LRStepParams *)user_data;
	if (params->step_size == 0)
		return base_lr;
	unsigned steps = (epoch - 1) / params->step_size;
	real multiplier = (real)1.0;

	for (unsigned i = 0; i < steps; i++)
		multiplier *= params->gamma;

	return base_lr * multiplier;
}

//------------------------------
// Exponential decay LR scheduler
// LR = base_lr * (gamma ^ epoch)
//------------------------------
real ann_lr_scheduler_exponential(unsigned epoch, real base_lr, void *user_data)
{
	if (!user_data)
		return base_lr;

	LRExponentialParams *params = (LRExponentialParams *)user_data;
	return base_lr * (real)pow(params->gamma, (double)(epoch - 1));
}

//------------------------------
// Cosine annealing LR scheduler
// LR = min_lr + (base_lr - min_lr) * (1 + cos(pi * epoch / T_max)) / 2
//------------------------------
real ann_lr_scheduler_cosine(unsigned epoch, real base_lr, void *user_data)
{
	if (!user_data)
		return base_lr;

	LRCosineParams *params = (LRCosineParams *)user_data;
	if (params->T_max == 0)
		return base_lr;
	
	// Clamp epoch to T_max
	unsigned t = (epoch > params->T_max) ? params->T_max : epoch;
	
	real cos_val = (real)cos(3.14159265358979323846 * (double)t / (double)params->T_max);
	return params->min_lr + (base_lr - params->min_lr) * ((real)1.0 + cos_val) / (real)2.0;
}

//------------------------------
// set default dropout rate for hidden layers
//------------------------------
void ann_set_dropout(PNetwork pnet, real rate)
{
	if (!pnet)
		return;
	
	// Clamp rate to valid range [0, 1)
	if (rate < (real)0.0)
		rate = (real)0.0;
	if (rate >= (real)1.0)
		rate = (real)0.99;
	
	pnet->default_dropout = rate;
	
	// Apply to all existing hidden dense and conv layers that don't have a
	// custom rate (pooling and flatten layers only move values around)
	for (int i = 1; i < pnet->layer_count - 1; i++)
	{
		Layer_kind kind = pnet->layers[i].kind;
		if ((kind == LAYER_KIND_DENSE || kind == LAYER_KIND_CONV2D) && pnet->layers[i].dropout_rate == (real)0.0)
			pnet->layers[i].dropout_rate = rate;
	}
}

//------------------------------
// set dropout rate for a specific layer
//------------------------------
void ann_set_layer_dropout(PNetwork pnet, int layer, real rate)
{
	if (!pnet || layer < 0 || layer >= pnet->layer_count)
		return;
	
	// Don't allow dropout on input or output layers
	if (pnet->layers[layer].layer_type == LAYER_INPUT ||
	    pnet->layers[layer].layer_type == LAYER_OUTPUT)
		return;
	
	// Clamp rate to valid range [0, 1)
	if (rate < (real)0.0)
		rate = (real)0.0;
	if (rate >= (real)1.0)
		rate = (real)0.99;
	
	pnet->layers[layer].dropout_rate = rate;
}

//------------------------------
// set training/inference mode
//------------------------------
void ann_set_training_mode(PNetwork pnet, int is_training)
{
	if (!pnet)
		return;
	
	pnet->is_training = is_training ? 1 : 0;
}

//------------------------------
// get the number of layers
//------------------------------
int ann_get_layer_count(const PNetwork pnet)
{
	if (!pnet)
		return -1;

	return pnet->layer_count;
}

//------------------------------
// get the number of nodes in a layer
//------------------------------
int ann_get_layer_nodes(const PNetwork pnet, int layer)
{
	if (!pnet || layer < 0 || layer >= pnet->layer_count)
		return -1;

	return pnet->layers[layer].node_count;
}

//------------------------------
// get the activation type of a layer
//------------------------------
Activation_type ann_get_layer_activation(const PNetwork pnet, int layer)
{
	if (!pnet || layer < 0 || layer >= pnet->layer_count)
		return ACTIVATION_NULL;

	return pnet->layers[layer].activation;
}

//------------------------------
// set or get the CPU thread count
//------------------------------
void ann_set_num_threads(int threads)
{
	ann_parallel_set_threads(threads < 0 ? 0 : threads);
}

int ann_get_num_threads(void)
{
	return ann_parallel_get_threads();
}

//------------------------------
// set or get where a network's progress output goes
//------------------------------
void ann_set_print_function(PNetwork pnet, Output_func print_func)
{
	if (pnet)
		pnet->print_func = print_func;
}

Output_func ann_get_print_function(const PNetwork pnet)
{
	return pnet ? pnet->print_func : NULL;
}

//------------------------------
// get what a layer computes
//------------------------------
Layer_kind ann_get_layer_kind(const PNetwork pnet, int layer)
{
	if (!pnet || layer < 0 || layer >= pnet->layer_count)
		return LAYER_KIND_DENSE;

	return pnet->layers[layer].kind;
}

//------------------------------
// get a layer's output shape
//------------------------------
int ann_get_layer_shape(const PNetwork pnet, int layer, int *channels, int *height, int *width)
{
	if (!pnet || !channels || !height || !width)
		return ERR_NULL_PTR;
	if (layer < 0 || layer >= pnet->layer_count)
		return ERR_INVALID;

	*channels = pnet->layers[layer].shape.c;
	*height = pnet->layers[layer].shape.h;
	*width = pnet->layers[layer].shape.w;
	return ERR_OK;
}

//------------------------------
// get a conv or pooling layer's window settings
//------------------------------
int ann_get_layer_settings(const PNetwork pnet, int layer, int *size, int *stride, int *padding)
{
	if (!pnet || !size || !stride || !padding)
		return ERR_NULL_PTR;
	if (layer < 0 || layer >= pnet->layer_count)
		return ERR_INVALID;

	const Layer *cur = &pnet->layers[layer];
	*size = *stride = *padding = 0;
	if (cur->kind == LAYER_KIND_CONV2D)
	{
		*size = cur->config[CONV_KERNEL];
		*stride = cur->config[CONV_STRIDE];
		*padding = cur->config[CONV_PADDING];
	}
	else if (cur->kind == LAYER_KIND_MAXPOOL2D)
	{
		*size = cur->config[POOL_SIZE];
		*stride = cur->config[POOL_STRIDE];
	}
	return ERR_OK;
}

//------------------------------
// set the loss function
//------------------------------
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type)
{
	if (!pnet)
		return;

	// The loss is computed during batched backprop, which reads loss_type
	pnet->loss_type = loss_type;
}

//------------------------------
// free a network
//------------------------------
void ann_free_network(PNetwork pnet)
{
	if (!pnet)
		return;

	// free layers
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// Add NULL check before freeing t_values
		if (pnet->layers[layer].t_values)
			tensor_free(pnet->layers[layer].t_values);

		free_layer_params(&pnet->layers[layer]);

		// Free batch tensors
		tensor_free(pnet->layers[layer].t_batch_values);
		tensor_free(pnet->layers[layer].t_batch_dl_dz);
		tensor_free(pnet->layers[layer].t_batch_z);
		
		// Free dropout mask
		tensor_free(pnet->layers[layer].t_dropout_mask);
	}

	free(pnet->layers);
	
	// Free training history
	free(pnet->loss_history);
	free(pnet->lr_history);
	free(pnet->val_loss_history);

	// free network
	free(pnet);
}

//------------------------------
// load data from a csv file
//------------------------------
int ann_load_csv(const char *filename, int has_header, real **data, int *rows, int *stride)
{
	FILE *f;
	char *s, buf[DEFAULT_BUFFER_SIZE];
	int size = DEFAULT_BUFFER_SIZE;
	real *dbuf;
	int lastStride = 0;
	uint32_t lineno = 0;
	int count = 0;

	if (!filename || !data || !rows || !stride)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_load_csv");
		return ERR_NULL_PTR;
	}

	f = fopen(filename, "rt");
	if (!f)
	{
		invoke_error_callback(ERR_IO, "ann_load_csv");
		return ERR_IO;
	}

	*rows = 0;

	dbuf = malloc(size * sizeof(real));

	// skip header if present
	if (has_header && !fgets(buf, DEFAULT_BUFFER_SIZE, f))
	{
		invoke_error_callback(ERR_FAIL, "ann_load_csv");
		return ERR_FAIL;
	}

	// read a line
	while (fgets(buf, DEFAULT_BUFFER_SIZE, f))
	{
		if (buf[0] == 0)
			continue;

		lineno++;

		// tokenize the line
		*stride = 0;
		s = strtok(buf, ", \n");

		// parse the line
		while (s) {
			dbuf[count++] = (real)atof(s);
			(*stride)++;

			if (count >= size)
			{
				// double the size
				size <<= 1;

				real *newbuf = realloc(dbuf, size * sizeof(real));
				
				// check for OOM
				if (!dbuf)
				{
					free(dbuf);
					invoke_error_callback(ERR_FAIL, "ann_load_csv");
					return ERR_FAIL;
				}

				dbuf = newbuf;
			}

			s = strtok(NULL, ", \n");
		}

		if (lastStride == 0)
			lastStride = *stride;

		// check that all row strides are the same
		if (lastStride != *stride)
		{
			printf("Error: malformed CSV file at line %u\n", lineno);
			free(dbuf);
			fclose(f);
			invoke_error_callback(ERR_FAIL, "ann_load_csv");
			return ERR_FAIL;
		}

		(*rows)++;

	}

	*data = dbuf;

	fclose(f);
	return ERR_OK;
}

//-----------------------------------
// predict an outcome from trained nn
//-----------------------------------
int ann_predict(const PNetwork pnet, const real *inputs, real *outputs)
{
	if (!pnet || !inputs || !outputs)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_predict");
		return ERR_NULL_PTR;
	}

	if (pnet->layer_count <= 0 || !pnet->layers)
	{
		invoke_error_callback(ERR_INVALID, "ann_predict");
		return ERR_INVALID;
	}

	// set inputs
	int node_count = pnet->layers[0].node_count;
	for (int node = 0; node < node_count; node++)
	{
		pnet->layers[0].t_values->values[node] = *inputs++;
	}

	// Temporarily disable training mode for inference (prevents dropout)
	int was_training = pnet->is_training;
	pnet->is_training = 0;

	// evaluate network
	eval_network(pnet);

	// Restore training mode
	pnet->is_training = was_training;

	// get the outputs
	node_count = pnet->layers[pnet->layer_count - 1].node_count;
	for (int node = 0; node < node_count; node++)
	{
		*outputs++ = pnet->layers[pnet->layer_count - 1].t_values->values[node];
	}

	return ERR_OK;
}

//-----------------------------------------------------------
// Set the row count of every layer's forward batch tensors (t_batch_values,
// t_batch_z). A count below the allocated size runs a short chunk through
// eval_network_batched() without reallocating.
//-----------------------------------------------------------
static void set_batch_rows(PNetwork pnet, int rows)
{
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		pnet->layers[layer].t_batch_values->rows = rows;
		if (pnet->layers[layer].t_batch_z)
			pnet->layers[layer].t_batch_z->rows = rows;
	}
}

//-----------------------------------------------------------
// CPU batch inference: runs the batched forward pass over chunks of rows.
// The chunk is the allocated batch size. Outside training the batch tensors
// grow to PREDICT_CHUNK rows; during training they are left at the training
// batch size, since the training loop reuses them.
//-----------------------------------------------------------
#define PREDICT_CHUNK 256

static int predict_batch_cpu(PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
	int in_nodes  = pnet->layers[0].node_count;
	int out_nodes = pnet->layers[pnet->layer_count - 1].node_count;
	unsigned want = batch_size < PREDICT_CHUNK ? (unsigned)batch_size : PREDICT_CHUNK;

	if (pnet->current_batch_size < want && (!pnet->is_training || pnet->current_batch_size == 0))
	{
		int err = ensure_batch_tensors(pnet, want);
		if (err != ERR_OK)
			return err;
	}

	int chunk = (int)pnet->current_batch_size;
	PTensor X = pnet->layers[0].t_batch_values;
	PTensor Y = pnet->layers[pnet->layer_count - 1].t_batch_values;

	// Inference never applies dropout
	int was_training = pnet->is_training;
	pnet->is_training = 0;

	for (int start = 0; start < batch_size; start += chunk)
	{
		int count = batch_size - start < chunk ? batch_size - start : chunk;

		memcpy(X->values, inputs + (size_t)start * in_nodes, (size_t)count * in_nodes * sizeof(real));
		set_batch_rows(pnet, count);
		eval_network_batched(pnet, count);
		memcpy(outputs + (size_t)start * out_nodes, Y->values, (size_t)count * out_nodes * sizeof(real));
	}

	set_batch_rows(pnet, chunk);
	pnet->is_training = was_training;
	return ERR_OK;
}

//-----------------------------------------------------------
// GPU batch inference (ann_predict_batch)
//
// Runs forward pass on `batch_size` samples in parallel on the GPU.
// Otherwise runs the CPU batched forward pass in chunks.
//
// @param pnet       Network with weights (GPU-uploaded for acceleration)
// @param inputs     [batch_size x input_nodes] row-major flat array
// @param outputs    [batch_size x output_nodes] row-major flat array (written)
// @param batch_size Number of samples to process
// @return ERR_OK on success, error code otherwise
//-----------------------------------------------------------
int ann_predict_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
	if (!pnet || !inputs || !outputs || batch_size <= 0)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_predict_batch");
		return ERR_NULL_PTR;
	}

	if (pnet->layer_count <= 0 || !pnet->layers)
	{
		invoke_error_callback(ERR_INVALID, "ann_predict_batch");
		return ERR_INVALID;
	}

#ifdef USE_GPU
	// Use GPU batch inference when a backend is active and the uploaded weights are current
	if (gpu_ready(pnet))
		return g_gpu_backend->eval_batch(pnet, inputs, outputs, batch_size);
#endif

	return predict_batch_cpu(pnet, inputs, outputs, batch_size);
}

//================================================================================================
// Model files (.nna text, .nnb binary). Both hold the same values: text writes one per line,
// binary writes raw ints and reals.
//
// v1, written when every layer is dense so older libann versions can read it:
//   version optimizer loss layer_count
//   per layer:            node_count layer_type activation
//   per layer 1..n-1:     bias values, then weight values
// v2, written when any layer is another kind:
//   version optimizer loss layer_count
//   per layer:            kind layer_type activation c h w config_count config...
//   per layer 1..n-1:     each params[] tensor's values, in order
//================================================================================================

//------------------------------
// value I/O shared by the text and binary formats; each returns 1 on success
//------------------------------
static int write_int(FILE *fptr, int binary, int value)
{
	if (binary)
		return fwrite(&value, sizeof(value), 1, fptr) == 1;
	return fprintf(fptr, "%d\n", value) > 0;
}

static int write_real(FILE *fptr, int binary, real value)
{
	if (binary)
		return fwrite(&value, sizeof(value), 1, fptr) == 1;
	return fprintf(fptr, "%f\n", value) > 0;
}

static int write_values(FILE *fptr, int binary, const PTensor t)
{
	int count = t->rows * t->cols;
	for (int i = 0; i < count; i++)
	{
		if (!write_real(fptr, binary, t->values[i]))
			return 0;
	}
	return 1;
}

static int read_int(FILE *fptr, int binary, int *value)
{
	if (binary)
		return fread(value, sizeof(*value), 1, fptr) == 1;
	return fscanf(fptr, "%d", value) == 1;
}

static int read_values(FILE *fptr, int binary, PTensor t)
{
	int count = t->rows * t->cols;
	for (int i = 0; i < count; i++)
	{
		int ok = binary ? fread(&t->values[i], sizeof(real), 1, fptr) == 1
		                : fscanf(fptr, "%f", &t->values[i]) == 1;
		if (!ok)
			return 0;
	}
	return 1;
}

//------------------------------
// 1 if every layer is dense (the network fits the v1 file format)
//------------------------------
static int network_is_dense(const PNetwork pnet)
{
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		if (pnet->layers[layer].kind != LAYER_KIND_DENSE)
			return 0;
	}
	return 1;
}

//------------------------------
// 1 if the v1 format can hold the network: dense layers with flat shapes
//------------------------------
static int network_fits_v1(const PNetwork pnet)
{
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		if (pnet->layers[layer].shape.h != 1 || pnet->layers[layer].shape.w != 1)
			return 0;
	}
	return network_is_dense(pnet);
}

//------------------------------
// save a network in the text or binary format
//------------------------------
static int save_network(const PNetwork pnet, const char *filename, int binary, const char *function_name)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, function_name);
		return ERR_NULL_PTR;
	}

	FILE *fptr = fopen(filename, binary ? "wb" : "wt");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, function_name);
		return ERR_IO;
	}

	int v1 = network_fits_v1(pnet);

	// header
	int ok = write_int(fptr, binary, v1 ? ANN_FORMAT_V1 : ANN_FORMAT_V2) &&
	         write_int(fptr, binary, pnet->optimizer) &&
	         write_int(fptr, binary, pnet->loss_type) &&
	         write_int(fptr, binary, pnet->layer_count);

	// layer details
	for (int layer = 0; ok && layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		if (v1)
		{
			ok = write_int(fptr, binary, cur->node_count) &&
			     write_int(fptr, binary, cur->layer_type) &&
			     write_int(fptr, binary, cur->activation);
			continue;
		}

		int config_count = layer_ops(cur)->config_count;
		ok = write_int(fptr, binary, cur->kind) &&
		     write_int(fptr, binary, cur->layer_type) &&
		     write_int(fptr, binary, cur->activation) &&
		     write_int(fptr, binary, cur->shape.c) &&
		     write_int(fptr, binary, cur->shape.h) &&
		     write_int(fptr, binary, cur->shape.w) &&
		     write_int(fptr, binary, config_count);
		for (int i = 0; ok && i < config_count; i++)
			ok = write_int(fptr, binary, cur->config[i]);
	}

	// parameters
	for (int layer = 1; ok && layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		if (v1)
		{
			ok = write_values(fptr, binary, cur->t_bias) && write_values(fptr, binary, cur->t_weights);
			continue;
		}

		for (int p = 0; ok && p < cur->param_count; p++)
			ok = write_values(fptr, binary, cur->params[p].value);
	}

	if (fclose(fptr) != 0)
		ok = 0;

	if (!ok)
	{
		invoke_error_callback(ERR_IO, function_name);
		return ERR_IO;
	}
	return ERR_OK;
}

//------------------------------
// load a network saved in the text or binary format (v1 or v2)
//------------------------------
static PNetwork load_network(const char *filename, int binary)
{
	if (!filename)
		return NULL;

	const char *function_name = binary ? "ann_load_network_binary" : "ann_load_network";
	FILE *fptr = fopen(filename, binary ? "rb" : "rt");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, function_name);
		return NULL;
	}

	PNetwork pnet = NULL;
	int ver, optimizer, loss_type, layer_count;

	if (!read_int(fptr, binary, &ver))
		goto fail;

	if (ver != ANN_FORMAT_V1 && ver != ANN_FORMAT_V2)
	{
		invoke_error_callback(ERR_INVALID, binary ? "ann_load_network_binary: unsupported file version"
		                                           : "ann_load_network: unsupported file version");
		fclose(fptr);
		return NULL;
	}

	if (!read_int(fptr, binary, &optimizer) ||
		!read_int(fptr, binary, &loss_type) ||
		!read_int(fptr, binary, &layer_count) ||
		layer_count < 1)
		goto fail;

	pnet = ann_make_network(optimizer, loss_type);
	if (!pnet)
		goto fail;

	// create layers
	for (int layer = 0; layer < layer_count; layer++)
	{
		int layer_type, activation;

		if (ver == ANN_FORMAT_V1)
		{
			int node_count;
			if (!read_int(fptr, binary, &node_count) ||
				!read_int(fptr, binary, &layer_type) ||
				!read_int(fptr, binary, &activation) ||
				ann_add_layer(pnet, node_count, layer_type, activation) != ERR_OK)
				goto fail;
			continue;
		}

		int kind, config_count, config[ANN_LAYER_CONFIG_MAX];
		Layer_shape shape;
		if (!read_int(fptr, binary, &kind) ||
			!read_int(fptr, binary, &layer_type) ||
			!read_int(fptr, binary, &activation) ||
			!read_int(fptr, binary, &shape.c) ||
			!read_int(fptr, binary, &shape.h) ||
			!read_int(fptr, binary, &shape.w) ||
			!read_int(fptr, binary, &config_count) ||
			config_count < 0 || config_count > ANN_LAYER_CONFIG_MAX)
			goto fail;

		for (int i = 0; i < config_count; i++)
		{
			if (!read_int(fptr, binary, &config[i]))
				goto fail;
		}

		if (add_layer(pnet, kind, shape, config, config_count, layer_type, activation) != ERR_OK)
			goto fail;
	}

	// read parameters
	for (int layer = 1; layer < layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		if (ver == ANN_FORMAT_V1)
		{
			if (!read_values(fptr, binary, cur->t_bias) || !read_values(fptr, binary, cur->t_weights))
				goto fail;
			continue;
		}

		for (int p = 0; p < cur->param_count; p++)
		{
			if (!read_values(fptr, binary, cur->params[p].value))
				goto fail;
		}
	}

	fclose(fptr);

	// mark network weights as set
	pnet->weights_set = 1;

	return pnet;

fail:
	invoke_error_callback(ERR_INVALID, function_name);
	fclose(fptr);
	ann_free_network(pnet);
	return NULL;
}

//------------------------------
// save network to a binary file
//------------------------------
int ann_save_network_binary(const PNetwork pnet, const char *filename)
{
	return save_network(pnet, filename, 1, "ann_save_network_binary");
}

//------------------------------
// read network from binary file
//------------------------------
PNetwork ann_load_network_binary(const char *filename)
{
	return load_network(filename, 1);
}

//------------------------------
// save network to a text file
//------------------------------
int ann_save_network(const PNetwork pnet, const char *filename)
{
	return save_network(pnet, filename, 0, "ann_save_network");
}

//--------------------------------
// load a previously saved network
//--------------------------------
PNetwork ann_load_network(const char *filename)
{
	return load_network(filename, 0);
}

//-----------------------------------------------
// display the network properties
//-----------------------------------------------
void ann_print_props(const PNetwork pnet)
{
	ann_printf(pnet, "%s  Network shape:%s %s", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_CYAN));

	for (int i = 0; i < pnet->layer_count; i++)
	{
		const Layer_shape *shape = &pnet->layers[i].shape;
		if (i != 0)
			ann_printf(pnet, "-");
		if (shape->h == 1 && shape->w == 1)
			ann_printf(pnet, "%d", pnet->layers[i].node_count);
		else
			ann_printf(pnet, "%dx%dx%d", shape->c, shape->h, shape->w);
	}
	
	ann_printf(pnet, "%s\n", RESET());

	ann_printf(pnet, "%s      Optimizer:%s %s%s%s\n", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_MAGENTA), optimizers[pnet->optimizer], RESET());
	ann_printf(pnet, "%s  Loss function:%s %s%s%s\n", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_YELLOW), loss_types[pnet->loss_type], RESET());
	ann_printf(pnet, "%sMini-batch size:%s %s%u%s\n", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_BLUE), pnet->batchSize, RESET());
}

//-----------------------------------------------
// Helper: get ONNX op_type string for activation
//-----------------------------------------------
static const char* onnx_activation_op(Activation_type act)
{
	switch (act) {
		case ACTIVATION_SIGMOID:    return "Sigmoid";
		case ACTIVATION_RELU:       return "Relu";
		case ACTIVATION_LEAKY_RELU: return "LeakyRelu";
		case ACTIVATION_TANH:       return "Tanh";
		case ACTIVATION_SOFTSIGN:   return "Softsign";
		case ACTIVATION_SOFTMAX:    return "Softmax";
		case ACTIVATION_GELU:       return "Gelu";
		case ACTIVATION_NULL:
		default:                    return NULL;
	}
}

//-----------------------------------------------
// Helper: write JSON float array (tensor values)
//-----------------------------------------------
static void write_json_float(FILE *fptr, real value)
{
	double v = (double)value;
	if (isnan(v) || isinf(v))
		fprintf(fptr, "0.0");
	else
		fprintf(fptr, "%.9g", v);	// 9 significant digits round-trip any float exactly
}

static void write_json_float_array(FILE *fptr, const real *values, int count)
{
	fprintf(fptr, "[");

	for (int i = 0; i < count; i++)
	{
		if (i > 0) fprintf(fptr, ", ");
		write_json_float(fptr, values[i]);
	}
	
	fprintf(fptr, "]");
}

//-----------------------------------------------
// Helper: write a row-major [rows x cols] matrix transposed, as the
// row-major [cols x rows] data of a JSON float array
//-----------------------------------------------
static void write_json_float_array_transposed(FILE *fptr, const real *values, int rows, int cols)
{
	fprintf(fptr, "[");

	for (int c = 0; c < cols; c++)
	{
		for (int r = 0; r < rows; r++)
		{
			if (c > 0 || r > 0) fprintf(fptr, ", ");
			write_json_float(fptr, values[r * cols + c]);
		}
	}

	fprintf(fptr, "]");
}

//-----------------------------------------------
// Helper: write JSON int array (dims)
//-----------------------------------------------
static void write_json_int_array(FILE *fptr, const int *values, int count)
{
	fprintf(fptr, "[");

	for (int i = 0; i < count; i++)
	{
		if (i > 0) fprintf(fptr, ", ");
		fprintf(fptr, "%d", values[i]);
	}

	fprintf(fptr, "]");
}

//-----------------------------------------------
// Export network to ONNX JSON format
//-----------------------------------------------
// Dense ONNX initializers: weight_N [in_nodes, nodes] and bias_N [nodes],
// where N = layer - 1
//-----------------------------------------------
static void dense_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first)
{
	PLayer cur = &pnet->layers[layer];
	int index = layer - 1;
	int next_nodes = cur->node_count;
	int curr_nodes = pnet->layers[layer - 1].node_count;

	// ONNX MatMul: input [batch, curr_nodes] @ weights [curr_nodes, next_nodes] -> [batch, next_nodes].
	// libann stores W as [next_nodes x curr_nodes], so it is written transposed.
	if (!*first) fprintf(fptr, ",\n");
	*first = 0;

	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"weight_%d\",\n", index);
	fprintf(fptr, "        \"data_type\": 1,\n");  // 1 = FLOAT
	int w_dims[] = {curr_nodes, next_nodes};
	fprintf(fptr, "        \"dims\": ");
	write_json_int_array(fptr, w_dims, 2);
	fprintf(fptr, ",\n");
	fprintf(fptr, "        \"float_data\": ");
	write_json_float_array_transposed(fptr, cur->t_weights->values, next_nodes, curr_nodes);
	fprintf(fptr, "\n");
	fprintf(fptr, "      }");

	// Bias: shape is [next_nodes]
	fprintf(fptr, ",\n");
	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"bias_%d\",\n", index);
	fprintf(fptr, "        \"data_type\": 1,\n");
	int b_dims[] = {next_nodes};
	fprintf(fptr, "        \"dims\": ");
	write_json_int_array(fptr, b_dims, 1);
	fprintf(fptr, ",\n");
	fprintf(fptr, "        \"float_data\": ");
	write_json_float_array(fptr, cur->t_bias->values, next_nodes);
	fprintf(fptr, "\n");
	fprintf(fptr, "      }");
}

//-----------------------------------------------
// Dense ONNX nodes: MatMul with weight_N, then Add bias_N
//-----------------------------------------------
static void dense_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                             char *output, size_t output_size, int *first)
{
	(void)pnet;
	int index = layer - 1;
	char matmul_out[64];
	snprintf(matmul_out, sizeof(matmul_out), "matmul_%d_out", index);
	snprintf(output, output_size, "add_%d_out", index);

	// MatMul node: input @ weight_N -> matmul_N_out
	if (!*first) fprintf(fptr, ",\n");
	*first = 0;

	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"op_type\": \"MatMul\",\n");
	fprintf(fptr, "        \"name\": \"matmul_%d\",\n", index);
	fprintf(fptr, "        \"input\": [\"%s\", \"weight_%d\"],\n", input, index);
	fprintf(fptr, "        \"output\": [\"%s\"]\n", matmul_out);
	fprintf(fptr, "      }");

	// Add node: matmul_N_out + bias_N -> add_N_out
	fprintf(fptr, ",\n");
	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"op_type\": \"Add\",\n");
	fprintf(fptr, "        \"name\": \"add_%d\",\n", index);
	fprintf(fptr, "        \"input\": [\"%s\", \"bias_%d\"],\n", matmul_out, index);
	fprintf(fptr, "        \"output\": [\"%s\"]\n", output);
	fprintf(fptr, "      }");
}

//-----------------------------------------------
// Helper: 1 if a shape is a plain vector (H = W = 1)
//-----------------------------------------------
static int onnx_is_flat(const Layer_shape *shape)
{
	return shape->h == 1 && shape->w == 1;
}

//-----------------------------------------------
// Helper: write a tensor's shape, [batch, C, H, W] for an image or
// [batch, C*H*W] otherwise
//-----------------------------------------------
static void write_onnx_shape(FILE *fptr, const Layer_shape *shape, int image)
{
	if (image)
		fprintf(fptr, "            \"shape\": {\"dim\": [{\"dim_param\": \"batch\"}, {\"dim_value\": %d}, "
		              "{\"dim_value\": %d}, {\"dim_value\": %d}]}\n", shape->c, shape->h, shape->w);
	else
		fprintf(fptr, "            \"shape\": {\"dim\": [{\"dim_param\": \"batch\"}, {\"dim_value\": %d}]}\n",
		        shape->c * shape->h * shape->w);
}

//-----------------------------------------------
// Helper: write one ONNX node with a single input and output; attributes is
// the JSON for its attribute list, or NULL
//-----------------------------------------------
static void write_onnx_node(FILE *fptr, const char *op_type, const char *name, const char *input,
                            const char *output, const char *attributes, int *first)
{
	if (!*first) fprintf(fptr, ",\n");
	*first = 0;

	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"op_type\": \"%s\",\n", op_type);
	fprintf(fptr, "        \"name\": \"%s\",\n", name);
	fprintf(fptr, "        \"input\": [\"%s\"],\n", input);
	fprintf(fptr, "        \"output\": [\"%s\"]%s\n", output, attributes ? "," : "");
	if (attributes)
		fprintf(fptr, "        \"attribute\": %s\n", attributes);
	fprintf(fptr, "      }");
}

//-----------------------------------------------
// Conv ONNX initializers: weight_N [F, C, k, k] (libann's W [F x C*k*k] is
// already in that order) and bias_N [F], where N = layer - 1
//-----------------------------------------------
static void conv_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first)
{
	PLayer cur = &pnet->layers[layer];
	int index = layer - 1, k = cur->config[CONV_KERNEL];
	int w_dims[4];
	int b_dims[1];
	w_dims[0] = cur->shape.c;
	w_dims[1] = cur->in_shape.c;
	w_dims[2] = k;
	w_dims[3] = k;
	b_dims[0] = cur->shape.c;

	if (!*first) fprintf(fptr, ",\n");
	*first = 0;

	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"weight_%d\",\n", index);
	fprintf(fptr, "        \"data_type\": 1,\n");
	fprintf(fptr, "        \"dims\": ");
	write_json_int_array(fptr, w_dims, 4);
	fprintf(fptr, ",\n");
	fprintf(fptr, "        \"float_data\": ");
	write_json_float_array(fptr, cur->params[0].value->values, w_dims[0] * w_dims[1] * k * k);
	fprintf(fptr, "\n");
	fprintf(fptr, "      },\n");
	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"bias_%d\",\n", index);
	fprintf(fptr, "        \"data_type\": 1,\n");
	fprintf(fptr, "        \"dims\": ");
	write_json_int_array(fptr, b_dims, 1);
	fprintf(fptr, ",\n");
	fprintf(fptr, "        \"float_data\": ");
	write_json_float_array(fptr, cur->params[1].value->values, b_dims[0]);
	fprintf(fptr, "\n");
	fprintf(fptr, "      }");
}

//-----------------------------------------------
// Conv ONNX node: Conv(input, weight_N, bias_N) with kernel_shape, strides, pads
//-----------------------------------------------
static void conv_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                            char *output, size_t output_size, int *first)
{
	PLayer cur = &pnet->layers[layer];
	int index = layer - 1;
	int k = cur->config[CONV_KERNEL], st = cur->config[CONV_STRIDE], p = cur->config[CONV_PADDING];
	snprintf(output, output_size, "conv_%d_out", index);

	if (!*first) fprintf(fptr, ",\n");
	*first = 0;

	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"op_type\": \"Conv\",\n");
	fprintf(fptr, "        \"name\": \"conv_%d\",\n", index);
	fprintf(fptr, "        \"input\": [\"%s\", \"weight_%d\", \"bias_%d\"],\n", input, index, index);
	fprintf(fptr, "        \"output\": [\"%s\"],\n", output);
	fprintf(fptr, "        \"attribute\": [{\"name\": \"kernel_shape\", \"type\": 7, \"ints\": [%d, %d]}, "
	              "{\"name\": \"strides\", \"type\": 7, \"ints\": [%d, %d]}, "
	              "{\"name\": \"pads\", \"type\": 7, \"ints\": [%d, %d, %d, %d]}]\n", k, k, st, st, p, p, p, p);
	fprintf(fptr, "      }");
}

//-----------------------------------------------
// Kinds without parameters have no initializers
//-----------------------------------------------
static void no_onnx_initializers(FILE *fptr, const PNetwork pnet, int layer, int *first)
{
	(void)fptr;
	(void)pnet;
	(void)layer;
	(void)first;
}

//-----------------------------------------------
// MaxPool ONNX node with kernel_shape and strides
//-----------------------------------------------
static void maxpool_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                               char *output, size_t output_size, int *first)
{
	PLayer cur = &pnet->layers[layer];
	int size = cur->config[POOL_SIZE], stride = cur->config[POOL_STRIDE];
	char name[32], attributes[160];
	snprintf(name, sizeof(name), "maxpool_%d", layer - 1);
	snprintf(output, output_size, "maxpool_%d_out", layer - 1);
	snprintf(attributes, sizeof(attributes),
	         "[{\"name\": \"kernel_shape\", \"type\": 7, \"ints\": [%d, %d]}, "
	         "{\"name\": \"strides\", \"type\": 7, \"ints\": [%d, %d]}]", size, size, stride, stride);
	write_onnx_node(fptr, "MaxPool", name, input, output, attributes, first);
}

//-----------------------------------------------
// Flatten ONNX node: [batch, C, H, W] -> [batch, C*H*W]
//-----------------------------------------------
static void flatten_onnx_nodes(FILE *fptr, const PNetwork pnet, int layer, const char *input,
                               char *output, size_t output_size, int *first)
{
	char name[32];
	(void)pnet;
	snprintf(name, sizeof(name), "flatten_%d", layer - 1);
	snprintf(output, output_size, "flatten_%d_out", layer - 1);
	write_onnx_node(fptr, "Flatten", name, input, output, "[{\"name\": \"axis\", \"type\": 2, \"i\": 1}]", first);
}

//-----------------------------------------------
int ann_export_onnx(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_export_onnx");
		return ERR_NULL_PTR;
	}

	if (pnet->layer_count < 2)
	{
		invoke_error_callback(ERR_INVALID, "ann_export_onnx");
		return ERR_INVALID;
	}

	// The ONNX input is an image [batch, C, H, W] when the input layer has a
	// 2-D shape or conv/pool layers follow; dense layers take [batch, n], so
	// the tensor is flattened before them. Conv/pool need an image input.
	int image_input = !onnx_is_flat(&pnet->layers[0].shape);
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		Layer_kind kind = pnet->layers[layer].kind;
		if (kind == LAYER_KIND_CONV2D || kind == LAYER_KIND_MAXPOOL2D)
			image_input = 1;
	}

	int image = image_input;
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PLayer cur = &pnet->layers[layer];
		const Layer_ops *ops = layer_ops(cur);
		int needs_image = cur->kind == LAYER_KIND_CONV2D || cur->kind == LAYER_KIND_MAXPOOL2D;
		if (!ops->onnx_initializers || !ops->onnx_nodes || (needs_image && !image))
		{
			invoke_error_callback(ERR_INVALID, "ann_export_onnx");
			return ERR_INVALID;
		}
		image = needs_image;
	}

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_export_onnx");
		return ERR_IO;
	}

	// Start JSON object
	fprintf(fptr, "{\n");
	// Gelu first appears in opset 20 (IR version 9); other models keep opset 17
	int uses_gelu = 0;
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		if (pnet->layers[layer].activation == ACTIVATION_GELU)
			uses_gelu = 1;
	}
	fprintf(fptr, "  \"ir_version\": %d,\n", uses_gelu ? 9 : 8);
	fprintf(fptr, "  \"opset_import\": [{\"version\": %d}],\n", uses_gelu ? 20 : 17);
	fprintf(fptr, "  \"producer_name\": \"%s\",\n", ONNX_PRODUCER_NAME);
	fprintf(fptr, "  \"producer_version\": \"%s\",\n", ONNX_PRODUCER_VERSION);
	fprintf(fptr, "  \"graph\": {\n");
	fprintf(fptr, "    \"name\": \"ann_model\",\n");

	// ========== INITIALIZERS (weights and biases) ==========
	fprintf(fptr, "    \"initializer\": [\n");
	int first_init = 1;
	for (int layer = 1; layer < pnet->layer_count; layer++)
		layer_ops(&pnet->layers[layer])->onnx_initializers(fptr, pnet, layer, &first_init);
	fprintf(fptr, "\n    ],\n");

	// ========== NODES (operations) ==========
	fprintf(fptr, "    \"node\": [\n");
	int first_node = 1;
	char prev_output[64];
	snprintf(prev_output, sizeof(prev_output), "input");
	image = image_input;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// ONNX names count the layers that have parameters, from 0
		int index = layer - 1;
		char linear_out[64], act_out[64];
		snprintf(act_out, sizeof(act_out), "layer_%d_out", index);

		// Dense layers take [batch, n]: flatten an image first
		Layer_kind kind = pnet->layers[layer].kind;
		if (image && kind == LAYER_KIND_DENSE)
		{
			char flat_name[32], flat_out[64];
			snprintf(flat_name, sizeof(flat_name), "flatten_input_%d", index);
			snprintf(flat_out, sizeof(flat_out), "flatten_input_%d_out", index);
			write_onnx_node(fptr, "Flatten", flat_name, prev_output, flat_out,
			                "[{\"name\": \"axis\", \"type\": 2, \"i\": 1}]", &first_node);
			snprintf(prev_output, sizeof(prev_output), "%s", flat_out);
		}
		image = kind == LAYER_KIND_CONV2D || kind == LAYER_KIND_MAXPOOL2D;

		// The layer's own nodes (e.g. MatMul + Add), then its activation
		layer_ops(&pnet->layers[layer])->onnx_nodes(fptr, pnet, layer, prev_output,
		                                            linear_out, sizeof(linear_out), &first_node);

		Activation_type next_act = pnet->layers[layer].activation;
		const char *act_op = onnx_activation_op(next_act);

		// Activation node (if any)
		if (act_op != NULL)
		{
			fprintf(fptr, ",\n");
			fprintf(fptr, "      {\n");
			fprintf(fptr, "        \"op_type\": \"%s\",\n", act_op);
			fprintf(fptr, "        \"name\": \"activation_%d\",\n", index);
			fprintf(fptr, "        \"input\": [\"%s\"],\n", linear_out);
			fprintf(fptr, "        \"output\": [\"%s\"]", act_out);

			// LeakyRelu needs alpha attribute
			if (next_act == ACTIVATION_LEAKY_RELU)
			{
				fprintf(fptr, ",\n");
				fprintf(fptr, "        \"attribute\": [{\"name\": \"alpha\", \"type\": 1, \"f\": 0.01}]\n");
			}
			// Softmax needs axis attribute (default -1 for last axis)
			else if (next_act == ACTIVATION_SOFTMAX)
			{
				fprintf(fptr, ",\n");
				fprintf(fptr, "        \"attribute\": [{\"name\": \"axis\", \"type\": 2, \"i\": -1}]\n");
			}
			// Gelu defaults to the exact erf form; libann uses the tanh approximation.
			// String attributes are protobuf bytes, so JSON holds base64("tanh").
			else if (next_act == ACTIVATION_GELU)
			{
				fprintf(fptr, ",\n");
				fprintf(fptr, "        \"attribute\": [{\"name\": \"approximate\", \"type\": 3, \"s\": \"dGFuaA==\"}]\n");
			}
			else
			{
				fprintf(fptr, "\n");
			}
			fprintf(fptr, "      }");
			snprintf(prev_output, sizeof(prev_output), "%s", act_out);
		}
		else
		{
			// No activation, next input is the linear output
			snprintf(prev_output, sizeof(prev_output), "%s", linear_out);
		}
	}
	fprintf(fptr, "\n    ],\n");

	// ========== INPUT ==========
	fprintf(fptr, "    \"input\": [\n");
	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"input\",\n");
	fprintf(fptr, "        \"type\": {\n");
	fprintf(fptr, "          \"tensor_type\": {\n");
	fprintf(fptr, "            \"elem_type\": 1,\n");
	write_onnx_shape(fptr, &pnet->layers[0].shape, image_input);
	fprintf(fptr, "          }\n");
	fprintf(fptr, "        }\n");
	fprintf(fptr, "      }\n");
	fprintf(fptr, "    ],\n");

	// ========== OUTPUT ==========
	fprintf(fptr, "    \"output\": [\n");
	fprintf(fptr, "      {\n");
	fprintf(fptr, "        \"name\": \"%s\",\n", prev_output);
	fprintf(fptr, "        \"type\": {\n");
	fprintf(fptr, "          \"tensor_type\": {\n");
	fprintf(fptr, "            \"elem_type\": 1,\n");
	write_onnx_shape(fptr, &pnet->layers[pnet->layer_count - 1].shape, image);
	fprintf(fptr, "          }\n");
	fprintf(fptr, "        }\n");
	fprintf(fptr, "      }\n");
	fprintf(fptr, "    ]\n");

	// Close graph and root
	fprintf(fptr, "  }\n");
	fprintf(fptr, "}\n");

	fclose(fptr);
	return ERR_OK;
}

//------------------------------
// Map ONNX activation op to libann type
//------------------------------
static Activation_type onnx_op_to_activation(const char *op_type)
{
	if (!op_type) return ACTIVATION_NULL;
	if (strcmp(op_type, "Sigmoid") == 0) return ACTIVATION_SIGMOID;
	if (strcmp(op_type, "Relu") == 0) return ACTIVATION_RELU;
	if (strcmp(op_type, "LeakyRelu") == 0) return ACTIVATION_LEAKY_RELU;
	if (strcmp(op_type, "Tanh") == 0) return ACTIVATION_TANH;
	if (strcmp(op_type, "Softsign") == 0) return ACTIVATION_SOFTSIGN;
	if (strcmp(op_type, "Softmax") == 0) return ACTIVATION_SOFTMAX;
	if (strcmp(op_type, "Gelu") == 0) return ACTIVATION_GELU;
	return ACTIVATION_NULL;
}

//================================================================================================
// ONNX import. The graph is read as a chain of nodes from its input, each becoming a layer:
//   MatMul [+ Add]  dense layer (weights [in, out], bias [out])
//   Gemm            dense layer (PyTorch's Linear: weights [out, in] with transB = 1)
//   Conv            conv layer (weights [F, C, k, k], optional bias [F])
//   MaxPool         max pooling layer
//   Flatten         flatten layer (libann's own "flatten_input_N" nodes, inserted before
//                   dense layers, are skipped: dense layers read any shape)
//   activation op   the activation of the layer just before it
// Anything libann can't reproduce exactly is rejected rather than approximated.
//================================================================================================

//------------------------------
// a field of an ONNX JSON object by its proto name (e.g. "op_type"), also
// accepting the lowerCamelCase spelling protobuf's JSON printer uses by
// default ("opType")
//------------------------------
static JsonValue *onnx_get(const JsonValue *obj, const char *name)
{
	JsonValue *v = obj ? json_get(obj, name) : NULL;
	if (v || !obj || !strchr(name, '_'))
		return v;

	char camel[64];
	size_t n = 0;
	for (const char *c = name; *c && n + 1 < sizeof(camel); c++)
	{
		if (*c == '_' && c[1])
			camel[n++] = (char)toupper((unsigned char)*++c);
		else
			camel[n++] = *c;
	}
	camel[n] = '\0';
	return json_get(obj, camel);
}

//------------------------------
// a JSON number, or a numeric string (protobuf's JSON form writes int64 as strings)
//------------------------------
static int onnx_number(const JsonValue *v, double *out)
{
	if (v && v->type == JSON_STRING)
	{
		char *end;
		*out = strtod(v->u.str_val, &end);
		return end != v->u.str_val && *end == '\0';
	}
	return v && json_number(v, out) == 0;
}

static int onnx_int(const JsonValue *v, int *out)
{
	double d;
	if (!onnx_number(v, &d) || d != (double)(int)d)
		return 0;
	*out = (int)d;
	return 1;
}

//------------------------------
// name of a node's i-th input or output (NULL when absent)
//------------------------------
static const char *onnx_io(const JsonValue *node, const char *key, size_t i)
{
	JsonValue *list = onnx_get(node, key);
	return (list && i < json_array_length(list)) ? json_string(json_at(list, i)) : NULL;
}

static JsonValue *onnx_initializer(const JsonValue *graph, const char *name)
{
	JsonValue *list = onnx_get(graph, "initializer");
	for (size_t i = 0; name && list && i < json_array_length(list); i++)
	{
		const char *n = json_string(onnx_get(json_at(list, i), "name"));
		if (n && strcmp(n, name) == 0)
			return json_at(list, i);
	}
	return NULL;
}

static JsonValue *onnx_attribute(const JsonValue *node, const char *name)
{
	JsonValue *list = onnx_get(node, "attribute");
	for (size_t i = 0; list && i < json_array_length(list); i++)
	{
		const char *n = json_string(onnx_get(json_at(list, i), "name"));
		if (n && strcmp(n, name) == 0)
			return json_at(list, i);
	}
	return NULL;
}

//------------------------------
// read an int list attribute ("ints") into out; returns the count, 0 when
// the attribute is absent, or -1 when it is malformed or longer than max
//------------------------------
static int onnx_attribute_ints(const JsonValue *node, const char *name, int *out, int max)
{
	JsonValue *attr = onnx_attribute(node, name);
	if (!attr)
		return 0;

	JsonValue *ints = onnx_get(attr, "ints");
	int count = ints ? (int)json_array_length(ints) : -1;
	if (count < 0 || count > max)
		return -1;
	for (int i = 0; i < count; i++)
	{
		if (!onnx_int(json_at(ints, i), &out[i]))
			return -1;
	}
	return count;
}

//------------------------------
// 1 if an int list attribute is absent or holds only the value expected
//------------------------------
static int onnx_ints_all(const JsonValue *node, const char *name, int expected)
{
	int v[8];
	int count = onnx_attribute_ints(node, name, v, 8);
	for (int i = 0; i < count; i++)
	{
		if (v[i] != expected)
			return 0;
	}
	return count >= 0;
}

//------------------------------
// an int list attribute whose entries are all equal: returns 1 and sets *value
// (fallback when absent), 0 otherwise
//------------------------------
static int onnx_uniform_ints(const JsonValue *node, const char *name, int fallback, int *value)
{
	int v[8];
	int count = onnx_attribute_ints(node, name, v, 8);
	*value = count > 0 ? v[0] : fallback;
	for (int i = 1; i < count; i++)
	{
		if (v[i] != v[0])
			return 0;
	}
	return count >= 0;
}

//------------------------------
// 1 if a string attribute is absent or equals text (protobuf's JSON form
// holds string attributes as base64, so both spellings are accepted)
//------------------------------
static int onnx_string_is(const JsonValue *node, const char *name, const char *text, const char *base64)
{
	JsonValue *attr = onnx_attribute(node, name);
	const char *v = attr ? json_string(onnx_get(attr, "s")) : NULL;
	return !attr || (v && (strcmp(v, text) == 0 || strcmp(v, base64) == 0));
}

//------------------------------
// decode base64 text into out (at most max bytes); returns the byte count or -1
//------------------------------
static int onnx_base64_decode(const char *text, unsigned char *out, int max)
{
	int bits = 0, count = 0;
	unsigned long acc = 0;

	for (const char *c = text; *c && *c != '='; c++)
	{
		const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		const char *pos = strchr(alphabet, *c);
		if (!pos)
		{
			if (*c == '-' || *c == '_')	// URL-safe alphabet
				pos = alphabet + (*c == '-' ? 62 : 63);
			else
				return -1;
		}

		acc = (acc << 6) | (unsigned long)(pos - alphabet);
		bits += 6;
		if (bits >= 8)
		{
			bits -= 8;
			if (count >= max)
				return -1;
			out[count++] = (unsigned char)((acc >> bits) & 0xFF);
		}
	}
	return count;
}

//------------------------------
// copy an initializer's values into t, from float_data or from raw_data
// (base64 of little-endian float32, the form most ONNX exporters write);
// with transpose the data is a [t->cols x t->rows] matrix (ONNX MatMul
// layout) and is stored transposed
//------------------------------
static int onnx_load_values(const JsonValue *init, PTensor t, int transpose)
{
	JsonValue *data = init ? onnx_get(init, "float_data") : NULL;
	const char *raw = init ? json_string(onnx_get(init, "raw_data")) : NULL;
	int count = t->rows * t->cols;
	unsigned char *bytes = NULL;
	int data_type = 1;

	if (init && onnx_get(init, "data_type") && !onnx_int(onnx_get(init, "data_type"), &data_type))
		data_type = -1;	// e.g. an enum name; only FLOAT (1) is supported
	if (data_type != 1)
		return 0;

	if (raw)
	{
		bytes = malloc((size_t)count * 4);
		if (!bytes || onnx_base64_decode(raw, bytes, count * 4) != count * 4)
		{
			free(bytes);
			return 0;
		}
	}
	else if (!data || data->type != JSON_ARRAY || (int)json_array_length(data) != count)
		return 0;

	for (int j = 0; j < count; j++)
	{
		double v;
		if (bytes)
		{
			unsigned char *b = bytes + 4 * j;
			unsigned long u = b[0] | ((unsigned long)b[1] << 8) | ((unsigned long)b[2] << 16) | ((unsigned long)b[3] << 24);
			float f;
			unsigned int u32 = (unsigned int)u;
			memcpy(&f, &u32, sizeof(f));
			v = f;
		}
		else if (!onnx_number(json_at(data, j), &v))
			return 0;

		// element j of the [cols x rows] matrix is t[j % rows][j / rows]
		int dst = transpose ? (j % t->rows) * t->cols + j / t->rows : j;
		t->values[dst] = (real)v;
	}
	free(bytes);
	return 1;
}

//------------------------------
// an initializer's dims; returns their count, or -1 when malformed
//------------------------------
static int onnx_dims(const JsonValue *init, int *dims, int max)
{
	JsonValue *list = init ? onnx_get(init, "dims") : NULL;
	int count = list ? (int)json_array_length(list) : -1;
	if (count < 0 || count > max)
		return -1;
	for (int i = 0; i < count; i++)
	{
		if (!onnx_int(json_at(list, i), &dims[i]))
			return -1;
	}
	return count;
}

//------------------------------
// the activation of a node that may follow a layer; ACTIVATION_NULL with
// *ok = 1 when it is not an activation, *ok = 0 when it is one libann can't
// match exactly
//------------------------------
static Activation_type onnx_node_activation(const JsonValue *node, int *ok)
{
	const char *op = json_string(onnx_get(node, "op_type"));
	Activation_type act = onnx_op_to_activation(op);
	*ok = 1;

	if (act == ACTIVATION_LEAKY_RELU)
	{
		JsonValue *alpha = onnx_attribute(node, "alpha");
		double a = 0.01;
		if (alpha && !onnx_number(onnx_get(alpha, "f"), &a))
			a = -1;
		*ok = fabs(a - 0.01) < 1e-6;
	}
	else if (act == ACTIVATION_GELU)
	{
		// libann implements the tanh approximation only
		JsonValue *attr = onnx_attribute(node, "approximate");
		*ok = attr && onnx_string_is(node, "approximate", "tanh", "dGFuaA==");
	}
	else if (act == ACTIVATION_SOFTMAX)
	{
		JsonValue *axis = onnx_attribute(node, "axis");
		int a = -1;
		if (axis && !onnx_int(onnx_get(axis, "i"), &a))
			a = 0;
		*ok = (a == -1 || a == 1);
	}
	return act;
}

//------------------------------
// Import network from ONNX JSON file
//------------------------------
PNetwork ann_import_onnx(const char *filename)
{
	if (!filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_import_onnx");
		return NULL;
	}

	// Parse JSON file
	JsonValue root;
	if (json_parse_file(filename, &root) != 0)
	{
		invoke_error_callback(ERR_IO, "ann_import_onnx");
		return NULL;
	}

	PNetwork pnet = NULL;
	const char *error = "ann_import_onnx: invalid model";

	JsonValue *graph = onnx_get(&root, "graph");
	JsonValue *nodes = graph ? onnx_get(graph, "node") : NULL;
	if (!graph || graph->type != JSON_OBJECT || !nodes || nodes->type != JSON_ARRAY)
	{
		error = "ann_import_onnx: missing graph or nodes";
		goto fail;
	}

	// Dense weights are stored [in, out] as ONNX MatMul expects; ann-library 1.0
	// exports wrote libann's [out, in] data without transposing it
	const char *producer = json_string(onnx_get(&root, "producer_name"));
	const char *producer_version = json_string(onnx_get(&root, "producer_version"));
	int legacy_layout = producer && producer_version &&
	                    strcmp(producer, ONNX_PRODUCER_NAME) == 0 && strcmp(producer_version, "1.0") == 0;

	// The graph input: the first one that isn't an initializer (older models list both)
	JsonValue *inputs = onnx_get(graph, "input");
	JsonValue *input = NULL;
	for (size_t i = 0; inputs && i < json_array_length(inputs); i++)
	{
		JsonValue *candidate = json_at(inputs, i);
		if (!onnx_initializer(graph, json_string(onnx_get(candidate, "name"))))
		{
			input = candidate;
			break;
		}
	}

	JsonValue *dim = onnx_get(onnx_get(onnx_get(onnx_get(input, "type"), "tensor_type"), "shape"), "dim");
	int rank = dim ? (int)json_array_length(dim) : 0;
	int size[4] = {0, 0, 0, 0};
	for (int i = 1; i < rank && i < 4; i++)
	{
		if (!onnx_int(onnx_get(json_at(dim, i), "dim_value"), &size[i]))
			rank = 0;
	}
	if (!input && json_array_length(nodes) > 0)
	{
		// no declared input: a dense model's width comes from its first weights
		JsonValue *first = json_at(nodes, 0);
		const char *first_op = json_string(onnx_get(first, "op_type"));
		int dims[2];
		if (first_op && strcmp(first_op, "MatMul") == 0 &&
			onnx_dims(onnx_initializer(graph, onnx_io(first, "input", 1)), dims, 2) == 2)
		{
			rank = 2;
			size[1] = dims[0];
		}
	}
	if (rank != 2 && rank != 4)
	{
		error = "ann_import_onnx: input must be [batch, n] or [batch, C, H, W]";
		goto fail;
	}

	pnet = ann_make_network(OPT_ADAM, LOSS_MSE);
	if (!pnet)
	{
		error = "ann_import_onnx";
		goto fail;
	}
	int err = rank == 2 ? ann_add_layer(pnet, size[1], LAYER_INPUT, ACTIVATION_NULL)
	                    : ann_add_input(pnet, size[1], size[2], size[3]);
	if (err != ERR_OK)
		goto fail;

	// Walk the chain of nodes
	const char *current = input ? json_string(onnx_get(input, "name")) : onnx_io(json_at(nodes, 0), "input", 0);
	int image = rank == 4;
	size_t count = json_array_length(nodes);

	for (size_t i = 0; i < count; i++)
	{
		JsonValue *node = json_at(nodes, i);
		const char *op = json_string(onnx_get(node, "op_type"));
		const char *name = json_string(onnx_get(node, "name"));
		const char *output = onnx_io(node, "output", 0);
		const char *in0 = onnx_io(node, "input", 0);

		if (!op || !output || !in0 || !current || strcmp(in0, current) != 0)
		{
			error = "ann_import_onnx: only a single chain of nodes is supported";
			goto fail;
		}

		PLayer prev = &pnet->layers[pnet->layer_count - 1];
		Layer_kind kind;
		int units = 0, kernel = 0, stride = 1, pad = 0;
		int transpose = 0;	// dense weights stored [in, out] (MatMul layout)
		JsonValue *weight = NULL, *bias = NULL;

		if (strcmp(op, "MatMul") == 0)
		{
			int dims[2];
			weight = onnx_initializer(graph, onnx_io(node, "input", 1));
			if (image || onnx_dims(weight, dims, 2) != 2 || dims[0] != prev->node_count)
			{
				error = "ann_import_onnx: MatMul weights must be [in, out] on a flat input";
				goto fail;
			}
			kind = LAYER_KIND_DENSE;
			units = dims[1];
			transpose = !legacy_layout;

			// an Add of a bias vector right after it belongs to the same layer
			JsonValue *next = i + 1 < count ? json_at(nodes, i + 1) : NULL;
			const char *next_op = next ? json_string(onnx_get(next, "op_type")) : NULL;
			const char *next_in = next ? onnx_io(next, "input", 0) : NULL;
			if (next_op && strcmp(next_op, "Add") == 0 && next_in && strcmp(next_in, output) == 0)
			{
				int bdims[1];
				bias = onnx_initializer(graph, onnx_io(next, "input", 1));
				if (onnx_dims(bias, bdims, 1) != 1 || bdims[0] != units)
				{
					error = "ann_import_onnx: dense bias must be [out]";
					goto fail;
				}
				output = onnx_io(next, "output", 0);
				i++;
			}
		}
		else if (strcmp(op, "Gemm") == 0)
		{
			// Gemm(x, W, b) = alpha * x * op(W) + beta * b, as PyTorch writes Linear
			int dims[2], trans_a = 0, trans_b = 0, bdims[2], nb;
			double alpha = 1.0, beta = 1.0;
			JsonValue *attr;
			if ((attr = onnx_attribute(node, "transA")) && !onnx_int(onnx_get(attr, "i"), &trans_a)) trans_a = 1;
			if ((attr = onnx_attribute(node, "transB")) && !onnx_int(onnx_get(attr, "i"), &trans_b)) trans_b = -1;
			if ((attr = onnx_attribute(node, "alpha")) && !onnx_number(onnx_get(attr, "f"), &alpha)) alpha = 0;
			if ((attr = onnx_attribute(node, "beta")) && !onnx_number(onnx_get(attr, "f"), &beta)) beta = 0;

			weight = onnx_initializer(graph, onnx_io(node, "input", 1));
			bias = onnx_initializer(graph, onnx_io(node, "input", 2));
			int in_dim = trans_b ? 1 : 0;
			if (image || trans_a != 0 || (trans_b != 0 && trans_b != 1) || alpha != 1.0 || beta != 1.0 ||
				onnx_dims(weight, dims, 2) != 2 || dims[in_dim] != prev->node_count)
			{
				error = "ann_import_onnx: unsupported Gemm (needs transA = 0, alpha = beta = 1, flat input)";
				goto fail;
			}
			kind = LAYER_KIND_DENSE;
			units = dims[1 - in_dim];
			transpose = !trans_b;	// transB = 1 stores W as [out, in], libann's layout

			// the bias must be one value per output ([out] or [1, out])
			nb = bias ? onnx_dims(bias, bdims, 2) : 0;
			if (bias && !((nb == 1 && bdims[0] == units) || (nb == 2 && bdims[0] == 1 && bdims[1] == units)))
			{
				error = "ann_import_onnx: Gemm bias must have one value per output";
				goto fail;
			}
		}
		else if (strcmp(op, "Conv") == 0)
		{
			JsonValue *group = onnx_attribute(node, "group");
			int groups = 1;
			if (group && !onnx_int(onnx_get(group, "i"), &groups))
				groups = 0;
			if (groups != 1)
			{
				error = "ann_import_onnx: grouped convolutions are not supported";
				goto fail;
			}
			int dims[4];
			weight = onnx_initializer(graph, onnx_io(node, "input", 1));
			bias = onnx_initializer(graph, onnx_io(node, "input", 2));
			if (!image || onnx_dims(weight, dims, 4) != 4 || dims[1] != prev->shape.c || dims[2] != dims[3] ||
				!onnx_uniform_ints(node, "kernel_shape", dims[2], &kernel) || kernel != dims[2] ||
				!onnx_uniform_ints(node, "strides", 1, &stride) || !onnx_uniform_ints(node, "pads", 0, &pad) ||
				!onnx_ints_all(node, "dilations", 1) || !onnx_string_is(node, "auto_pad", "NOTSET", "Tk9UU0VU"))
			{
				error = "ann_import_onnx: unsupported Conv (needs square kernels, equal strides and pads, no dilation)";
				goto fail;
			}
			kind = LAYER_KIND_CONV2D;
			units = dims[0];
		}
		else if (strcmp(op, "MaxPool") == 0)
		{
			int ceil_mode = 0;
			JsonValue *ceil = onnx_attribute(node, "ceil_mode");
			if (ceil && !onnx_int(onnx_get(ceil, "i"), &ceil_mode))
				ceil_mode = 1;
			if (!image || !onnx_attribute(node, "kernel_shape") ||
				!onnx_uniform_ints(node, "kernel_shape", 0, &kernel) || !onnx_uniform_ints(node, "strides", 1, &stride) ||
				!onnx_ints_all(node, "pads", 0) || !onnx_ints_all(node, "dilations", 1) || ceil_mode != 0 ||
				!onnx_string_is(node, "auto_pad", "NOTSET", "Tk9UU0VU"))
			{
				error = "ann_import_onnx: unsupported MaxPool (needs square windows, equal strides, no padding)";
				goto fail;
			}
			kind = LAYER_KIND_MAXPOOL2D;
		}
		else if (strcmp(op, "Flatten") == 0)
		{
			JsonValue *axis = onnx_attribute(node, "axis");
			int a = 1;
			if (axis && !onnx_int(onnx_get(axis, "i"), &a))
				a = 0;
			if (a != 1)
			{
				error = "ann_import_onnx: Flatten must use axis 1";
				goto fail;
			}
			image = 0;
			current = output;

			// libann inserts these before dense layers, which read any shape
			if (name && strncmp(name, "flatten_input_", 14) == 0)
				continue;
			kind = LAYER_KIND_FLATTEN;
		}
		else
		{
			error = "ann_import_onnx: unsupported op";
			goto fail;
		}

		// an activation op right after the layer is the layer's activation
		Activation_type act = ACTIVATION_NULL;
		JsonValue *next = i + 1 < count ? json_at(nodes, i + 1) : NULL;
		const char *next_in = next ? onnx_io(next, "input", 0) : NULL;
		if (next && next_in && strcmp(next_in, output) == 0)
		{
			int ok;
			Activation_type next_act = onnx_node_activation(next, &ok);
			if (!ok || (next_act != ACTIVATION_NULL && (kind == LAYER_KIND_MAXPOOL2D || kind == LAYER_KIND_FLATTEN)))
			{
				error = "ann_import_onnx: unsupported activation";
				goto fail;
			}
			if (next_act != ACTIVATION_NULL)
			{
				act = next_act;
				output = onnx_io(next, "output", 0);
				i++;
			}
		}

		switch (kind)
		{
		case LAYER_KIND_DENSE:		err = ann_add_layer(pnet, units, LAYER_HIDDEN, act); break;
		case LAYER_KIND_CONV2D:		err = ann_add_conv2d(pnet, units, kernel, stride, pad, act); break;
		case LAYER_KIND_MAXPOOL2D:	err = ann_add_maxpool2d(pnet, kernel, stride); break;
		default:					err = ann_add_flatten(pnet); break;
		}
		if (err != ERR_OK)
			goto fail;

		// parameters
		PLayer cur = &pnet->layers[pnet->layer_count - 1];
		if (weight && !onnx_load_values(weight, cur->params[0].value, transpose))
		{
			error = "ann_import_onnx: weight data does not match its dims";
			goto fail;
		}
		if (weight && bias && !onnx_load_values(bias, cur->params[1].value, 0))
		{
			error = "ann_import_onnx: bias data does not match its dims";
			goto fail;
		}

		if (kind == LAYER_KIND_CONV2D || kind == LAYER_KIND_MAXPOOL2D)
			image = 1;
		current = output;
	}

	if (pnet->layer_count < 2)
	{
		error = "ann_import_onnx: no layers found";
		goto fail;
	}

	pnet->layers[pnet->layer_count - 1].layer_type = LAYER_OUTPUT;
	pnet->weights_set = 1;
	json_free(&root);
	return pnet;

fail:
	invoke_error_callback(ERR_INVALID, error);
	ann_free_network(pnet);
	json_free(&root);
	return NULL;
}

//------------------------------
// Helper to get activation name string
//------------------------------
static const char *get_activation_name(Activation_type act)
{
	switch (act)
	{
	case ACTIVATION_SIGMOID:    return "Sigmoid";
	case ACTIVATION_RELU:       return "ReLU";
	case ACTIVATION_TANH:       return "Tanh";
	case ACTIVATION_LEAKY_RELU: return "LeakyReLU";
	case ACTIVATION_SOFTMAX:    return "Softmax";
	case ACTIVATION_SOFTSIGN:   return "Softsign";
	case ACTIVATION_GELU:       return "GELU";
	default:                    return NULL;
	}
}

//------------------------------
// Export network as PIKCHR diagram
//------------------------------
int ann_export_pikchr(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_export_pikchr");
		return ERR_NULL_PTR;
	}

	if (pnet->layer_count == 0)
	{
		invoke_error_callback(ERR_INVALID, "ann_export_pikchr");
		return ERR_INVALID;
	}

	FILE *fptr = fopen(filename, "w");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_export_pikchr");
		return ERR_IO;
	}

	// Determine if we should use detailed mode (show individual nodes)
	int max_nodes = 0;
	for (int i = 0; i < pnet->layer_count; i++)
	{
		if (pnet->layers[i].node_count > max_nodes)
			max_nodes = pnet->layers[i].node_count;
	}
	// image-shaped and non-dense layers are drawn as boxes
	int detailed_mode = (max_nodes <= 10) && network_fits_v1(pnet);

	fprintf(fptr, "# Neural Network Architecture\n");
	fprintf(fptr, "# Generated by libann\n\n");

	if (detailed_mode)
	{
		// Detailed mode: draw individual nodes
		real layer_spacing = 2.0f;
		real node_radius = 0.12f;
		
		// Draw nodes for each layer
		for (int layer = 0; layer < pnet->layer_count; layer++)
		{
			int nodes = pnet->layers[layer].node_count;
			real layer_x = layer * layer_spacing;
			real layer_height = (nodes - 1) * 0.4f;
			real start_y = layer_height / 2.0f;
			
			const char *layer_type = (layer == 0) ? "Input" : 
			                         (layer == pnet->layer_count - 1) ? "Output" : "Hidden";
			
			fprintf(fptr, "# Layer %d: %s (%d nodes)\n", layer, layer_type, nodes);
			
			for (int n = 0; n < nodes; n++)
			{
				real node_y = start_y - n * 0.4f;
				fprintf(fptr, "L%dN%d: circle rad %.2f at (%.1f, %.2f)\n", 
					layer, n, node_radius, layer_x, node_y);
			}
			fprintf(fptr, "\n");
		}
		
		// Draw connections between adjacent layers
		fprintf(fptr, "# Connections\n");
		for (int layer = 0; layer < pnet->layer_count - 1; layer++)
		{
			int from_nodes = pnet->layers[layer].node_count;
			int to_nodes = pnet->layers[layer + 1].node_count;
			
			// For small networks, draw all connections
			// For larger ones, just draw a few representative ones
			int max_connections = 20;
			int drawn = 0;
			
			for (int from = 0; from < from_nodes && drawn < max_connections; from++)
			{
				for (int to = 0; to < to_nodes && drawn < max_connections; to++)
				{
					fprintf(fptr, "line thin color gray from L%dN%d.e to L%dN%d.w\n",
						layer, from, layer + 1, to);
					drawn++;
				}
			}
		}
		
		// Add layer labels
		fprintf(fptr, "\n# Labels\n");
		for (int layer = 0; layer < pnet->layer_count; layer++)
		{
			real layer_x = layer * layer_spacing;
			int nodes = pnet->layers[layer].node_count;
			real layer_height = (nodes - 1) * 0.4f;
			real label_y = layer_height / 2.0f + 0.4f;
			
			const char *layer_type = (layer == 0) ? "Input" : 
			                         (layer == pnet->layer_count - 1) ? "Output" : "Hidden";
			const char *act_name = get_activation_name(pnet->layers[layer].activation);
			
			if (act_name)
				fprintf(fptr, "text \"%s\" \"%s\" at (%.1f, %.2f)\n", 
					layer_type, act_name, layer_x, label_y);
			else
				fprintf(fptr, "text \"%s\" at (%.1f, %.2f)\n", 
					layer_type, layer_x, label_y);
		}
	}
	else
	{
		// Simple mode: boxes for each layer
		fprintf(fptr, "right\n\n");
		
		for (int layer = 0; layer < pnet->layer_count; layer++)
		{
			const Layer *cur = &pnet->layers[layer];
			const char *layer_type = (layer == 0) ? "Input" : 
			                         (layer == pnet->layer_count - 1) ? "Output" :
			                         (cur->kind != LAYER_KIND_DENSE) ? layer_ops(cur)->name : "Hidden";
			const char *act_name = get_activation_name(cur->activation);
			char size[48];
			if (cur->shape.h == 1 && cur->shape.w == 1)
				snprintf(size, sizeof(size), "%d nodes", cur->node_count);
			else
				snprintf(size, sizeof(size), "%dx%dx%d", cur->shape.c, cur->shape.h, cur->shape.w);
			
			if (act_name)
				fprintf(fptr, "box \"%s\" \"%s\" \"(%s)\"\n", 
					layer_type, size, act_name);
			else
				fprintf(fptr, "box \"%s\" \"%s\"\n", layer_type, size);
			
			if (layer < pnet->layer_count - 1)
				fprintf(fptr, "arrow right 0.3\n");
		}
	}

	fclose(fptr);
	return ERR_OK;
}

//------------------------------
// Record epoch in training history
//------------------------------
static void record_history(PNetwork pnet, real loss, real learning_rate, real val_loss)
{
	// Check if we need to allocate or expand history
	if (pnet->history_count >= pnet->history_capacity)
	{
		unsigned new_capacity = pnet->history_capacity == 0 ? 256 : pnet->history_capacity * 2;
		if (new_capacity > MAX_HISTORY_SIZE)
			new_capacity = MAX_HISTORY_SIZE;
		
		if (pnet->history_count >= MAX_HISTORY_SIZE)
			return;  // History full, stop recording
		
		// Store each buffer as soon as it grows: a failed realloc leaves the
		// old buffer valid, so nothing is lost or freed twice
		real *new_loss = (real *)realloc(pnet->loss_history, new_capacity * sizeof(real));
		if (!new_loss)
			return;  // Allocation failed, skip recording
		pnet->loss_history = new_loss;

		real *new_lr = (real *)realloc(pnet->lr_history, new_capacity * sizeof(real));
		if (!new_lr)
			return;
		pnet->lr_history = new_lr;

		real *new_val = (real *)realloc(pnet->val_loss_history, new_capacity * sizeof(real));
		if (!new_val)
			return;
		pnet->val_loss_history = new_val;

		pnet->history_capacity = new_capacity;
	}
	
	pnet->loss_history[pnet->history_count] = loss;
	pnet->lr_history[pnet->history_count] = learning_rate;
	pnet->val_loss_history[pnet->history_count] = val_loss;
	pnet->history_count++;
}

//------------------------------
// Clear training history
//------------------------------
void ann_clear_history(PNetwork pnet)
{
	if (!pnet)
		return;
	
	free(pnet->loss_history);
	free(pnet->lr_history);
	free(pnet->val_loss_history);
	pnet->loss_history = NULL;
	pnet->lr_history = NULL;
	pnet->val_loss_history = NULL;
	pnet->history_count = 0;
	pnet->history_capacity = 0;
}

//------------------------------
// Export learning curve as CSV
//------------------------------
int ann_export_learning_curve(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_export_learning_curve");
		return ERR_NULL_PTR;
	}

	if (pnet->history_count == 0)
	{
		invoke_error_callback(ERR_INVALID, "ann_export_learning_curve");
		return ERR_INVALID;
	}

	FILE *fptr = fopen(filename, "w");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_export_learning_curve");
		return ERR_IO;
	}

	// Add a val_loss column only when some epoch measured one
	int has_val = 0;
	for (unsigned i = 0; i < pnet->history_count; i++)
	{
		if (pnet->val_loss_history[i] >= (real)0.0)
			has_val = 1;
	}

	// Write header
	fprintf(fptr, has_val ? "epoch,loss,learning_rate,val_loss\n" : "epoch,loss,learning_rate\n");

	// Write data
	for (unsigned i = 0; i < pnet->history_count; i++)
	{
		fprintf(fptr, "%u,%g,%g", i + 1, pnet->loss_history[i], pnet->lr_history[i]);
		if (has_val && pnet->val_loss_history[i] >= (real)0.0)
			fprintf(fptr, ",%g", pnet->val_loss_history[i]);
		else if (has_val)
			fputc(',', fptr);
		fputc('\n', fptr);
	}

	fclose(fptr);
	return ERR_OK;
}
