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

#ifdef _WIN32
#	define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include "ann.h"
#include "json.h"

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

ANN_API int ann_gpu_upload_network(PNetwork pnet)
{
    if (!g_gpu_backend)
        return ERR_FAIL;

    int err = g_gpu_backend->upload_network(pnet);
    if (err == ERR_OK)
        pnet->gpu_stale = 0;
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
// Weights are on the GPU (dispatch checks layer 1's weights, which every backend uploads)
static int gpu_uploaded(const PNetwork pnet)
{
    return g_gpu_backend && pnet->layer_count > 1 &&
           pnet->layers[1].t_weights && pnet->layers[1].t_weights->gpu_buf;
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
static const int ANN_TEXT_FORMAT_VERSION = 1;
static const int ANN_BINARY_FORMAT_VERSION = 1;

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

	va_start(valist, format);
		vsprintf(buf, format, valist);
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
		int fan_in = pnet->layers[layer - 1].node_count;
		int fan_out = pnet->layers[layer].node_count;
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

		// Initialize weights based on strategy
		switch (init_type)
		{
		case WEIGHT_INIT_HE:
			// He initialization: std = sqrt(2 / fan_in)
			{
				real std = (real)sqrt(2.0 / fan_in);
				tensor_random_normal(pnet->layers[layer].t_weights, (real)0.0, std);
				tensor_fill(pnet->layers[layer].t_bias, (real)0.0);
			}
			break;

		case WEIGHT_INIT_XAVIER:
			// Xavier/Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
			{
				real std = (real)sqrt(2.0 / (fan_in + fan_out));
				tensor_random_normal(pnet->layers[layer].t_weights, (real)0.0, std);
				tensor_fill(pnet->layers[layer].t_bias, (real)0.0);
			}
			break;

		default:
		case WEIGHT_INIT_UNIFORM:
			// Original uniform initialization
			tensor_random_uniform(pnet->layers[layer].t_bias, (real)-pnet->weight_limit, (real)pnet->weight_limit);
			tensor_random_uniform(pnet->layers[layer].t_weights, (real)-pnet->weight_limit, (real)pnet->weight_limit);
			break;
		}
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

	puts("");

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
 * Pass NULL to disable error logging.
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
 * Clear the error logging callback.
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

		// y = Wx + b
		tensor_matvec(Tensor_NoTranspose, (real)1.0, cur->t_weights, (real)0.0, pnet->layers[layer - 1].t_values, cur->t_values);
		tensor_add(cur->t_values, cur->t_bias);

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
// Batched forward propagation
// Input: layers[0].t_batch_values has input data [batch_size × input_nodes]
// Output: all layers have t_batch_values populated
// Also stores pre-activation values in t_batch_z for backprop derivatives
//---------------------------------------------
static void eval_network_batched(PNetwork pnet, int batch_size)
{
	if (!pnet)
		return;

	// Loop over non-input layers (each owns the weights from the layer below)
	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		PTensor X = pnet->layers[layer].t_batch_values;      // [batch × in_nodes]
		PTensor W = pnet->layers[layer + 1].t_weights;       // [out_nodes × in_nodes]
		PTensor Y = pnet->layers[layer + 1].t_batch_values;  // [batch × out_nodes]
		PTensor Z = pnet->layers[layer + 1].t_batch_z;       // [batch × out_nodes]
		PTensor bias = pnet->layers[layer + 1].t_bias;       // [out_nodes × 1]
		int out_nodes = pnet->layers[layer + 1].node_count;

		// Y = X * W^T (gemm with B transposed)
		// X is [batch × in], W is [out × in], Y is [batch × out]
		tensor_gemm_transB((real)1.0, X, W, (real)0.0, Y);

		// Fused bias add + pre-activation save + activation (single pass over Y)
		Activation_type act_type = pnet->layers[layer + 1].activation;
		int total = batch_size * out_nodes;

		// Add bias to each row and save pre-activation Z
		for (int b = 0; b < batch_size; b++)
		{
			int row_offset = b * out_nodes;
			for (int n = 0; n < out_nodes; n++)
			{
				Y->values[row_offset + n] += bias->values[n];
			}
		}
		memcpy(Z->values, Y->values, total * sizeof(real));

		// Inlined activation (avoids function pointer overhead, enables auto-vectorization)
		if (act_type != ACTIVATION_SOFTMAX)
		{
			switch (act_type)
			{
			case ACTIVATION_SIGMOID:
				for (int i = 0; i < total; i++)
					Y->values[i] = (real)1.0 / ((real)1.0 + (real)exp(-Y->values[i]));
				break;
			case ACTIVATION_RELU:
				for (int i = 0; i < total; i++)
					Y->values[i] = Y->values[i] > (real)0.0 ? Y->values[i] : (real)0.0;
				break;
			case ACTIVATION_LEAKY_RELU:
				for (int i = 0; i < total; i++)
					Y->values[i] = Y->values[i] > (real)0.0 ? Y->values[i] : (real)0.01 * Y->values[i];
				break;
			case ACTIVATION_TANH:
				for (int i = 0; i < total; i++)
					Y->values[i] = (real)tanh(Y->values[i]);
				break;
			case ACTIVATION_SOFTSIGN:
				for (int i = 0; i < total; i++)
					Y->values[i] = Y->values[i] / ((real)1.0 + (real)fabs(Y->values[i]));
				break;
			case ACTIVATION_GELU:
				for (int i = 0; i < total; i++)
					Y->values[i] = gelu(Y->values[i]);
				break;
			default:
				break;
			}
		}
		
		// Apply dropout to hidden layers (not output layer)
		if (layer + 1 < pnet->layer_count - 1)
		{
			apply_dropout_batched(pnet, layer + 1, batch_size);
		}
	}

	// Apply softmax on output if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax_batched(pnet, batch_size);
}

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

//-------------------------------------------
// Turn the MSE output error delta = T - Y into the gradient with respect to
// the output pre-activations, -dL/dz, by applying the output activation's
// derivative. Needed for MSE with a nonlinear output; cross-entropy with
// softmax already has -dL/dz = T - Y. Softmax uses its full Jacobian per row:
// delta_j = y_j * (d_j - sum_i d_i * y_i).
// Shared with the GPU backends (declared in ann_gpu_backend.h) so every path
// computes the same output gradient.
//-------------------------------------------
void ann_mse_output_delta(Activation_type act, real *delta, const real *Y, const real *Z,
                          int batch_size, int nodes)
{
	int total = batch_size * nodes;

	switch (act)
	{
	case ACTIVATION_SIGMOID:
		for (int i = 0; i < total; i++)
			delta[i] *= Y[i] * ((real)1.0 - Y[i]);
		break;

	case ACTIVATION_TANH:
		for (int i = 0; i < total; i++)
			delta[i] *= (real)1.0 - Y[i] * Y[i];
		break;

	case ACTIVATION_SOFTSIGN:
		for (int i = 0; i < total; i++)
		{
			real d = (real)1.0 - (real)fabs(Y[i]);
			delta[i] *= d * d;
		}
		break;

	case ACTIVATION_RELU:
		for (int i = 0; i < total; i++)
			if (Z[i] <= (real)0.0)
				delta[i] = (real)0.0;
		break;

	case ACTIVATION_LEAKY_RELU:
		for (int i = 0; i < total; i++)
			if (Z[i] <= (real)0.0)
				delta[i] *= (real)0.01;
		break;

	case ACTIVATION_GELU:
		for (int i = 0; i < total; i++)
			delta[i] *= gelu_derivative(Z[i]);
		break;

	case ACTIVATION_SOFTMAX:
		for (int b = 0; b < batch_size; b++)
		{
			real *d = delta + b * nodes;
			const real *y = Y + b * nodes;
			real dot = (real)0.0;
			for (int n = 0; n < nodes; n++)
				dot += d[n] * y[n];
			for (int n = 0; n < nodes; n++)
				d[n] = y[n] * (d[n] - dot);
		}
		break;

	default:
		// linear output: delta is already -dL/dz
		break;
	}
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
	
	// Gradient: dW += delta^T * A_prev
	// delta is [batch × out], A_prev is [batch × in]
	// delta^T is [out × batch], result is [out × in]
	tensor_gemm_transA((real)1.0, delta, A_prev, (real)1.0, layer->t_gradients);
	
	// Bias gradient: sum of delta columns
	accumulate_bias_gradient(layer->t_bias_grad, delta, batch_size, out_nodes);
	
	// Propagate to previous layer: dl_dz_prev = delta * W
	// delta is [batch × out], W is [out × in]
	// Result is [batch × in]
	tensor_gemm((real)1.0, delta, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	
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
// Batched hidden layer backpropagation for sigmoid
// Applies sigmoid derivative: dL/dz = dL/da * a * (1 - a)
// Then computes gradient and propagates
//-------------------------------------------
static void back_propagate_sigmoid_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;        // [batch × nodes]
	PTensor A = layer->t_batch_values;           // [batch × nodes] activations
	PTensor A_prev = prev_layer->t_batch_values; // [batch × prev_nodes]
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply sigmoid derivative: dL/dz *= a * (1 - a)
	for (int b = 0; b < batch_size; b++)
	{
		int offset = b * nodes;
		for (int n = 0; n < nodes; n++)
		{
			real a = A->values[offset + n];
			real deriv = a * ((real)1.0 - a);
			dl_dz->values[offset + n] *= deriv;
		}
	}
	
	// Gradient: dW += dl_dz^T * A_prev
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	// Bias gradient: sum of dl_dz columns
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	// Propagate: dl_dz_prev = dl_dz * W
	if (layer_idx > 1)  // Don't propagate to input layer
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation for ReLU
//-------------------------------------------
static void back_propagate_relu_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;
	PTensor Z = layer->t_batch_z;  // Pre-activation values
	PTensor A_prev = prev_layer->t_batch_values;
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply ReLU derivative: dL/dz *= (z > 0 ? 1 : 0)
	for (int b = 0; b < batch_size; b++)
	{
		int offset = b * nodes;
		for (int n = 0; n < nodes; n++)
		{
			if (Z->values[offset + n] <= (real)0.0)
				dl_dz->values[offset + n] = (real)0.0;
		}
	}
	
	// Gradient: dW += dl_dz^T * A_prev
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	// Bias gradient
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	// Propagate
	if (layer_idx > 1)
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation for Leaky ReLU
//-------------------------------------------
static void back_propagate_leaky_relu_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;
	PTensor Z = layer->t_batch_z;
	PTensor A_prev = prev_layer->t_batch_values;
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply Leaky ReLU derivative: dL/dz *= (z > 0 ? 1 : 0.01)
	for (int b = 0; b < batch_size; b++)
	{
		int offset = b * nodes;
		for (int n = 0; n < nodes; n++)
		{
			real deriv = Z->values[offset + n] > (real)0.0 ? (real)1.0 : (real)0.01;
			dl_dz->values[offset + n] *= deriv;
		}
	}
	
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	if (layer_idx > 1)
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation for Tanh
//-------------------------------------------
static void back_propagate_tanh_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;
	PTensor A = layer->t_batch_values;  // tanh output
	PTensor A_prev = prev_layer->t_batch_values;
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply tanh derivative: dL/dz *= (1 - a^2)
	for (int b = 0; b < batch_size; b++)
	{
		int offset = b * nodes;
		for (int n = 0; n < nodes; n++)
		{
			real a = A->values[offset + n];
			real deriv = (real)1.0 - a * a;
			dl_dz->values[offset + n] *= deriv;
		}
	}
	
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	if (layer_idx > 1)
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation for Softsign
//-------------------------------------------
static void back_propagate_softsign_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;
	PTensor A = layer->t_batch_values;  // softsign output
	PTensor A_prev = prev_layer->t_batch_values;
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply softsign derivative: dL/dz *= (1 - |a|)^2
	for (int b = 0; b < batch_size; b++)
	{
		int offset = b * nodes;
		for (int n = 0; n < nodes; n++)
		{
			real a = A->values[offset + n];
			real one_minus_abs = (real)1.0 - (real)fabs(a);
			real deriv = one_minus_abs * one_minus_abs;
			dl_dz->values[offset + n] *= deriv;
		}
	}
	
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	if (layer_idx > 1)
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
}

//-------------------------------------------
// Batched hidden layer backpropagation for GELU
//-------------------------------------------
static void back_propagate_gelu_batched(PNetwork pnet, int batch_size, int layer_idx)
{
	PLayer layer = &pnet->layers[layer_idx];
	PLayer prev_layer = &pnet->layers[layer_idx - 1];
	
	int nodes = layer->node_count;
	
	PTensor dl_dz = layer->t_batch_dl_dz;
	PTensor Z = layer->t_batch_z;  // Pre-activation values
	PTensor A_prev = prev_layer->t_batch_values;
	
	// Apply dropout mask to gradient (if dropout was used in forward pass)
	apply_dropout_mask_to_gradient(layer, batch_size);
	
	// Apply GELU derivative: dL/dz *= gelu'(z)
	int total = batch_size * nodes;
	for (int i = 0; i < total; i++)
		dl_dz->values[i] *= gelu_derivative(Z->values[i]);
	
	tensor_gemm_transA((real)1.0, dl_dz, A_prev, (real)1.0, layer->t_gradients);
	
	accumulate_bias_gradient(layer->t_bias_grad, dl_dz, batch_size, nodes);
	
	if (layer_idx > 1)
	{
		tensor_gemm((real)1.0, dl_dz, layer->t_weights, (real)0.0, prev_layer->t_batch_dl_dz);
	}
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
	{
		Activation_type act = pnet->layers[layer].activation;
		
		switch (act)
		{
		case ACTIVATION_SIGMOID:
			back_propagate_sigmoid_batched(pnet, batch_size, layer);
			break;
		case ACTIVATION_RELU:
			back_propagate_relu_batched(pnet, batch_size, layer);
			break;
		case ACTIVATION_LEAKY_RELU:
			back_propagate_leaky_relu_batched(pnet, batch_size, layer);
			break;
		case ACTIVATION_TANH:
			back_propagate_tanh_batched(pnet, batch_size, layer);
			break;
		case ACTIVATION_SOFTSIGN:
			back_propagate_softsign_batched(pnet, batch_size, layer);
			break;
		case ACTIVATION_GELU:
			back_propagate_gelu_batched(pnet, batch_size, layer);
			break;
		default:
			// Default to sigmoid behavior
			back_propagate_sigmoid_batched(pnet, batch_size, layer);
			break;
		}
	}
	
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
		tensor_clip(pnet->layers[layer].t_gradients, -pnet->max_gradient, pnet->max_gradient);
		tensor_clip(pnet->layers[layer].t_bias_grad, -pnet->max_gradient, pnet->max_gradient);
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
		PTensor W = pnet->layers[layer].t_weights;
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
		// W = W + n * gradients
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_gradients, pnet->layers[layer].t_weights);

		// bias = bias + n * bias_grad
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_bias_grad, pnet->layers[layer].t_bias);
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
		// momentum = beta * m + one_minus_beta * gradients
		tensor_axpby(one_minus_beta, pnet->layers[layer].t_gradients, beta, pnet->layers[layer].t_m);

		// W = W + n * m
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_m, pnet->layers[layer].t_weights);

		// bias momentum = beta * bias_m + one_minus_beta * bias_grad
		tensor_axpby(one_minus_beta, pnet->layers[layer].t_bias_grad, beta, pnet->layers[layer].t_bias_m);

		// bias = bias + n * bias_m
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_bias_m, pnet->layers[layer].t_bias);
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
		PTensor g = pnet->layers[layer].t_gradients;
		PTensor v = pnet->layers[layer].t_v;
		PTensor w = pnet->layers[layer].t_weights;

		int size = g->rows * g->cols;

		// v = v + g^2, then W = W + lr * g / (sqrt(v) + epsilon)
		for (int i = 0; i < size; i++)
		{
			real grad = g->values[i];
			v->values[i] += grad * grad;
			w->values[i] += pnet->learning_rate * grad / ((real)sqrt(v->values[i]) + epsilon);
		}

		// Update biases with AdaGrad
		PTensor bg = pnet->layers[layer].t_bias_grad;
		PTensor bv = pnet->layers[layer].t_bias_v;
		PTensor b = pnet->layers[layer].t_bias;

		int bias_size = bg->cols;

		for (int i = 0; i < bias_size; i++)
		{
			real grad = bg->values[i];
			bv->values[i] += grad * grad;
			b->values[i] += pnet->learning_rate * grad / ((real)sqrt(bv->values[i]) + epsilon);
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
		PTensor g = pnet->layers[layer].t_gradients;
		PTensor v = pnet->layers[layer].t_v;
		PTensor w = pnet->layers[layer].t_weights;

		int size = g->rows * g->cols;

		// v = beta * v + (1 - beta) * g^2
		// W = W + lr * g / (sqrt(v) + epsilon)
		for (int i = 0; i < size; i++)
		{
			real grad = g->values[i];
			v->values[i] = beta * v->values[i] + one_minus_beta * grad * grad;
			w->values[i] += pnet->learning_rate * grad / ((real)sqrt(v->values[i]) + epsilon);
		}

		// Update biases with RMSProp
		PTensor bg = pnet->layers[layer].t_bias_grad;
		PTensor bv = pnet->layers[layer].t_bias_v;
		PTensor b = pnet->layers[layer].t_bias;

		int bias_size = bg->cols;

		for (int i = 0; i < bias_size; i++)
		{
			real grad = bg->values[i];
			bv->values[i] = beta * bv->values[i] + one_minus_beta * grad * grad;
			b->values[i] += pnet->learning_rate * grad / ((real)sqrt(bv->values[i]) + epsilon);
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
		PTensor g = pnet->layers[layer].t_gradients;
		PTensor m = pnet->layers[layer].t_m;
		PTensor v = pnet->layers[layer].t_v;
		PTensor w = pnet->layers[layer].t_weights;

		int size = g->rows * g->cols;

		for (int i = 0; i < size; i++)
		{
			real grad = g->values[i];

			// Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
			m->values[i] = beta1 * m->values[i] + ((real)1.0 - beta1) * grad;

			// Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
			v->values[i] = beta2 * v->values[i] + ((real)1.0 - beta2) * grad * grad;

			// Compute bias-corrected estimates
			real mhat = m->values[i] * bias_correction1;
			real vhat = v->values[i] * bias_correction2;

			// Update weights: W = W + lr * mhat / (sqrt(vhat) + epsilon)
			w->values[i] += pnet->learning_rate * mhat / ((real)sqrt(vhat) + epsilon);
		}

		// Update biases with Adam
		PTensor bg = pnet->layers[layer].t_bias_grad;
		PTensor bm = pnet->layers[layer].t_bias_m;
		PTensor bv = pnet->layers[layer].t_bias_v;
		PTensor b = pnet->layers[layer].t_bias;

		int bias_size = bg->cols;

		for (int i = 0; i < bias_size; i++)
		{
			real grad = bg->values[i];

			// Update biased first moment estimate: bm = beta1 * bm + (1 - beta1) * g
			bm->values[i] = beta1 * bm->values[i] + ((real)1.0 - beta1) * grad;

			// Update biased second moment estimate: bv = beta2 * bv + (1 - beta2) * g^2
			bv->values[i] = beta2 * bv->values[i] + ((real)1.0 - beta2) * grad * grad;

			// Compute bias-corrected estimates
			real mhat = bm->values[i] * bias_correction1;
			real vhat = bv->values[i] * bias_correction2;

			// Update biases: b = b + lr * mhat / (sqrt(vhat) + epsilon)
			b->values[i] += pnet->learning_rate * mhat / ((real)sqrt(vhat) + epsilon);
		}
	}

	apply_regularization(pnet);
}

//------------------------------
// free a layer's parameters, gradients and optimizer state
//------------------------------
static void free_layer_params(PLayer layer)
{
	PTensor *params[] = {
		&layer->t_weights, &layer->t_gradients, &layer->t_m, &layer->t_v,
		&layer->t_bias, &layer->t_bias_grad, &layer->t_bias_m, &layer->t_bias_v
	};

	for (size_t i = 0; i < sizeof(params) / sizeof(params[0]); i++)
	{
		tensor_free(*params[i]);
		*params[i] = NULL;
	}
}

//------------------------------
// allocate a dense layer's weights [nodes x in_nodes], bias [1 x nodes],
// their gradients and optimizer state; frees everything on failure
//------------------------------
static int alloc_dense_params(PLayer layer, int in_nodes)
{
	int nodes = layer->node_count;

	layer->t_weights	= tensor_zeros(nodes, in_nodes);
	layer->t_gradients	= tensor_zeros(nodes, in_nodes);
	layer->t_m			= tensor_zeros(nodes, in_nodes);
	layer->t_v			= tensor_zeros(nodes, in_nodes);
	layer->t_bias		= tensor_zeros(1, nodes);
	layer->t_bias_grad	= tensor_zeros(1, nodes);
	layer->t_bias_m		= tensor_zeros(1, nodes);
	layer->t_bias_v		= tensor_zeros(1, nodes);

	if (!layer->t_weights || !layer->t_gradients || !layer->t_m || !layer->t_v ||
		!layer->t_bias || !layer->t_bias_grad || !layer->t_bias_m || !layer->t_bias_v)
	{
		free_layer_params(layer);
		return ERR_ALLOC;
	}

	return ERR_OK;
}

//[]---------------------------------------------[]
//	Public ANN library interfaces
//[]---------------------------------------------[]

//------------------------------
// add a new layer to the network
//------------------------------
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type)
{
	if (!pnet) {
		invoke_error_callback(ERR_NULL_PTR, "ann_add_layer");
		return ERR_NULL_PTR;
	}

	if (node_count <= 0) {
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

	// create the parameters that produce this layer from the one below
	if (cur_layer > 0 && alloc_dense_params(&pnet->layers[cur_layer], pnet->layers[cur_layer - 1].node_count) != ERR_OK)
	{
		tensor_free(pnet->layers[cur_layer].t_values);
		pnet->layers[cur_layer].t_values = NULL;
		pnet->layer_count--;
		invoke_error_callback(ERR_ALLOC, "ann_add_layer");
		return ERR_ALLOC;
	}

	return ERR_OK;
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
// Copy every layer's weights and biases to or from one flat buffer, used to
// keep the best epoch's parameters for early stopping. Pass NULL as buf to
// get the number of reals needed.
//-----------------------------------------------
static size_t copy_parameters(PNetwork pnet, real *buf, int restore)
{
	size_t offset = 0;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		PTensor params[2];
		params[0] = pnet->layers[layer].t_weights;
		params[1] = pnet->layers[layer].t_bias;

		for (int i = 0; i < 2; i++)
		{
			size_t count = (size_t)params[i]->rows * params[i]->cols;
			if (buf && restore)
				memcpy(params[i]->values, buf + offset, count * sizeof(real));
			else if (buf)
				memcpy(buf + offset, params[i]->values, count * sizeof(real));
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
			// zero the gradients
			for (int layer = 1; layer < pnet->layer_count; layer++)
			{
				tensor_fill(pnet->layers[layer].t_gradients, (real)0.0);
				tensor_fill(pnet->layers[layer].t_bias_grad, (real)0.0);
			}

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
				// CPU path: forward pass
				eval_network_batched(pnet, actual_batch_size);

				// Batched backward pass (also computes loss)
				loss = back_propagate_batched(pnet, actual_batch_size, batch_targets);

				// Increment training iteration for Adam bias correction
				pnet->train_iteration++;

				// update weights based on batched gradients
				// using the chosen optimization function
				pnet->optimize_func(pnet);
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

	// Zero gradients
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		tensor_fill(pnet->layers[layer].t_gradients, (real)0.0);
		tensor_fill(pnet->layers[layer].t_bias_grad, (real)0.0);
	}

	// Copy inputs into batch input tensor
	PTensor batch_input = pnet->layers[0].t_batch_values;
	memcpy(batch_input->values, inputs, actual_batch_size * input_node_count * sizeof(real));

	// Copy targets into batch target tensor
	memcpy(batch_targets->values, targets, actual_batch_size * output_node_count * sizeof(real));

	// Forward pass
	eval_network_batched(pnet, actual_batch_size);

	// Backward pass (computes loss and gradients)
	real loss = back_propagate_batched(pnet, actual_batch_size, batch_targets);

	// Increment training iteration (for Adam bias correction)
	pnet->train_iteration++;

	// Update weights
	pnet->optimize_func(pnet);

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
	real *pred = alloca(classes * sizeof(real));
	int pred_class, act_class;

	for (int i = 0; i < inputs->rows; i++)
	{
		ann_predict(pnet, &inputs->values[i * inputs->cols], pred);
		pred_class = ann_class_prediction(pred, classes);
		act_class = ann_class_prediction(&outputs->values[i * outputs->cols], classes);

		if (pred_class == act_class)
			correct++;
	}

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
	
	int true_pos = 0, false_pos = 0, true_neg = 0, false_neg = 0;
	real pred[2];
	
	for (int i = 0; i < inputs->rows; i++)
	{
		ann_predict(pnet, &inputs->values[i * inputs->cols], pred);
		int pred_class = pred[1] > pred[0] ? 1 : 0;
		int actual_class = outputs->values[i * 2 + 1] > outputs->values[i * 2] ? 1 : 0;
		
		if (actual_class == 1 && pred_class == 1)
			true_pos++;
		else if (actual_class == 0 && pred_class == 1)
			false_pos++;
		else if (actual_class == 0 && pred_class == 0)
			true_neg++;
		else
			false_neg++;
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
	
	// Apply to all existing hidden layers that don't have a custom rate
	for (int i = 1; i < pnet->layer_count - 1; i++)
	{
		if (pnet->layers[i].dropout_rate == (real)0.0)
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
// GPU batch inference (ann_predict_batch)
//
// Runs forward pass on `batch_size` samples in parallel on the GPU.
// Falls back to sequential CPU inference if Metal is not available.
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

	// CPU fallback: sequential single-sample prediction
	int in_nodes  = pnet->layers[0].node_count;
	int out_nodes = pnet->layers[pnet->layer_count - 1].node_count;

	for (int i = 0; i < batch_size; i++)
	{
		int err = ann_predict(pnet, inputs + i * in_nodes, outputs + i * out_nodes);
		if (err != ERR_OK)
			return err;
	}

	return ERR_OK;
}

//-----------------------------------
// save network to a binary file
//-----------------------------------
int ann_save_network_binary(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_save_network_binary");
		return ERR_NULL_PTR;
	}

	FILE *fptr = fopen(filename, "wb");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_save_network_binary");
		return ERR_IO;
	}

	// write binary version
	fwrite (&ANN_BINARY_FORMAT_VERSION, sizeof(int), 1, fptr);

	// save out network
	// save optimizer
	fwrite(&pnet->optimizer, sizeof(int), 1, fptr);

	// save loss
	fwrite(&pnet->loss_type, sizeof(int), 1, fptr);

	// save network props
	fwrite(&pnet->layer_count, sizeof(int), 1, fptr);

	// save layer details
	int val;
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// node count
		val = pnet->layers[layer].node_count;
		fwrite(&val, sizeof(val), 1, fptr);

		// layer type
		val = pnet->layers[layer].layer_type;
		fwrite(&val, sizeof(val), 1, fptr);

		// activation type
		val = pnet->layers[layer].activation;
		fwrite(&val, sizeof(val), 1, fptr);
	}

	real w;
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// save bias vector
		for (int element = 0; element < pnet->layers[layer].t_bias->cols; element++)
		{
			w = pnet->layers[layer].t_bias->values[element];
			fwrite(&w, sizeof(w), 1, fptr);
		}

		// save node weights
		int limit = pnet->layers[layer].t_weights->cols * pnet->layers[layer].t_weights->rows;
		for (int element = 0; element < limit; element++)
		{
			w = pnet->layers[layer].t_weights->values[element];
			fwrite(&w, sizeof(w), 1, fptr);
		}
	}

	fclose(fptr);
	return ERR_OK;
}

//------------------------------
// read network from binary file
//------------------------------
PNetwork ann_load_network_binary(const char *filename)
{
	if (!filename)
		return NULL;

	FILE *fptr = fopen(filename, "rb");
	if (!fptr)
		return NULL;

	// load version
	int ver;
	CHECK_RESULT(fread(&ver, sizeof(ver), 1, fptr), 1, NULL);
	if (ver != ANN_BINARY_FORMAT_VERSION)
	{
		printf("Incompatible version, was %d, expected %d\n", ver, ANN_TEXT_FORMAT_VERSION);
		fclose(fptr);
		return NULL;
	}

	// load network
	int optimizer, loss_type, layer_count, node_count, layer_type, activation;
	CHECK_RESULT(fread(&optimizer, sizeof(optimizer), 1, fptr), 1, NULL);
	CHECK_RESULT(fread(&loss_type, sizeof(loss_type), 1, fptr), 1, NULL);
	CHECK_RESULT(fread(&layer_count, sizeof(layer_count), 1, fptr), 1, NULL);

	PNetwork pnet = ann_make_network(optimizer, loss_type);
	if (!pnet)
	{
		fclose(fptr);
		return NULL;
	}

	ann_printf(pnet, "%sloading network%s %s%s%s...", 
		COLOR(ANSI_CYAN), RESET(),
		COLOR(ANSI_WHITE), filename, RESET());

	// create layers
	for (int layer = 0; layer < layer_count; layer++)
	{
		CHECK_RESULT(fread(&node_count, sizeof(node_count), 1, fptr), 1, NULL);
		CHECK_RESULT(fread(&layer_type, sizeof(layer_type), 1, fptr), 1, NULL);
		CHECK_RESULT(fread(&activation, sizeof(activation), 1, fptr), 1, NULL);

		ann_add_layer(pnet, node_count, layer_type, activation);
	}

	for (int layer = 1; layer < layer_count; layer++)
	{
		// read bias vector
		for (int element = 0; element < pnet->layers[layer].t_bias->cols; element++)
		{
			CHECK_RESULT(fread(&pnet->layers[layer].t_bias->values[element], sizeof(real), 1, fptr), 1, NULL);
		}

		// read node weights
		int limit = pnet->layers[layer].t_weights->cols * pnet->layers[layer].t_weights->rows;
		for (int element = 0; element < limit; element++)
		{
			CHECK_RESULT(fread(&pnet->layers[layer].t_weights->values[element], sizeof(real), 1, fptr), 1, NULL);
		}
	}

	ann_printf(pnet, "%sdone.%s\n", COLOR(ANSI_GREEN), RESET());

	fclose(fptr);

	// mark network weights as set
	pnet->weights_set = 1;

	return pnet;
}

//------------------------------
// save network to a text file
//------------------------------
int ann_save_network(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
	{
		invoke_error_callback(ERR_NULL_PTR, "ann_save_network");
		return ERR_NULL_PTR;
	}

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_save_network");
		return ERR_IO;
	}

	// save out version
	fprintf(fptr, "%d\n", ANN_TEXT_FORMAT_VERSION);

	// save out network
	// save optimizer
	fprintf(fptr, "%d\n", pnet->optimizer);

	// save loss
	fprintf(fptr, "%d\n", pnet->loss_type);

	// save network props
	fprintf(fptr, "%d\n", pnet->layer_count);

	// save layer details
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// node count
		fprintf(fptr, "%d\n", pnet->layers[layer].node_count);

		// layer type
		fprintf(fptr, "%d\n", pnet->layers[layer].layer_type);

		// activation type
		fprintf(fptr, "%d\n", pnet->layers[layer].activation);
	}

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// save bias vector
		for (int element = 0; element < pnet->layers[layer].t_bias->cols; element++)
		{
			fprintf(fptr, "%f\n", pnet->layers[layer].t_bias->values[element]);
		}

		// save node weights
		int limit = pnet->layers[layer].t_weights->cols * pnet->layers[layer].t_weights->rows;
		for (int element = 0; element < limit; element++)
		{
			fprintf(fptr, "%f\n", pnet->layers[layer].t_weights->values[element]);			
		}
	}

	fclose(fptr);
	return ERR_OK;
}

//--------------------------------
// load a previously saved network
//--------------------------------
PNetwork ann_load_network(const char *filename)
{
	FILE *fptr = fopen(filename, "rt");
	if (!fptr)
		return NULL;

	int ver;
	CHECK_RESULT(fscanf(fptr, "%d", &ver), 1, NULL);
	if (ver != ANN_TEXT_FORMAT_VERSION)
	{
		printf("Incompatible version, was %d, expected %d\n", ver, ANN_TEXT_FORMAT_VERSION);
		fclose(fptr);
		return NULL;
	}

	// load network
	int optimizer, loss_type, layer_count, node_count, layer_type, activation;
	CHECK_RESULT(fscanf(fptr, "%d", &optimizer), 1, NULL);
	CHECK_RESULT(fscanf(fptr, "%d", &loss_type), 1, NULL);
	CHECK_RESULT(fscanf(fptr, "%d", &layer_count), 1, NULL);

	PNetwork pnet = ann_make_network(optimizer, loss_type);
	if (!pnet)
	{
		fclose(fptr);
		return NULL;
	}

	ann_printf(pnet, "%sloading network%s %s%s%s...", 
		COLOR(ANSI_CYAN), RESET(),
		COLOR(ANSI_WHITE), filename, RESET());

	// create layers
	for (int layer = 0; layer < layer_count; layer++)
	{
		CHECK_RESULT(fscanf(fptr, "%d", &node_count), 1, NULL);
		CHECK_RESULT(fscanf(fptr, "%d", &layer_type), 1, NULL);
		CHECK_RESULT(fscanf(fptr, "%d", &activation), 1, NULL);

		ann_add_layer(pnet, node_count, layer_type, activation);
	}

	for (int layer = 1; layer < layer_count; layer++)
	{
		// read bias vector
		for (int element = 0; element < pnet->layers[layer].t_bias->cols; element++)
		{
			CHECK_RESULT(fscanf(fptr, "%f", &pnet->layers[layer].t_bias->values[element]), 1, NULL);
		}

		// read node weights
		int limit = pnet->layers[layer].t_weights->cols * pnet->layers[layer].t_weights->rows;
		for (int element = 0; element < limit; element++)
		{
			CHECK_RESULT(fscanf(fptr, "%f", &pnet->layers[layer].t_weights->values[element]), 1, NULL);
		}
	}

	ann_printf(pnet, "%sdone.%s\n", COLOR(ANSI_GREEN), RESET());

	fclose(fptr);

	// mark network weights as set
	pnet->weights_set = 1;

	return pnet;
}

//-----------------------------------------------
// display the network properties
//-----------------------------------------------
void ann_print_props(const PNetwork pnet)
{
	ann_printf(pnet, "%s  Network shape:%s %s", COLOR(ANSI_DIM), RESET(), COLOR(ANSI_CYAN));

	for (int i = 0; i < pnet->layer_count; i++)
	{
		if (i != 0)
			ann_printf(pnet, "-");
		ann_printf(pnet, "%d", pnet->layers[i].node_count);
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
static void write_json_float_array(FILE *fptr, const real *values, int count)
{
	fprintf(fptr, "[");

	for (int i = 0; i < count; i++)
	{
		if (i > 0) fprintf(fptr, ", ");
		double v = (double)values[i];
		if (isnan(v) || isinf(v))
			fprintf(fptr, "0.0");
		else
			fprintf(fptr, "%.8g", v);
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

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
	{
		invoke_error_callback(ERR_IO, "ann_export_onnx");
		return ERR_IO;
	}

	int input_size = pnet->layers[0].node_count;
	int output_size = pnet->layers[pnet->layer_count - 1].node_count;

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
	fprintf(fptr, "  \"producer_name\": \"ann-library\",\n");
	fprintf(fptr, "  \"producer_version\": \"1.0\",\n");
	fprintf(fptr, "  \"graph\": {\n");
	fprintf(fptr, "    \"name\": \"ann_model\",\n");

	// ========== INITIALIZERS (weights and biases) ==========
	fprintf(fptr, "    \"initializer\": [\n");
	int first_init = 1;
	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		PLayer pL = &pnet->layers[layer];
		PLayer next = &pnet->layers[layer + 1];	// owns the weights from pL to next
		int next_nodes = next->node_count;
		int curr_nodes = pL->node_count;

		// Weights: shape is [curr_nodes, next_nodes] stored row-major
		// ONNX MatMul: input [batch, curr_nodes] @ weights [curr_nodes, next_nodes] -> [batch, next_nodes]
		if (!first_init) fprintf(fptr, ",\n");
		first_init = 0;

		fprintf(fptr, "      {\n");
		fprintf(fptr, "        \"name\": \"weight_%d\",\n", layer);
		fprintf(fptr, "        \"data_type\": 1,\n");  // 1 = FLOAT
		int w_dims[] = {curr_nodes, next_nodes};
		fprintf(fptr, "        \"dims\": ");
		write_json_int_array(fptr, w_dims, 2);
		fprintf(fptr, ",\n");
		fprintf(fptr, "        \"float_data\": ");
		write_json_float_array(fptr, next->t_weights->values, curr_nodes * next_nodes);
		fprintf(fptr, "\n");
		fprintf(fptr, "      }");

		// Bias: shape is [next_nodes]
		fprintf(fptr, ",\n");
		fprintf(fptr, "      {\n");
		fprintf(fptr, "        \"name\": \"bias_%d\",\n", layer);
		fprintf(fptr, "        \"data_type\": 1,\n");
		int b_dims[] = {next_nodes};
		fprintf(fptr, "        \"dims\": ");
		write_json_int_array(fptr, b_dims, 1);
		fprintf(fptr, ",\n");
		fprintf(fptr, "        \"float_data\": ");
		write_json_float_array(fptr, next->t_bias->values, next_nodes);
		fprintf(fptr, "\n");
		fprintf(fptr, "      }");
	}
	fprintf(fptr, "\n    ],\n");

	// ========== NODES (operations) ==========
	fprintf(fptr, "    \"node\": [\n");
	int first_node = 1;
	char prev_output[64];
	snprintf(prev_output, sizeof(prev_output), "input");

	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		char matmul_out[64], add_out[64], act_out[64];
		snprintf(matmul_out, sizeof(matmul_out), "matmul_%d_out", layer);
		snprintf(add_out, sizeof(add_out), "add_%d_out", layer);
		snprintf(act_out, sizeof(act_out), "layer_%d_out", layer);

		// Get activation for the NEXT layer (output of this transform)
		Activation_type next_act = pnet->layers[layer + 1].activation;
		const char *act_op = onnx_activation_op(next_act);

		// MatMul node: prev_output @ weight_N -> matmul_N_out
		if (!first_node) fprintf(fptr, ",\n");
		first_node = 0;

		fprintf(fptr, "      {\n");
		fprintf(fptr, "        \"op_type\": \"MatMul\",\n");
		fprintf(fptr, "        \"name\": \"matmul_%d\",\n", layer);
		fprintf(fptr, "        \"input\": [\"%s\", \"weight_%d\"],\n", prev_output, layer);
		fprintf(fptr, "        \"output\": [\"%s\"]\n", matmul_out);
		fprintf(fptr, "      }");

		// Add node: matmul_N_out + bias_N -> add_N_out
		fprintf(fptr, ",\n");
		fprintf(fptr, "      {\n");
		fprintf(fptr, "        \"op_type\": \"Add\",\n");
		fprintf(fptr, "        \"name\": \"add_%d\",\n", layer);
		fprintf(fptr, "        \"input\": [\"%s\", \"bias_%d\"],\n", matmul_out, layer);
		fprintf(fptr, "        \"output\": [\"%s\"]\n", add_out);
		fprintf(fptr, "      }");

		// Activation node (if any)
		if (act_op != NULL)
		{
			fprintf(fptr, ",\n");
			fprintf(fptr, "      {\n");
			fprintf(fptr, "        \"op_type\": \"%s\",\n", act_op);
			fprintf(fptr, "        \"name\": \"activation_%d\",\n", layer);
			fprintf(fptr, "        \"input\": [\"%s\"],\n", add_out);
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
			// No activation, next input is add output
			snprintf(prev_output, sizeof(prev_output), "%s", add_out);
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
	fprintf(fptr, "            \"shape\": {\"dim\": [{\"dim_param\": \"batch\"}, {\"dim_value\": %d}]}\n", input_size);
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
	fprintf(fptr, "            \"shape\": {\"dim\": [{\"dim_param\": \"batch\"}, {\"dim_value\": %d}]}\n", output_size);
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

	// Get graph
	JsonValue *graph = json_get(&root, "graph");
	if (!graph || graph->type != JSON_OBJECT)
	{
		invoke_error_callback(ERR_INVALID, "ann_import_onnx: missing graph");
		goto cleanup;
	}

	// Get initializers (weights and biases)
	JsonValue *initializer = json_get(graph, "initializer");
	if (!initializer || initializer->type != JSON_ARRAY)
	{
		invoke_error_callback(ERR_INVALID, "ann_import_onnx: missing initializer");
		goto cleanup;
	}

	// Count layers by counting weight tensors (weight_0, weight_1, ...)
	size_t num_weights = 0;
	for (size_t i = 0; i < json_array_length(initializer); i++)
	{
		JsonValue *init = json_at(initializer, i);
		const char *name = json_string(json_get(init, "name"));
		if (name && strncmp(name, "weight_", 7) == 0)
			num_weights++;
	}

	if (num_weights == 0)
	{
		invoke_error_callback(ERR_INVALID, "ann_import_onnx: no weights found");
		goto cleanup;
	}

	// Get layer sizes from weight dimensions
	// weight_i has dims [layer_i_nodes, layer_i+1_nodes]
	int *layer_sizes = (int *)malloc((num_weights + 1) * sizeof(int));
	if (!layer_sizes)
	{
		invoke_error_callback(ERR_ALLOC, "ann_import_onnx");
		goto cleanup;
	}

	// Extract layer sizes from weight tensors
	for (size_t layer = 0; layer < num_weights; layer++)
	{
		char weight_name[32];
		snprintf(weight_name, sizeof(weight_name), "weight_%zu", layer);

		// Find this weight tensor
		JsonValue *weight_init = NULL;
		for (size_t i = 0; i < json_array_length(initializer); i++)
		{
			JsonValue *init = json_at(initializer, i);
			const char *name = json_string(json_get(init, "name"));
			if (name && strcmp(name, weight_name) == 0)
			{
				weight_init = init;
				break;
			}
		}

		if (!weight_init)
		{
			invoke_error_callback(ERR_INVALID, "ann_import_onnx: missing weight tensor");
			free(layer_sizes);
			goto cleanup;
		}

		JsonValue *dims = json_get(weight_init, "dims");
		if (!dims || json_array_length(dims) != 2)
		{
			invoke_error_callback(ERR_INVALID, "ann_import_onnx: invalid weight dims");
			free(layer_sizes);
			goto cleanup;
		}

		int curr_nodes, next_nodes;
		json_int(json_at(dims, 0), &curr_nodes);
		json_int(json_at(dims, 1), &next_nodes);

		layer_sizes[layer] = curr_nodes;
		if (layer == num_weights - 1)
			layer_sizes[layer + 1] = next_nodes;
	}

	// Get nodes to determine activations
	JsonValue *nodes = json_get(graph, "node");
	if (!nodes || nodes->type != JSON_ARRAY)
	{
		invoke_error_callback(ERR_INVALID, "ann_import_onnx: missing nodes");
		free(layer_sizes);
		goto cleanup;
	}

	// Build activation list for each layer (except input)
	Activation_type *activations = (Activation_type *)malloc((num_weights + 1) * sizeof(Activation_type));
	if (!activations)
	{
		invoke_error_callback(ERR_ALLOC, "ann_import_onnx");
		free(layer_sizes);
		goto cleanup;
	}

	// Initialize activations to NULL (input layer always NULL)
	activations[0] = ACTIVATION_NULL;
	for (size_t i = 1; i <= num_weights; i++)
		activations[i] = ACTIVATION_NULL;

	// Parse nodes to find activations
	// Nodes are: MatMul, Add, [Activation] for each layer
	for (size_t i = 0; i < json_array_length(nodes); i++)
	{
		JsonValue *node = json_at(nodes, i);
		const char *op_type = json_string(json_get(node, "op_type"));
		const char *name = json_string(json_get(node, "name"));

		if (!op_type) continue;

		// Skip MatMul and Add operations
		if (strcmp(op_type, "MatMul") == 0 || strcmp(op_type, "Add") == 0)
			continue;

		// Check for unsupported operations
		Activation_type act = onnx_op_to_activation(op_type);
		if (act == ACTIVATION_NULL && 
		    strcmp(op_type, "MatMul") != 0 && 
		    strcmp(op_type, "Add") != 0)
		{
			// Unsupported operation
			invoke_error_callback(ERR_INVALID, "ann_import_onnx: unsupported op");
			free(layer_sizes);
			free(activations);
			goto cleanup;
		}

		// Extract layer index from activation name (e.g., "activation_0")
		if (name && strncmp(name, "activation_", 11) == 0)
		{
			int layer_idx = atoi(name + 11);
			if (layer_idx >= 0 && layer_idx < (int)num_weights)
			{
				activations[layer_idx + 1] = act;
			}
		}
	}

	// Create network
	pnet = ann_make_network(OPT_ADAM, LOSS_MSE);
	if (!pnet)
	{
		free(layer_sizes);
		free(activations);
		goto cleanup;
	}

	// Add layers
	ann_add_layer(pnet, layer_sizes[0], LAYER_INPUT, ACTIVATION_NULL);
	for (size_t i = 1; i < num_weights; i++)
	{
		ann_add_layer(pnet, layer_sizes[i], LAYER_HIDDEN, activations[i]);
	}
	ann_add_layer(pnet, layer_sizes[num_weights], LAYER_OUTPUT, activations[num_weights]);

	// Load weights and biases from initializers
	for (size_t layer = 0; layer < num_weights; layer++)
	{
		char weight_name[32], bias_name[32];
		snprintf(weight_name, sizeof(weight_name), "weight_%zu", layer);
		snprintf(bias_name, sizeof(bias_name), "bias_%zu", layer);

		// Find and load weights
		for (size_t i = 0; i < json_array_length(initializer); i++)
		{
			JsonValue *init = json_at(initializer, i);
			const char *name = json_string(json_get(init, "name"));

			if (name && strcmp(name, weight_name) == 0)
			{
				JsonValue *float_data = json_get(init, "float_data");
				if (float_data && float_data->type == JSON_ARRAY)
				{
					PTensor weights = pnet->layers[layer + 1].t_weights;	// weight_N produces layer N+1
					size_t count = json_array_length(float_data);
					for (size_t j = 0; j < count && j < (size_t)(weights->rows * weights->cols); j++)
					{
						double val;
						json_number(json_at(float_data, j), &val);
						weights->values[j] = (real)val;
					}
				}
			}
			else if (name && strcmp(name, bias_name) == 0)
			{
				JsonValue *float_data = json_get(init, "float_data");
				if (float_data && float_data->type == JSON_ARRAY)
				{
					PTensor bias = pnet->layers[layer + 1].t_bias;
					size_t count = json_array_length(float_data);
					for (size_t j = 0; j < count && j < (size_t)bias->cols; j++)
					{
						double val;
						json_number(json_at(float_data, j), &val);
						bias->values[j] = (real)val;
					}
				}
			}
		}
	}

	// Mark network weights as set
	pnet->weights_set = 1;
	
	free(layer_sizes);
	free(activations);
	json_free(&root);
	return pnet;

cleanup:
	json_free(&root);
	if (pnet) ann_free_network(pnet);
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
	int detailed_mode = (max_nodes <= 10);

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
			const char *layer_type = (layer == 0) ? "Input" : 
			                         (layer == pnet->layer_count - 1) ? "Output" : "Hidden";
			int nodes = pnet->layers[layer].node_count;
			const char *act_name = get_activation_name(pnet->layers[layer].activation);
			
			if (act_name)
				fprintf(fptr, "box \"%s\" \"%d nodes\" \"(%s)\"\n", 
					layer_type, nodes, act_name);
			else
				fprintf(fptr, "box \"%s\" \"%d nodes\"\n", layer_type, nodes);
			
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
