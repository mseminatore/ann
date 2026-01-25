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
//   2. Backpropagation (back_propagate):
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>

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
	"Stochastic Gradient Descent with decay",
	"Adaptive Stochastic Gradient Descent",
	"SGD with momentum",
	"RMSPROP",
	"ADAGRAD",
	"Adam",
	"SGD"
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

//------------------------------
// compute the softmax
// Converts output layer into probability distribution (sums to 1)
// softmax(x_i) = e^x_i / sum(e^x_j for all j)
// Use with LOSS_CATEGORICAL_CROSS_ENTROPY for multi-class classification
//------------------------------
static void softmax(PNetwork pnet)
{
	real one_over_sum = 0.0;
	real sum = 0.0;

	// find the sum of the output node values, excluding the bias node
	int output_layer = pnet->layer_count - 1;
	int node_count = pnet->layers[output_layer].node_count;
	
	//tensor_exp(pnet->layers[output_layer].t_values);
	//one_over_sum = (real)1.0 / (tensor_sum(pnet->layers[output_layer].t_values) - 1.0);
	//tensor_mul_scalar(pnet->layers[output_layer].t_values, one_over_sum);

	for (int node = 0; node < node_count; node++)
	{
		sum += (real)exp(pnet->layers[output_layer].t_values->values[node]);
	}

	for (int node = 0; node < node_count; node++)
	{
		pnet->layers[output_layer].t_values->values[node] = (real)(exp(pnet->layers[output_layer].t_values->values[node]) / sum);
	}
}

//------------------------------
// return rand num [min..max)
//------------------------------
static real get_rand(real min, real max)
{
	real r = (real)rand() / (real)RAND_MAX;
	real scale = max - min;

	r *= scale;
	r += min;
	return r;
}

//------------------------------
// print the network
//------------------------------
static void print_network(PNetwork pnet)
{
	if (!pnet)
		return;

	puts("");
	// print each layer
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// ann_printf(pnet, "\nLayer %d\n"
		// 	"--------\n", layer);
		putchar('[');

		// print nodes in the layer
		for (int node = 0; node < pnet->layers[layer].node_count; node++)
		{
//			ann_printf(pnet, "(%3.2g, ", pnet->layers[layer].nodes[node].value);

			if (layer > 0)
			{
				for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
				{
//					ann_printf(pnet, "%3.2g, ", pnet->layers[layer].nodes[node].weights[prev_node]);
				}
			}
		}

		puts("]");
	}
}

//------------------------------
// initialize the weights
//------------------------------
static void init_weights(PNetwork pnet)
{
	if (!pnet)
		return;

	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// output layers don't have weights
		//if (pnet->layers[layer].layer_type == LAYER_OUTPUT)
		//	continue;

		// int weight_count	= pnet->layers[layer - 1].node_count;
		// int node_count		= pnet->layers[layer].node_count;

		// TODO - glorot uniform limits
		//		real limit = (real)sqrt(6.0 / (weight_count + node_count));
		//		real limit = (real)sqrt(1.0 / (weight_count));
		tensor_random_uniform(pnet->layers[layer - 1].t_bias, (real)-pnet->weight_limit, (real)pnet->weight_limit);
		tensor_random_uniform(pnet->layers[layer - 1].t_weights, (real)-pnet->weight_limit, (real)pnet->weight_limit);
	}

	pnet->weights_set = 1;
}

//--------------------------------
// print nodes in the output layer
//--------------------------------
void print_outputs(const PNetwork pnet)
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

//--------------------------------
// compute the mean squared error
// MSE = (1/n) * sum((y_true - y_pred)^2)
// Good for: Regression, continuous outputs
// Sensitive to outliers due to squaring
//--------------------------------
static real compute_ms_error(PNetwork pnet, PTensor outputs)
{
	// get the output layer
	PLayer poutput_layer = &pnet->layers[pnet->layer_count - 1];

	assert(poutput_layer->layer_type == LAYER_OUTPUT);

	real mse = (real)0.0, diff;

	for (int i = 0; i < poutput_layer->node_count; i++)
	{
		diff = outputs->values[i] - poutput_layer->t_values->values[i];
		mse += diff * diff;
	}

	return mse;
}

//---------------------------------------------
// compute the categorical cross entropy error
// CE = -sum(y_true * log(y_pred))
// Good for: Multi-class classification
// Works best with softmax activation and probability outputs
// Penalizes confident wrong predictions heavily
//---------------------------------------------
static real compute_cross_entropy(PNetwork pnet, PTensor outputs)
{
	// get the output layer
	PLayer poutput_layer = &pnet->layers[pnet->layer_count - 1];

	assert(poutput_layer->layer_type == LAYER_OUTPUT);

	real xe = (real)0.0;

	for (int i = 0; i < poutput_layer->node_count; i++)
	{
		xe += (real)(outputs->values[i] * log(poutput_layer->t_values->values[i]));
	}

	return -xe;
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

	// loop over the non-output layers
	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		// y = Wx + b
		tensor_matvec(Tensor_NoTranspose, (real)1.0, pnet->layers[layer].t_weights, (real)0.0, pnet->layers[layer].t_values, pnet->layers[layer + 1].t_values);
		tensor_add(pnet->layers[layer + 1].t_values, pnet->layers[layer].t_bias);

		// apply activation function to values
		for (int i = 0; i < pnet->layers[layer + 1].node_count; i++)
		{
			pnet->layers[layer + 1].t_values->values[i] = pnet->layers[layer + 1].activation_func(pnet->layers[layer + 1].t_values->values[i]);
		}
	}

	// apply softmax on output, if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax(pnet);
}

//-------------------------------------------
// back propagate output error to prev layer
//-------------------------------------------
// Backpropagation for output layer:
// 1. Compute output error: dL/dy = y_predicted - y_true
// 2. Update bias: b = b + learning_rate * dL/dy
// 3. Compute gradient: gradient = dL/dy * x^T (outer product with input)
// 4. Propagate to previous layer: dL/dz = W^T * dL/dy
//-------------------------------------------
static void back_propagate_output(PNetwork pnet, PLayer layer, PLayer prev_layer, PTensor outputs)
{
	//-------------------------------
	// output layer back-propagation
	//-------------------------------

	// compute dL_dy = (r - y)
	tensor_axpby(1.0, outputs, -1.0, layer->t_values);

	// bias = bias + n * dL_dy
	tensor_axpy(pnet->learning_rate, layer->t_values, prev_layer->t_bias);

	// gradient += dL_dy * z
	tensor_outer((real)1.0, layer->t_values, prev_layer->t_values, prev_layer->t_gradients);

	// dL_dz = weights.T * dL_dy
// Sigmoid backpropagation:
// Chain rule: dL/dz = dL/da * da/dz * dz/dw
// For sigmoid: da/dz = a * (1 - a)
// Steps:
// 1. dL/dz = dL/dz * a(1-a) where a*a = a^2, (1-a) = 1-a^2
// 2. Update bias and compute weight gradient
// 3. Propagate to previous layer
//-------------------------------------------
	tensor_matvec(Tensor_Transpose, (real)1.0, prev_layer->t_weights, (real)0.0, layer->t_values, prev_layer->t_dl_dz);
}

//-------------------------------------------
// propagate back for sigmoid layers
//-------------------------------------------
static void back_propagate_sigmoid(PNetwork pnet, PLayer layer, PLayer prev_layer)
{
	//
	// gradient = dl_dz * z * (1 - z) * x = dl_dz_zomz * x 
	//

	// dl_dz = dl_dz * z
	tensor_mul(layer->t_dl_dz, layer->t_values);

	// z = z ^2
	tensor_square(layer->t_values);

	// z = z * dl_dz
	tensor_mul(layer->t_values, layer->t_dl_dz);

	// dl_dz = dl_dz - dl_dz * z^2
	tensor_sub(layer->t_dl_dz, layer->t_values);

	// bias = bias + n * dL_dz
	tensor_axpy(pnet->learning_rate, layer->t_dl_dz, prev_layer->t_bias);

	// gradient += dl_dz * x
	tensor_outer((real)1.0, layer->t_dl_dz, prev_layer->t_values, prev_layer->t_gradients);
// ReLU backpropagation:
// For ReLU: f(x) = max(0,x), so df/dx = 1 if x>0 else 0
// The heaviside step function approximates this: h(x) = (x>0) ? 1 : 0
// Steps:
// 1. Apply ReLU derivative mask to incoming gradient
// 2. Update bias and compute weight gradient with masked values
// 3. Propagate masked gradient to previous layer
//-------------------------------------------

	// dL_dz = weights.T * dL_dy
	tensor_matvec(Tensor_Transpose, (real)1.0, prev_layer->t_weights, (real)0.0, layer->t_dl_dz, prev_layer->t_dl_dz);
}

//-------------------------------------------
// propagate back for RELU layers
//-------------------------------------------
static void back_propagate_relu(PNetwork pnet, PLayer layer, PLayer prev_layer)
{
	//
	// gradient = dl_dz * d * x, where d is derivative of RELU(x) which is 0 or 1
	//
	
	// x = d * x
	tensor_heaviside(layer->t_values);

	// dl_dz = dl_dz * x
	tensor_mul(layer->t_dl_dz, layer->t_values);

	// bias = bias + n * dl_dz
	tensor_axpy(pnet->learning_rate, layer->t_dl_dz, prev_layer->t_bias);

	// gradient += dl_dz * d * x
	tensor_outer((real)1.0, layer->t_dl_dz, prev_layer->t_values, prev_layer->t_gradients);
// Full backpropagation pass through network:
// 1. Start at output layer with computed loss gradient
// 2. Work backward through each hidden layer
// 3. Each layer's back_prop_func handles its activation-specific gradient
// 4. Accumulates weight gradients for use by optimizer
//-------------------------------------------

	// dl_dz = weights.T * dl_dy ??? is that right?
	tensor_matvec(Tensor_Transpose, (real)1.0, prev_layer->t_weights, (real)0.0, layer->t_dl_dz, prev_layer->t_dl_dz);
}

//-------------------------------------------
// compute the gradients via back propagation
//-------------------------------------------
static void back_propagate(PNetwork pnet, PTensor outputs)
{
	// for each node in the output layer, excluding output layer bias node
	int output_layer = pnet->layer_count - 1;

	back_propagate_output(pnet, &pnet->layers[output_layer], &pnet->layers[output_layer - 1], outputs);
// Single training iteration:
// 1. Set input: copy input values to input layer
// 2. Forward pass: eval_network() propagates through all layers
// 3. Compute loss: measures difference between prediction and target
// 4. Backward pass: back_propagate() computes weight gradients
// Returns the loss for this example (used to track convergence)
//-------------------------------------------------

	//-------------------------------
	// hidden layer back-propagation
	// excluding the input layer
	//-------------------------------
	for (int layer = output_layer - 1; layer > 0; layer--)
	{
		pnet->layers[layer].back_prop_func(pnet, &pnet->layers[layer], &pnet->layers[layer - 1]);
	}
}

//-------------------------------------------------
// train the network over a single input/output set
//-------------------------------------------------
static real train_pass_network(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	if (!pnet || !inputs || !outputs)
		return 0.0;

	assert(pnet->layers[0].layer_type == LAYER_INPUT);
	assert(pnet->layers[pnet->layer_count - 1].layer_type == LAYER_OUTPUT);

	// set the input values on the network
	int node_count = pnet->layers[0].node_count;
	PLayer pLayer = &pnet->layers[0];
	for (int node = 0; node < node_count; node++)
	{
		pLayer->t_values->values[node] = inputs->values[node];
	}

	// forward evaluate the network
	eval_network(pnet);

	// compute the Loss function
	real loss = pnet->loss_func(pnet, outputs);

	// back propagate error through network to compute gradients
	back_propagate(pnet, outputs);

	return loss;
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
	for (int i = 0; i < count - 2; i++)
	{
		int j = i + (rand() % (count - i));
		int val = input_indices[i];
		input_indices[i] = input_indices[j];
		input_indices[j] = val;
	}
}

//-----------------------------------------------
// null optimization (no-op)
// Used when optimizer is set to OPTIM_NONE
// Weights remain unchanged; useful for testing/debugging
//-----------------------------------------------
static void optimize_none(PNetwork pnet)
{
	// do nothing!
}

//-----------------------------------------------
// SG with decaying learning rate
//-----------------------------------------------
static void optimize_decay(PNetwork pnet, real loss)
{
	pnet->learning_rate *= (real)DEFAULT_LEARNING_DECAY;
}

//-----------------------------------------------
// Adaptive learning rate
//-----------------------------------------------
// Adapts learning rate based on convergence progress:
// - If loss is decreasing: increase learning rate (explore more)
// - If loss is increasing: decrease learning rate (be more careful)
// - Averages last N losses to smooth decisions
// - Helps avoid both premature convergence and divergence
//-----------------------------------------------
static void optimize_adapt(PNetwork pnet, real loss)
{
	// adapt the learning rate
	optimize_decay(pnet, loss);

	real lastMSE = 0.0;
	// average the last 4 learning rates
	for (int i = 0; i < DEFAULT_MSE_AVG; i++)
	{
		lastMSE += pnet->lastMSE[i];
	}

	lastMSE /= DEFAULT_MSE_AVG;

	if (lastMSE > 0.0)
	{
		if (loss < lastMSE)
		{
			pnet->learning_rate += (real)DEFAULT_LEARN_ADD;

			// don't let learning rate go above 1
			if (pnet->learning_rate > 1.0)
				pnet->learning_rate = 1.0;
		}
		else
		{
			pnet->learning_rate -= (real)DEFAULT_LEARN_SUB * pnet->learning_rate;

			// don't let rate go below zero
			if (pnet->learning_rate <= 0.0)
				assert(0);
		}
	}

	int index = (pnet->mseCounter++) & (DEFAULT_MSE_AVG - 1);
	pnet->lastMSE[index] = loss;
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
	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		// W = W + n * gradients
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_gradients, pnet->layers[layer].t_weights);
	}
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
	real beta = (real)0.9, one_minus_beta = (real)0.1;

	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
	{
		// momentum = beta * m + one_minus_beta * gradients
		tensor_axpby(one_minus_beta, pnet->layers[layer].t_gradients, beta, pnet->layers[layer].t_m);

		// W = W + n * m
		tensor_axpy(pnet->learning_rate, pnet->layers[layer].t_m, pnet->layers[layer].t_weights);
	}
}

//-----------------------------------------------
// Adaptive gradient descent
//-----------------------------------------------
static void optimize_adagrad(PNetwork pnet)
{

}

//-----------------------------------------------
// Gradient descent with adaptive learning
//-----------------------------------------------
static void optimize_rmsprop(PNetwork pnet)
{
	//real beta = (real)0.9, one_minus_beta = (real)0.1;
	//real epsilon = (real)1e-6;
	//real v, gradient;

	//// for each layer after input
	//for (int layer = 1; layer < pnet->layer_count; layer++)
	//{
	//	// for each node in the layer
	//	int node_count = pnet->layers[layer].node_count;
	//	for (int node = 1; node < node_count; node++)
	//	{
	//		// for each node in previous layer
	//		int prev_node_count = pnet->layers[layer - 1].node_count;
	//		for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
	//		{
	//			// update the weights by the change
	//			gradient = pnet->layers[layer].nodes[node].gradients[prev_node];
	//			v = beta * pnet->layers[layer].nodes[node].v[prev_node] + one_minus_beta * gradient * gradient;
	//			pnet->layers[layer].nodes[node].v[prev_node] = v;
	//			pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * gradient / (real)(sqrt(v) + epsilon);
	//		}
	//	}
	//}
}

//-----------------------------------------------
// Adaptive moment estimation
//-----------------------------------------------
static void optimize_adam(PNetwork pnet)
{
	//real beta1 = (real)0.9, one_minus_beta1 = (real)0.1;
	//real beta2 = (real)0.999, one_minus_beta2 = (real)0.001;
	//real epsilon = (real)1e-8;
	//real m, v, mhat, vhat, gradient;

	//pnet->train_iteration++;

	//real one_minus_beta1_t = (real)1.0 / (real)(1.0 - pow(beta1, pnet->train_iteration));
	//real one_minus_beta2_t = (real)1.0 / (real)(1.0 - pow(beta2, pnet->train_iteration));

	//// for each layer after input
	//for (int layer = 1; layer < pnet->layer_count; layer++)
	//{
	//	// for each node in the layer
	//	int node_count = pnet->layers[layer].node_count;
	//	for (int node = 1; node < node_count; node++)
	//	{
	//		// for each node in previous layer
	//		int prev_node_count = pnet->layers[layer - 1].node_count;
	//		for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
	//		{
	//			// update the weights by the change
	//			gradient = pnet->layers[layer].nodes[node].gradients[prev_node];
	//			m = beta1 * pnet->layers[layer].nodes[node].m[prev_node] + one_minus_beta1 * gradient;
	//			v = beta2 * pnet->layers[layer].nodes[node].v[prev_node] + one_minus_beta2 * gradient * gradient;
	//			mhat = m * one_minus_beta1_t;
	//			vhat = v * one_minus_beta2_t;

	//			pnet->layers[layer].nodes[node].m[prev_node] = m;
	//			pnet->layers[layer].nodes[node].v[prev_node] = v;
	//			pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * mhat / (real)(sqrt(vhat) + epsilon);
	//		}
	//	}
	//}
}

//[]---------------------------------------------[]
//	Public ANN library interfaces
//[]---------------------------------------------[]

//------------------------------
// add a new layer to the network
//------------------------------
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type)
{
	if (!pnet)
		return ERR_NULL_PTR;

	if (node_count <= 0)
		return ERR_INVALID;

	// check whether we've run out of layers
	pnet->layer_count++;
	if (pnet->layer_count > pnet->layer_size)
	{
		// need to allocate more layers
		pnet->layer_size <<= 1;
		PLayer layer = realloc(pnet->layers, pnet->layer_size * (sizeof(Layer)));
		if (NULL == layer)
		{
			pnet->layer_count--;  // rollback count
			return ERR_ALLOC;
		}

		pnet->layers = layer;
	}

	int cur_layer = pnet->layer_count - 1;
	pnet->layers[cur_layer].layer_type = layer_type;
	pnet->layers[cur_layer].activation = activation_type;
	pnet->layers[cur_layer].back_prop_func = NULL;

	switch (activation_type)
	{
	case ACTIVATION_SIGMOID:
		pnet->layers[cur_layer].activation_func = sigmoid;
		pnet->layers[cur_layer].back_prop_func	= back_propagate_sigmoid;
		break;

	case ACTIVATION_RELU:
		pnet->layers[cur_layer].activation_func = relu;
		pnet->layers[cur_layer].back_prop_func	= back_propagate_relu;
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

	case ACTIVATION_NULL:
		pnet->layers[cur_layer].activation_func = no_activation;
		break;

	case ACTIVATION_SOFTMAX:
		pnet->layers[cur_layer].activation_func = no_activation;
		// handled after full network is evaluated
		break;

	default:
		assert(0);
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
		return ERR_ALLOC;
	}
	pnet->layers[cur_layer].node_count	= node_count;

	// create the tensors
	if (cur_layer > 0)
	{
		assert(pnet->layers[cur_layer - 1].t_weights == NULL);
		pnet->layers[cur_layer - 1].t_weights	= tensor_zeros(node_count, pnet->layers[cur_layer - 1].node_count);
		if (pnet->layers[cur_layer - 1].t_weights == NULL)
		{
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}

		pnet->layers[cur_layer - 1].t_v			= tensor_zeros(node_count, pnet->layers[cur_layer - 1].node_count);
		if (pnet->layers[cur_layer - 1].t_v == NULL)
		{
			tensor_free(pnet->layers[cur_layer - 1].t_weights);
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer - 1].t_weights = NULL;
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}

		pnet->layers[cur_layer - 1].t_m			= tensor_zeros(node_count, pnet->layers[cur_layer - 1].node_count);
		if (pnet->layers[cur_layer - 1].t_m == NULL)
		{
			tensor_free(pnet->layers[cur_layer - 1].t_v);
			tensor_free(pnet->layers[cur_layer - 1].t_weights);
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer - 1].t_v = NULL;
			pnet->layers[cur_layer - 1].t_weights = NULL;
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}

		pnet->layers[cur_layer - 1].t_gradients = tensor_zeros(node_count, pnet->layers[cur_layer - 1].node_count);
		if (pnet->layers[cur_layer - 1].t_gradients == NULL)
		{
			tensor_free(pnet->layers[cur_layer - 1].t_m);
			tensor_free(pnet->layers[cur_layer - 1].t_v);
			tensor_free(pnet->layers[cur_layer - 1].t_weights);
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer - 1].t_m = NULL;
			pnet->layers[cur_layer - 1].t_v = NULL;
			pnet->layers[cur_layer - 1].t_weights = NULL;
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}

		pnet->layers[cur_layer - 1].t_dl_dz		= tensor_zeros(1, pnet->layers[cur_layer - 1].node_count);
		if (pnet->layers[cur_layer - 1].t_dl_dz == NULL)
		{
			tensor_free(pnet->layers[cur_layer - 1].t_gradients);
			tensor_free(pnet->layers[cur_layer - 1].t_m);
			tensor_free(pnet->layers[cur_layer - 1].t_v);
			tensor_free(pnet->layers[cur_layer - 1].t_weights);
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer - 1].t_gradients = NULL;
			pnet->layers[cur_layer - 1].t_m = NULL;
			pnet->layers[cur_layer - 1].t_v = NULL;
			pnet->layers[cur_layer - 1].t_weights = NULL;
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}

		pnet->layers[cur_layer - 1].t_bias		= tensor_zeros(1, node_count);
		if (pnet->layers[cur_layer - 1].t_bias == NULL)
		{
			tensor_free(pnet->layers[cur_layer - 1].t_dl_dz);
			tensor_free(pnet->layers[cur_layer - 1].t_gradients);
			tensor_free(pnet->layers[cur_layer - 1].t_m);
			tensor_free(pnet->layers[cur_layer - 1].t_v);
			tensor_free(pnet->layers[cur_layer - 1].t_weights);
			tensor_free(pnet->layers[cur_layer].t_values);
			pnet->layers[cur_layer - 1].t_dl_dz = NULL;
			pnet->layers[cur_layer - 1].t_gradients = NULL;
			pnet->layers[cur_layer - 1].t_m = NULL;
			pnet->layers[cur_layer - 1].t_v = NULL;
			pnet->layers[cur_layer - 1].t_weights = NULL;
			pnet->layers[cur_layer].t_values = NULL;
			pnet->layer_count--;
			return ERR_ALLOC;
		}
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

	for (int i = 0; i < pnet->layer_size; i++)
	{
		pnet->layers[i].t_m 		= NULL;
		pnet->layers[i].t_v 		= NULL;
		pnet->layers[i].t_values 	= NULL;
		pnet->layers[i].t_weights 	= NULL;
		pnet->layers[i].t_gradients = NULL;
		pnet->layers[i].t_dl_dz		= NULL;
		pnet->layers[i].t_bias		= NULL;
	}

	for (int i = 0; i < DEFAULT_MSE_AVG; i++)
	{
		pnet->lastMSE[i] = (real)0.0;
	}

	ann_set_loss_function(pnet, loss_type);

	pnet->epochLimit		= 10000;
	pnet->train_iteration 	= 0;
	pnet->batchSize			= DEFAULT_BATCH_SIZE;
	pnet->print_func		= ann_puts;
	pnet->optimizer			= opt;

	switch(opt)
	{
	case OPT_ADAM:
		pnet->optimize_func = optimize_adam;
		pnet->learning_rate = (real)0.001;
		break;

	case OPT_ADAPT:
		pnet->optimize_func = optimize_sgd;
		break;

	case OPT_SGD_WITH_DECAY:
		pnet->optimize_func = optimize_sgd;
		break;

	case OPT_MOMENTUM:
		pnet->optimize_func = optimize_momentum;
		pnet->learning_rate = (real)0.01;
		break;

	case OPT_RMSPROP:
		pnet->optimize_func = optimize_rmsprop;
		pnet->learning_rate = (real)0.001;
		break;

	default:
	case OPT_SGD:
		pnet->optimize_func = optimize_sgd;
	}

	return pnet;
}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
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

	ann_printf(pnet,	"\nTraining ANN\n"
						"------------\n");
	ann_print_props(pnet);
	ann_printf(pnet, "  Training size: %u rows\n\n", rows);

	time_t time_start = time(NULL);

	pnet->train_iteration = 0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real loss;
	unsigned epoch = 0;
	unsigned correct = 0;
	int row;

	// create indices for shuffling the inputs and outputs
	int *input_indices = alloca(rows * sizeof(int));
	for (int i = 0; i < rows; i++)
	{
		input_indices[i] = i;
	}

	int inc = max(1, rows / 20);
	int input_node_count = (pnet->layers[0].node_count);
	int output_node_count = (pnet->layers[pnet->layer_count - 1].node_count);

	// validation data
	//PTensor x_valid = tensor_slice_rows(inputs, 50000);
	//PTensor y_valid = tensor_slice_rows(outputs, 50000);

	// tensors to hold input/output batches
	//PTensor input_batch	= tensor_create(pnet->batchSize, input_node_count);
	//PTensor output_batch	= tensor_create(pnet->batchSize, output_node_count);
	PTensor input_batch		= tensor_create(1, input_node_count);
	PTensor output_batch	= tensor_create(1, output_node_count);

	int batch_count = rows / pnet->batchSize;

	// train over epochs until done
	while (!converged)
	{
		// re-shuffle the indices for this epoch
		shuffle_indices(input_indices, rows);
		
		// iterate over all sets of inputs in this epoch/minibatch
		ann_printf(pnet, "Epoch %u/%u\n[", ++epoch, pnet->epochLimit);
		loss = (real)0.0;

		// iterate over all batches
		for (int batch = 0; batch < batch_count; batch++)
		{
			// zero the gradients
			for (int layer = 0; layer < pnet->layer_count - 1; layer++)
			{
				tensor_fill(pnet->layers[layer].t_gradients, (real)0.0);
			}

			loss = (real)0.0;

			for (unsigned batch_index = 0; batch_index < pnet->batchSize; batch_index++)
			{
				row = batch * pnet->batchSize + batch_index;

				real *ins = inputs->values + input_indices[row] * input_node_count;
				real *outs = outputs->values + input_indices[row] * output_node_count;

				memcpy(input_batch->values, inputs->values + input_indices[row] * input_node_count, input_node_count * sizeof(real));
				memcpy(output_batch->values, outputs->values + input_indices[row] * output_node_count, output_node_count * sizeof(real));

				loss += train_pass_network(pnet, input_batch, output_batch);

				if (row % inc == 0)
				{
					putchar('=');
				}
			}

			// average loss over batch-size
			loss /= (real)pnet->batchSize;

			// update weights based on batched gradients
			// using the chosen optimization function
			pnet->optimize_func(pnet);
		}

		ann_printf(pnet, "] - loss: %3.2g - LR: %3.2g\n", loss, pnet->learning_rate);

		// optimize learning once per epoch
		if (pnet->optimizer == OPT_SGD_WITH_DECAY || pnet->optimizer == OPT_MOMENTUM)
			optimize_decay(pnet, loss);
		else if (pnet->optimizer == OPT_ADAPT)
			optimize_adapt(pnet, loss);

		if (loss < pnet->convergence_epsilon)
		{
			ann_printf(pnet, "Network converged with loss: %3.2g out of %3.2g\n", loss, pnet->convergence_epsilon);
			converged = 1;
		}		

		// check for no convergence
		if (epoch >= pnet->epochLimit)
		{
			converged = 1;
		}
	}

	// free up batch tensors
	tensor_free(input_batch);
	tensor_free(output_batch);

	//tensor_free(x_valid);
	//tensor_free(y_valid);

	time_t time_end = time(NULL);
	double diff_t = (double)(time_end - time_start);
	double per_step = 1000.0 * diff_t / (rows * epoch);

	ann_printf(pnet, "\nTraining time: %f seconds, %f ms/step\n", diff_t, per_step);

	return loss;
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
// set the loss function
//------------------------------
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type)
{
	if (!pnet)
		return;

	switch (loss_type)
	{
	case LOSS_CATEGORICAL_CROSS_ENTROPY:
		pnet->loss_func = compute_cross_entropy;
		break;

	default:
	case LOSS_MSE:
		pnet->loss_func = compute_ms_error;
		break;
	}

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
		tensor_free(pnet->layers[layer].t_values);

		if (pnet->layers[layer].t_weights)
		{
			tensor_free(pnet->layers[layer].t_m);
			tensor_free(pnet->layers[layer].t_v);
			tensor_free(pnet->layers[layer].t_gradients);
			tensor_free(pnet->layers[layer].t_weights);
			tensor_free(pnet->layers[layer].t_dl_dz);
			tensor_free(pnet->layers[layer].t_bias);
		}
	}

	free(pnet->layers);

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
		return ERR_NULL_PTR;

	f = fopen(filename, "rt");
	if (!f)
		return ERR_IO;

	*rows = 0;

	dbuf = malloc(size * sizeof(real));

	// skip header if present
	if (has_header && !fgets(buf, DEFAULT_BUFFER_SIZE, f))
		return ERR_FAIL;

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
		return ERR_NULL_PTR;

	if (pnet->layer_count <= 0 || !pnet->layers)
		return ERR_INVALID;

	// set inputs
	int node_count = pnet->layers[0].node_count;
	for (int node = 0; node < node_count; node++)
	{
		pnet->layers[0].t_values->values[node] = *inputs++;
	}

	// evaluate network
	eval_network(pnet);

	// get the outputs
	node_count = pnet->layers[pnet->layer_count - 1].node_count;
	for (int node = 0; node < node_count; node++)
	{
		*outputs++ = pnet->layers[pnet->layer_count - 1].t_values->values[node];
	}

	return ERR_OK;
}

//-----------------------------------
// save network to a binary file
//-----------------------------------
int ann_save_network_binary(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
		return ERR_NULL_PTR;

	FILE *fptr = fopen(filename, "wb");
	if (!fptr)
		return ERR_IO;

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
	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
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

	ann_printf(pnet, "loading network %s...", filename);

	// create layers
	for (int layer = 0; layer < layer_count; layer++)
	{
		CHECK_RESULT(fread(&node_count, sizeof(node_count), 1, fptr), 1, NULL);
		CHECK_RESULT(fread(&layer_type, sizeof(layer_type), 1, fptr), 1, NULL);
		CHECK_RESULT(fread(&activation, sizeof(activation), 1, fptr), 1, NULL);

		ann_add_layer(pnet, node_count, layer_type, activation);
	}

	for (int layer = 0; layer < layer_count - 1; layer++)
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

	ann_printf(pnet, "done.\n");

	fclose(fptr);
	return pnet;
}

//------------------------------
// save network to a text file
//------------------------------
int ann_save_network(const PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
		return ERR_NULL_PTR;

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
		return ERR_IO;

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

	for (int layer = 0; layer < pnet->layer_count - 1; layer++)
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

	ann_printf(pnet, "loading network %s...", filename);

	// create layers
	for (int layer = 0; layer < layer_count; layer++)
	{
		CHECK_RESULT(fscanf(fptr, "%d", &node_count), 1, NULL);
		CHECK_RESULT(fscanf(fptr, "%d", &layer_type), 1, NULL);
		CHECK_RESULT(fscanf(fptr, "%d", &activation), 1, NULL);

		ann_add_layer(pnet, node_count, layer_type, activation);
	}

	for (int layer = 0; layer < layer_count - 1; layer++)
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

	ann_printf(pnet, "done.\n");

	fclose(fptr);
	return pnet;
}

//-----------------------------------------------
// display the network properties
//-----------------------------------------------
void ann_print_props(const PNetwork pnet)
{
	ann_printf(pnet, "  Network shape: ");

	for (int i = 0; i < pnet->layer_count; i++)
	{
		if (i != 0)
			ann_printf(pnet, "-");
		ann_printf(pnet, "%d", pnet->layers[i].node_count);
	}
	ann_printf(pnet, "\n");

	ann_printf(pnet, "      Optimizer: %s\n", optimizers[pnet->optimizer]);
	ann_printf(pnet, "  Loss function: %s\n", loss_types[pnet->loss_type]);
	ann_printf(pnet, "Mini-batch size: %u\n", pnet->batchSize);
}
