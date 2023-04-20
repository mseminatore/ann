#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include "ann.h"

#if defined(_WIN32) && !defined(_WIN64)
	#define R_MIN -0.05
	#define R_MAX 0.05

#else
	#define R_MIN -1.0
	#define R_MAX 1.0
#endif

#ifndef max
#	define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

//-----------------------------------------------
//
//-----------------------------------------------
static void ann_puts(const char *s)
{
	fputs(s, stdout);
}

//-----------------------------------------------
//
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
// compute the sigmoid activation
//------------------------------
static real sigmoid(real x)
{
	return (real)(1.0 / (1.0 + exp(-x)));
}

//------------------------------
// compute the ReLU activation
//------------------------------
static real relu(real x)
{
	return (real)fmax(0.0, x);
}

//------------------------------
// compute the leaky ReLU activation
//------------------------------
static real leaky_relu(real x)
{
	return (real)fmax(0.01 * x, x);
}

//------------------------------
// compute the softmax
//------------------------------
void softmax(PNetwork pnet)
{
	real sum = 0.0;

	// find the sum of the output node values, excluding the bias noad
	int output_layer = pnet->layer_count - 1;
	for (int node = 1; node < pnet->layers[output_layer].node_count; node++)
	{
		sum += (real)exp(pnet->layers[output_layer].nodes[node].value);
	}

	for (int node = 1; node < pnet->layers[output_layer].node_count; node++)
	{
		pnet->layers[output_layer].nodes[node].value = (real)(exp(pnet->layers[output_layer].nodes[node].value) / sum);
		ann_printf(pnet, "%3.2g ", pnet->layers[output_layer].nodes[node].value);
	}
	puts("");
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
// initialize the weights
//------------------------------
static void init_weights(PNetwork pnet)
{
	if (!pnet || pnet->weights_set)
		return;

	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// input layers don't have weights
		if (pnet->layers[layer].layer_type == LAYER_INPUT)
			continue;

		int weight_count	= pnet->layers[layer - 1].node_count;
		int node_count		= pnet->layers[layer].node_count;

		for (int node = 0; node < node_count; node++)
		{
			for (int weight = 0; weight < weight_count; weight++)
			{
				// initialize weights to random values
				pnet->layers[layer].nodes[node].weights[weight] = get_rand((real)R_MIN, (real)R_MAX);
			}
		}
	}

	pnet->weights_set = 1;
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
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			ann_printf(pnet, "(%3.2g, ", pnet->layers[layer].nodes[node].value);

			if (layer > 0)
			{
				for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
				{
					ann_printf(pnet, "%3.2g, ", pnet->layers[layer].nodes[node].weights[prev_node]);
				}
			}
		}

		puts("]");
	}
}

//--------------------------------
//
//--------------------------------
void print_outputs(PNetwork pnet)
{
	if (!pnet)
		return;

	puts("");

	putchar('[');

	PLayer pLayer = &pnet->layers[0];
	for (int node = 1; node < pLayer->node_count; node++)
	{
		ann_printf(pnet, "%3.2g, ", pLayer->nodes[node].value);
	}

	puts("]");

	putchar('[');

	// print nodes in the output layer
	pLayer = &pnet->layers[pnet->layer_count - 1];
	for (int node = 1; node < pLayer->node_count; node++)
	{
		ann_printf(pnet, "%3.2g, ", pLayer->nodes[node].value);
	}

	puts("]");
}

//--------------------------------
// compute the mean squared error
//--------------------------------
static real compute_ms_error(PNetwork pnet, real *outputs)
{
	// get the output layer
	PLayer pLayer = &pnet->layers[pnet->layer_count - 1];

	assert(pLayer->layer_type == LAYER_OUTPUT);

	real mse = 0.0, diff;

	#pragma clang loop vectorize(enable)
	for (int i = 1; i < pLayer->node_count; i++)
	{
		diff = pLayer->nodes[i].value - outputs[i - 1];
		mse += diff * diff;
	}

	mse *= 0.5;

	return mse;
}

//------------------------------
// compute the cross entropy error
//------------------------------
static real compute_cross_entropy(PNetwork pnet, real *outputs)
{
	// get the output layer
	PLayer pLayer = &pnet->layers[pnet->layer_count - 1];

	assert(pLayer->layer_type == LAYER_OUTPUT);

	real xe = 0.0;

	#pragma clang loop vectorize(enable)
	for (int i = 1; i < pLayer->node_count; i++)
	{
		xe += (real)(outputs[i - 1] * log(pLayer->nodes[i].value));
	}

	return -xe;
}

#include <time.h>

// call this function to start a nanosecond-resolution timer
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

// call this function to end a timer, returning nanoseconds elapsed as a long
long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

//------------------------------
// forward evaluate the network
//------------------------------
static void eval_network(PNetwork pnet)
{
	if (!pnet)
		return;

// struct timespec vartime = timer_start();

	// loop over the non-input layers
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// loop over each node in the layer, skipping the bias node
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			real sum = 0.0;

			// loop over nodes in previous layer, including the bias node
			#pragma clang loop vectorize(enable)
			for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
			{
				// accumulate sum of prev nodes value times this nodes weight for that value
				sum += pnet->layers[layer - 1].nodes[prev_node].value * pnet->layers[layer].nodes[node].weights[prev_node];
			}

			// TODO - switch to function pointers for perf??
			// update the nodes final value, using the correct activation function
			switch (pnet->layers[layer].activation)
			{
			case ACTIVATION_SIGMOID:
				pnet->layers[layer].nodes[node].value = sigmoid(sum);
				break;

			case ACTIVATION_RELU:
				pnet->layers[layer].nodes[node].value = relu(sum);
				break;

			case ACTIVATION_LEAKY_RELU:
				pnet->layers[layer].nodes[node].value = leaky_relu(sum);
				break;

			case ACTIVATION_SOFTMAX:
				// handled after full network is evaluated
				break;

			default:
				assert(0);
				break;
			}
		}
	}

	// apply softmax on output if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax(pnet);

// long time_elapsed_nanos = timer_end(vartime);
// printf("Network eval, Time taken (nanoseconds): %ld\n", time_elapsed_nanos);

//while(1);
}

//------------------------------
// train the network over 
//------------------------------
static real train_pass_network(PNetwork pnet, real *inputs, real *outputs)
{
	if (!pnet || !inputs || !outputs)
		return 0.0;

	assert(pnet->layers[0].layer_type == LAYER_INPUT);
	assert(pnet->layers[pnet->layer_count - 1].layer_type == LAYER_OUTPUT);

	// set the input values on the network
	int node_count = pnet->layers[0].node_count;
	for (int node = 1; node < node_count; node++)
	{
		pnet->layers[0].nodes[node].value = *inputs++;
	}

	// forward evaluate the network
	eval_network(pnet);

	// compute the Mean Squared Error
	real err = pnet->error_func(pnet, outputs);

	//--------------------------------------------------------
	// back propagate and adjust weights
	//
	// compute the dw across the net, THEN update the weights
	//--------------------------------------------------------

	// for each node in the output layer, excluding output layer bias node
	real *expected_values	= outputs;
	int output_layer		= pnet->layer_count - 1;
	real x, z, result, y, err_term;
	real delta_w;

	for (int node = 1; node < pnet->layers[output_layer].node_count; node++)
	{
		delta_w = (real)0.0;

		// for each incoming input for this node, calculate the change in weight for that node
		for (int prev_node = 0; prev_node < pnet->layers[output_layer - 1].node_count; prev_node++)
		{
			z = pnet->layers[output_layer - 1].nodes[prev_node].value;
			result = *expected_values;
			y = pnet->layers[output_layer].nodes[node].value;

			err_term = (result - y);
			delta_w = pnet->learning_rate * err_term * z;

			pnet->layers[output_layer].nodes[node].dw[prev_node] = delta_w;
			pnet->layers[output_layer].nodes[node].err = err_term * pnet->layers[output_layer].nodes[node].weights[prev_node];
		}

		// get next expected output value
		expected_values++;
	}

	// process all hidden layers, excluding the input layer
	for (int layer = output_layer - 1; layer > 0; layer--)
	{
		// for each node of this layer
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			delta_w = (real)0.0;

			// for each incoming input to this node, calculate the weight change
			for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
			{
				real err_sum = (real)0.0;

				// for each following node
				for (int next_node = 1; next_node < pnet->layers[layer + 1].node_count; next_node++)
				{
					err_sum += pnet->layers[layer + 1].nodes[next_node].err;
				}

				x = pnet->layers[layer - 1].nodes[prev_node].value;
				z = pnet->layers[layer].nodes[node].value;

				delta_w = pnet->learning_rate * err_sum * z * ((real)1.0 - z) * x;

				pnet->layers[layer].nodes[node].dw[prev_node] = delta_w;

				// TODO - figure this out for multiple hidden layer case
//				pnet->layers[layer].nodes[node].err = new_err_term;
			}

		}
	}

	// update the weights based on calculated changes
	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			// for each node in previous layer
			#pragma clang loop vectorize(enable)
			for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
			{
				// update the weights by the change
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->layers[layer].nodes[node].dw[prev_node];
			}
		}
	}

//	ann_printf(pnet, "Err: %5.2g\n", err);
	// print_network(pnet);

	return err;
}

//-----------------------------------------------
// shuffle the indices
// https://en.wikipedia.org/wiki/Fisher–Yates_shuffle
//-----------------------------------------------
static void shuffle_indices(size_t *input_indices, size_t count)
{
	// Knuth's algorithm to shuffle an array a of n elements (indices 0..n-1):
	for (size_t i = 0; i < count - 2; i++)
	{
		size_t j = i + (rand() % (count - i));
		size_t val = input_indices[i];
		input_indices[i] = input_indices[j];
		input_indices[j] = val;
	}
}

//[]---------------------------------------------[]
// Public ANN library interfaces
//[]---------------------------------------------[]

//------------------------------
// add a new layer to the network
//------------------------------
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type)
{
	if (!pnet)
		return E_FAIL;

	// check whether we've run out of layers
	pnet->layer_count++;
	if (pnet->layer_count > pnet->size)
	{
		// need to allocate more layers
		pnet->size <<= 1;
		pnet->layers = realloc(pnet->layers, pnet->size * (sizeof(Layer)));
		if (NULL == pnet->layers)
			return E_FAIL;
	}

	int cur_layer = pnet->layer_count - 1;
	pnet->layers[cur_layer].layer_type = layer_type;
	pnet->layers[cur_layer].activation = activation_type;

	// allocate the nodes
	
	// add extra node for bias node
	node_count++;

	PNode new_nodes = malloc(node_count * sizeof(Node));
	if (NULL == new_nodes)
		return E_FAIL;

	pnet->layers[cur_layer].nodes		= new_nodes;
	pnet->layers[cur_layer].node_count	= node_count;
	
	// init the nodes
	pnet->layers[cur_layer].nodes[0].value = 1.0;
	for (int i = 1; i < node_count; i++)
	{
		pnet->layers[cur_layer].nodes[i].value = 0.0;
	}

	// get node count from previous layer
	if (pnet->layer_count > 1)
	{
		int node_weights = pnet->layers[cur_layer - 1].node_count;

		// allocate array of weights for every node in the current layer
		for (int i = 0; i < node_count; i++)
		{
			pnet->layers[cur_layer].nodes[i].weights = malloc(node_weights * sizeof(real));
			pnet->layers[cur_layer].nodes[i].dw = malloc(node_weights * sizeof(real));
		}
	}

	return E_OK;
}

//-----------------------------------------------
// simple static learning rate
//-----------------------------------------------
static void ann_lr_none(PNetwork pnet, real loss)
{
	// do nothing!!
}

//-----------------------------------------------
// simple decaying learning rate
//-----------------------------------------------
static void ann_lr_decay(PNetwork pnet, real loss)
{
	pnet->learning_rate *= DEFAULT_LEARNING_DECAY;
}

//-----------------------------------------------
// adaptive learning rate
//-----------------------------------------------
static void ann_lr_adapt(PNetwork pnet, real loss)
{
	// adapt the learning rate
	ann_lr_decay(pnet, loss);

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

//-----------------------------------------------
//
//-----------------------------------------------
static void ann_lr_momentum(PNetwork pnet, real loss)
{

}

//-----------------------------------------------
//
//-----------------------------------------------
static void ann_lr_adam(PNetwork pnet, real loss)
{

}

//------------------------------
// make a new network
//------------------------------
PNetwork ann_make_network(Optimizer_type opt)
{
	PNetwork pnet = malloc(sizeof(Network));
	if (NULL == pnet)
		return NULL;

	// set default values
	pnet->size			= DEFAULT_LAYERS;
	pnet->layers		= malloc(pnet->size * (sizeof(Layer)));
	pnet->layer_count	= 0;
	pnet->learning_rate = (real)DEFAULT_LEARNING_RATE;
	pnet->weights_set	= 0;
	pnet->convergence_epsilon = (real)DEFAULT_CONVERGENCE;
	pnet->mseCounter	= 0;

	for (int i = 0; i < DEFAULT_MSE_AVG; i++)
	{
		pnet->lastMSE[i] = (real)0.0;
	}

	pnet->epochLimit	= 10000;
	pnet->loss_type		= LOSS_MSE;

	pnet->error_func	= compute_ms_error;
	pnet->print_func	= ann_puts;

	switch(opt)
	{
	case OPT_ADAM:
		pnet->opt_func = ann_lr_adam;
		break;

	case OPT_ADAPT:
		pnet->opt_func = ann_lr_adapt;
		break;

	case OPT_DECAY:
		pnet->opt_func = ann_lr_decay;
		break;

	case OPT_MOMENTUM:
		pnet->opt_func = ann_lr_momentum;
		break;

	default:
	case OPT_NONE:
		pnet->opt_func = ann_lr_none;
	}

	return pnet;
}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, size_t rows)
{
	if (!pnet)
		return 0.0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real loss = 0.0;
	unsigned epoch = 0;
	unsigned correct = 0;

	// shuffle the inputs and outputs
	size_t *input_indices = alloca(rows * sizeof(size_t));
	for (size_t i = 0; i < rows; i++)
	{
		input_indices[i] = i;
	}

	while (!converged)
	{
		shuffle_indices(input_indices, rows);
		
		// iterate over all sets of inputs in this epoch/minibatch
		ann_printf(pnet, "Epoch %u/%u\n[", ++epoch, pnet->epochLimit);
		
		size_t inc = max(1, rows / 20);
		
		size_t intput_node_count = (pnet->layers[0].node_count - 1);
		size_t output_node_count = (pnet->layers[pnet->layer_count - 1].node_count - 1);

		for (size_t i = 0; i < rows; i++)
		{
			real *ins = inputs->values + input_indices[i] * intput_node_count;
			real *outs = outputs->values + input_indices[i] * output_node_count;

			loss += train_pass_network(pnet, ins, outs);

			if (i % inc == 0)
				putchar('=');
		}

		loss /= (real)rows;
		if (loss < pnet->convergence_epsilon)
			converged = 1;

		ann_printf(pnet, "], loss=%3.2g, LR=%3.2g\n", loss, pnet->learning_rate);

		// optimize learning
		pnet->opt_func(pnet, loss);

		// check for no convergence
		if (epoch >= pnet->epochLimit)
		{
			// puts("Error: network not converged.\n");
			converged = 1;
		}
	}

	//print_network(pnet);

	return loss;
}

//------------------------------
// evaluate the accuracy 
//------------------------------
real ann_evaluate(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	size_t correct = 0;

	if (!pnet || !inputs || !outputs)
	{
		return -1.0;
	}

	int classes = outputs->cols;
	real *pred = alloca(classes * sizeof(real));
	int pred_class, act_class;

	for (size_t i = 0; i < inputs->rows; i++)
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
int ann_class_prediction(real *outputs, int classes)
{
	int class = -1;
	real prob = 0.0;

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
//
//------------------------------
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type)
{
	if (!pnet)
		return;

	switch (loss_type)
	{
	case LOSS_CROSS_ENTROPY:
		pnet->error_func = compute_cross_entropy;
		break;

	default:
	case LOSS_MSE:
		pnet->error_func = compute_ms_error;
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
		// free nodes
		for (int node = 0; node < pnet->layers[layer].node_count; node++)
		{
			if (pnet->layers[layer].layer_type != LAYER_INPUT)
			{
				free(pnet->layers[layer].nodes[node].weights);
				free(pnet->layers[layer].nodes[node].dw);
			}
		}

		free(pnet->layers[layer].nodes);
	}

	free(pnet->layers);

	// free network
	free(pnet);
}

//------------------------------
// load data from a csv file
//------------------------------
int ann_load_csv(const char *filename, int has_header, real **data, size_t *rows, size_t *stride)
{
	FILE *f;
	char *s, buf[DEFAULT_BUFFER_SIZE];
	size_t size = DEFAULT_BUFFER_SIZE;
	real *dbuf;
	size_t lastStride = 0;
	uint32_t lineno = 0;
	size_t count = 0;

	f = fopen(filename, "rt");
	if (!f)
		return E_FAIL;

	*rows = 0;

	dbuf = malloc(size * sizeof(real));

	// skip header if present
	if (has_header)
		fgets(buf, DEFAULT_BUFFER_SIZE, f);

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

				dbuf = realloc(dbuf, size * sizeof(real));
				
				// check for OOM
				if (!dbuf)
				{
					free(dbuf);
					return E_FAIL;
				}
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
			return E_FAIL;
		}

		(*rows)++;

	}

	*data = dbuf;

	fclose(f);
	return E_OK;
}

//-----------------------------------
// predict an outcome from trained nn
//-----------------------------------
int ann_predict(PNetwork pnet, real *inputs, real *outputs)
{
	if (!pnet || !inputs || !outputs)
		return E_FAIL;

	// set inputs
	int node_count = pnet->layers[0].node_count;
	for (int node = 1; node < node_count; node++)
	{
		pnet->layers[0].nodes[node].value = *inputs++;
	}

	// evaluate network
	eval_network(pnet);

	// get the outputs
	node_count = pnet->layers[pnet->layer_count - 1].node_count;
	for (int node = 1; node < node_count; node++)
	{
		*outputs++ = pnet->layers[pnet->layer_count - 1].nodes[node].value;
	}

	return E_OK;
}

//------------------------------
// save out a network
//------------------------------
int ann_save_network(PNetwork pnet, const char *filename)
{
	if (!pnet || !filename)
		return E_FAIL;

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
		return E_FAIL;

	// TODO - save out network
	// TODO - save network props
	// TODO - save layer details
	// TODO - save node weights

	fclose(fptr);
	return E_OK;
}

//------------------------------
// load a previously saved network
//------------------------------
PNetwork ann_load_network(const char *filename)
{
	PNetwork pnet = ann_make_network(OPT_NONE);

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
		return NULL;

	// TODO - load network props
	// TODO - create layers
	// TODO - set node weights

	fclose(fptr);
	return pnet;
}
