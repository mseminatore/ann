#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

#include <math.h>
#include <assert.h>
#include "ann.h"

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
static void softmax(PNetwork pnet)
{
	real sum = 0.0;

	// find the sum of the output node values, excluding the bias noad
	int output_layer = pnet->layer_count - 1;
	for (int node = 1; node < pnet->layers[output_layer].node_count - 1; node++)
	{
		sum += (real)exp(pnet->layers[output_layer].nodes[node].value);
	}

	for (int node = 1; node < pnet->layers[output_layer].node_count - 1; node++)
	{
		pnet->layers[output_layer].nodes[node].value = (real)(exp(pnet->layers[output_layer].nodes[node].value) / sum);
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
				pnet->layers[layer].nodes[node].weights[weight] = get_rand((real)-0.01, (real)0.01);	// (2.0 * (real)rand() / (real)RAND_MAX) - 1.0;
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
		// printf("\nLayer %d\n"
		// 	"--------\n", layer);
		putchar('(');

		// print nodes in the layer, skipping bias nodes
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			printf("%3.2g, ", pnet->layers[layer].nodes[node].value);
		}

		puts(")");
	}
}

//--------------------------------
// compute the mean squared error
//--------------------------------
static real compute_error(PNetwork pnet, real *outputs)
{
	// get the output layer
	PLayer pLayer = &pnet->layers[pnet->layer_count - 1];

	assert(pLayer->layer_type == LAYER_OUTPUT);

	real mse = 0.0, diff;

	for (int i = 1; i < pLayer->node_count; i++)
	{
		diff = pLayer->nodes[i].value - outputs[i - 1];
		mse += diff * diff;
	}

	mse *= 0.5;

	return mse;
}

//------------------------------
// forward evaluate the network
//------------------------------
static void eval_network(PNetwork pnet)
{
	if (!pnet)
		return;

	// loop over the non-input layers
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// loop over each node in the layer, skipping the bias node
		for (int node = 1; node < pnet->layers[layer].node_count; node++)
		{
			real sum = 0.0;

			// loop over nodes in previous layer, including the bias node
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

	// node 0 is a bias node in every layer
	// TODO - this not needed here?
	pnet->layers[0].nodes[0].value = 1.0;

	// set the input values on the network
	int node_count = pnet->layers[0].node_count;
	for (int node = 1; node < node_count; node++)
	{
		pnet->layers[0].nodes[node].value = *inputs++;
	}

	// forward evaluate the network
	eval_network(pnet);

	//
	// back propagate and adjust weights
	//

	//
	// compute the dw across the net, THEN update the weights
	//

	// for each node in the output layer, excluding output layer bias node
	real *expected_values = outputs;
	int output_layer = pnet->layer_count - 1;
	for (int node = 1; node < pnet->layers[output_layer].node_count; node++)
	{
		real delta_w = 0.0;

		// for each incoming input for this node, calculate the change in weight for that node
		for (int prev_node = 0; prev_node < pnet->layers[output_layer - 1].node_count; prev_node++)
		{
			real z = pnet->layers[output_layer - 1].nodes[prev_node].value;
			real result = *expected_values;
			real y = pnet->layers[output_layer].nodes[node].value;

			real err_term = (result - y) * z;
			delta_w = pnet->learning_rate * err_term;
			pnet->layers[output_layer].nodes[node].dw[prev_node] = delta_w;
			pnet->layers[output_layer].nodes[node].err = err_term;
		}

		// get next expected output value
		expected_values++;
	}

	// process hidden layers, excluding input layer
	//for (int layer = output_layer - 1; layer > 0; layer--)
	//{
	//	// for each node of this layer
	//	for (int node = 0; node < pnet->layers[layer].node_count; node++)
	//	{
	//		real delta_w = 0.0;

	//		real err_term = pnet->layers[layer + 1].nodes[node].err;

	//		// for each incoming input to this node, calculate the weight change
	//		for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
	//		{
	//			// dw = n * (r - y) * z * v * (1-z) * x
	//			// dw = n * err_term * v * (1-z) * x
	//			real x = pnet->layers[layer - 1].nodes[prev_node].value;
	//			real v = pnet->layers[]
	//			real z = pnet->layers[layer].nodes[]

	//			delta_w = pnet->learning_rate * err_term * v * (1.0 - z) * x;
	//			pnet->layers[layer].nodes[node].dw[prev_node] = delta_w;
	//			pnet->layers[layer].nodes[node].err = new_err_term;
	//		}

	//	}
	//}

	// update the weights based on calculated changes
	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		for (int node = 0; node < pnet->layers[layer].node_count; node++)
		{
			// for each node in previous layer
			for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
			{
				// update the weights by the change
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->layers[layer].nodes[node].dw[prev_node];
			}
		}
	}

	// compute the Mean Squared Error
	real err = compute_error(pnet, outputs);
//	printf("Err: %5.2g\n", err);
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
// Public interfaces
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

//------------------------------
// make a new network
//------------------------------
PNetwork ann_make_network(void)
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
	pnet->lastMSE[0]	= pnet->lastMSE[1] = pnet->lastMSE[2] = pnet->lastMSE[3] = 0.0;
	pnet->adaptiveLearning = 1;
	pnet->epochLimit	= 10000;

	return pnet;
}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
real ann_train_network(PNetwork pnet, real *inputs, size_t rows, size_t stride)
{
	if (!pnet)
		return 0.0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real mse = 0.0;
	unsigned epoch = 0;
	
	// shuffle the inputs and outputs
	size_t *input_indices = alloca(rows * sizeof(size_t));
	for (size_t i = 0; i < rows; i++)
	{
		input_indices[i] = i;
	}

	while (!converged)
	{
		shuffle_indices(input_indices, rows);
		
		// iterate over all sets of inputs in this epoch/batch
		for (size_t i = 0; i < rows; i++)
		{
			real *ins = inputs + input_indices[i] * stride;
			real *outs = ins + pnet->layers[0].node_count - 1;

			mse += train_pass_network(pnet, ins, outs);
		}

		mse /= (real)rows;
		if (mse < pnet->convergence_epsilon)
			converged = 1;

		printf("Epoch %u, MSE = %5.2g, LR = %5.2g\n", ++epoch, mse, pnet->learning_rate);

		// adapt the learning rate if enabled
		if (pnet->adaptiveLearning)
		{
			// average the last 4 learning rates
			real lastMSE = (real)0.25 * (pnet->lastMSE[0] + pnet->lastMSE[1] + pnet->lastMSE[2] + pnet->lastMSE[3]);
			if (lastMSE > 0.0)
			{
				if (mse < lastMSE)
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
						converged = 1;
				}
			}

			int index = (pnet->mseCounter++) & 3;
			pnet->lastMSE[index] = mse;
		}

		// check for no convergence
		if (epoch > pnet->epochLimit)
		{
			puts("Error: network not converged.\n");
			converged = 1;
		}
	}

	//print_network(pnet);

	return mse;
}

//------------------------------
//
//------------------------------
real ann_test_network(PNetwork pnet, real *inputs, real *outputs)
{
	return 0.0;
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
int ann_load_csv(const char *filename, real **data, size_t *rows, size_t *stride)
{
	FILE *f;
	char *s, buf[DEFAULT_BUFFER_SIZE];
	size_t size = 8;
	real *dbuf;
	size_t lastStride = 0;

	f = fopen(filename, "rt");
	if (!f)
		return E_FAIL;

	*rows = 0;

	dbuf = malloc(size * sizeof(real));

	// read a line
	while (fgets(buf, DEFAULT_BUFFER_SIZE - 1, f))
	{
		// tokenize the line
		*stride = 0;
		s = strtok(buf, ", \n");

		// parse the line
		while (s) {
			dbuf[*rows] = (real)atof(s);
			(*rows)++;
			(*stride)++;

			if (*rows >= size)
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
			puts("Error: malformed CSV file\n");
			free(dbuf);
			fclose(f);
			return E_FAIL;
		}
	}

	*data = dbuf;
	(*rows) /= *stride;
	fclose(f);
	return E_OK;
}
