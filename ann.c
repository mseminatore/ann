#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>
#include "ann.h"

//------------------------------
// compute the sigmoid activation
//------------------------------
static real sigmoid(real x)
{
	return 1.0 / (1.0 + exp(-x));
}

//------------------------------
// compute the ReLU activation
//------------------------------
static real relu(real x)
{
	return fmax(0.0, x);
}


//------------------------------
// compute the leaky ReLU activation
//------------------------------
static real leaky_relu(real x)
{
	return fmax(0.01 * x, x);
}

//[]---------------------------------------------
// Public interfaces
//[]---------------------------------------------

//------------------------------
// add a new layer to the network
//------------------------------
int add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type)
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
			pnet->layers[cur_layer].nodes[i].weights = malloc(node_weights * sizeof(real));
	}

	return E_OK;
}

//------------------------------
// make a new network
//------------------------------
PNetwork make_network(void)
{
	PNetwork pnet = malloc(sizeof(Network));
	if (NULL == pnet)
		return NULL;

	pnet->size			= DEFAULT_LAYERS;
	pnet->layers		= malloc(pnet->size * (sizeof(Layer)));
	pnet->layer_count	= 0;
	pnet->learning_rate = 0.15;	// pick a better default?
	pnet->weights_set	= 0;
	pnet->convergence_epsilon = DEFAULT_CONVERGENCE;

	return pnet;
}

//------------------------------
//
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
				pnet->layers[layer].nodes[node].weights[weight] = get_rand(-0.01, 0.01);	// (2.0 * (real)rand() / (real)RAND_MAX) - 1.0;
			}
		}
	}

	pnet->weights_set = 1;
}

//------------------------------
// forward evaluate the network
//------------------------------
real eval_network(PNetwork pnet)
{
	if (!pnet)
		return 0.0;

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

			default:
			case ACTIVATION_SOFTMAX:
				assert(0);
				break;
			}
		}
	}

	return 0.0;
}

//------------------------------
// print the network
//------------------------------
static void print_network(PNetwork pnet)
{
	if (!pnet)
		return;

	// print each layer
	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		printf(	"\nLayer %d\n"
				"--------\n", layer);

		// print nodes in the layer
		for (int node = 0; node < pnet->layers[layer].node_count; node++)
		{
			printf("(%3.2g), ", pnet->layers[layer].nodes[node].value);
		}

		puts("");
	}
}

//--------------------------------
// compute the mean squared error
//--------------------------------
real compute_error(PNetwork pnet, real *outputs)
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
// train the network
//------------------------------
real train_pass_network(PNetwork pnet, real *inputs, real *outputs)
{
	if (!pnet)
		return 0.0;

//	print_network(pnet);

	assert(pnet->layers[0].layer_type == LAYER_INPUT);

	// node 0 is a bias node in every layer
	pnet->layers[0].nodes[0].value = 1.0;

	// set the input values on the network
	int node_count = pnet->layers[0].node_count;
	for (int node = 1; node < node_count; node++)
	{
		pnet->layers[0].nodes[node].value = *inputs++;
	}

	// forward evaluate the network
	eval_network(pnet);

//	print_network(pnet);

	//
	// back propagate and adjust weights
	//

	// for each node in the output layer, excluding bias nodes
	real *expected_values = outputs;
	for (int node = 1; node < pnet->layers[pnet->layer_count - 1].node_count; node++)
	{
		real delta_w = 0.0;

		// for each incoming input for this node, calculate the change in weight for that node
		for (int prev_node = 0; prev_node < pnet->layers[pnet->layer_count - 2].node_count; prev_node++)
		{
			real x = pnet->layers[pnet->layer_count - 2].nodes[prev_node].value;
			real result = *expected_values;
			real y = pnet->layers[pnet->layer_count - 1].nodes[node].value;

			delta_w = pnet->learning_rate * (result - y) * x;
			pnet->layers[pnet->layer_count - 1].nodes[node].weights[prev_node] += delta_w;
		}

		// get next expected output value
		expected_values++;
	}

	// hidden layers
	//for (int layer = pnet->layer_count - 1; layer > 0; layer--)
	//{

	//}

	// compute the Mean Squared Error
	real err = compute_error(pnet, outputs);
//	printf("Err: %5.2g\n", err);
//	print_network(pnet);

	return err;
}

//-----------------------------------------------
// shuffle the indices
//-----------------------------------------------
void shuffle_indices(int *input_indices, int count)
{

}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
real train_network(PNetwork pnet, real *inputs, int input_set_count, real *outputs)
{
	if (!pnet)
		return 0.0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real mse = 0.0;
	int epoch = 0;

	// shuffle the inputs and outputs
	int *input_indices = alloca(input_set_count * sizeof(int));
	for (int i = 0; i < input_set_count; i++)
	{
		input_indices[i] = i;
	}

	shuffle_indices(input_indices, input_set_count);

	while (!converged)
	{
		real *ins = inputs;
		real *outs = outputs;

		// iterate over all sets of inputs
		for (int i = 0; i < input_set_count; i++)
		{
			mse += train_pass_network(pnet, ins, outs);
			ins += (pnet->layers[0].node_count - 1);
			outs += (pnet->layers[pnet->layer_count - 1].node_count - 1);
		}

		mse /= (real)input_set_count;
		if (mse < pnet->convergence_epsilon)
			converged = 1;

		printf("Epoch %d, MSE = %5.2g\n", ++epoch, mse);
	}

	return mse;
}

//------------------------------
//
//------------------------------
real test_network(PNetwork pnet, real *inputs, real *outputs)
{
	return 0.0;
}

//------------------------------
// set the network learning rate
//------------------------------
void set_learning_rate(PNetwork pnet, real rate)
{
	if (!pnet)
		return;

	pnet->learning_rate = rate;
}

//------------------------------
//
//------------------------------
void set_convergence(PNetwork pnet, real limit)
{
	if (!pnet)
		return;

	pnet->convergence_epsilon = limit;
}
//------------------------------
// free a network
//------------------------------
void free_network(PNetwork pnet)
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
				free(pnet->layers[layer].nodes[node].weights);
		}

		free(pnet->layers[layer].nodes);
	}

	free(pnet->layers);

	// free network
	free(pnet);
}
