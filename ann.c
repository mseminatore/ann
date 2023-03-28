#include <stdio.h>
#include <stdlib.h>
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
	
	// get node count from previous layer
	if (pnet->layer_count > 1)
	{
		int node_weights = pnet->layers[cur_layer - 1].node_count;

		// allocate array of weights for every node in the current layer
		for (int i = 0; i < node_count; i++)
			pnet->layers[cur_layer].nodes[i].w = malloc(node_weights * sizeof(real));
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

	return pnet;
}

//------------------------------
// initialize the weights
//------------------------------
void init_weights(PNetwork pnet)
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
				pnet->layers[layer].nodes[node].w[weight] = (2.0 * (real)rand() / (real)RAND_MAX) - 1.0;
			}
		}
	}

	pnet->weights_set = 1;
}

//------------------------------
//
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
				sum += pnet->layers[layer - 1].nodes[prev_node].value * pnet->layers[layer].nodes[node].w[prev_node];
			}

			// update the nodes final value, using the correct activation function
			pnet->layers[layer].nodes[node].value = sigmoid(sum);
		}
	}

	return 0.0;
}

//------------------------------
// train the network
//------------------------------
real train_network(PNetwork pnet, real *inputs, int input_set_count, real *outputs)
{
	if (!pnet)
		return 0.0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	assert(pnet->layers[0].layer_type == LAYER_INPUT);

	// iterate over all sets of inputs
	for (int i = 0; i < input_set_count; i++)
	{
		// node 0 is a bias node in every layer
		pnet->layers[0].nodes[0].value = 1.0;

		// set the input values
		int node_count = pnet->layers[0].node_count;
		for (int node = 1; node < node_count; node++)
		{
			pnet->layers[0].nodes[node].value = *inputs++;
		}

		// evaluate the network
		eval_network(pnet);

		// back propagate

	}

	return 0.0;
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
				free(pnet->layers[layer].nodes[node].w);
		}

		free(pnet->layers[layer].nodes);
	}

	free(pnet->layers);

	// free network
	free(pnet);
}
