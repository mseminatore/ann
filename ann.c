#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
	return max(0.0, x);
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
		pnet->size <<= 1;
		pnet->layers = realloc(pnet->layers, pnet->size * (sizeof(Layer)));
		if (NULL == pnet->layers)
			return E_FAIL;
	}

	int cur_layer = pnet->layer_count - 1;
	pnet->layers[cur_layer].layer_type = layer_type;
	pnet->layers[cur_layer].activation = activation_type;

	// allocate the nodes
	PNode new_nodes = malloc(node_count * sizeof(Node));
	if (NULL == new_nodes)
		return E_FAIL;

	pnet->layers[cur_layer].nodes = new_nodes;
	pnet->layers[cur_layer].node_count = node_count;

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
	pnet->learning_rate = 0.1;	// pick a better default?

	return pnet;
}

//------------------------------
//
//------------------------------
real train_network(PNetwork pnet)
{
	return 0.0;
}

//------------------------------
//
//------------------------------
real test_network(PNetwork pnet)
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
		free(pnet->layers[layer].nodes);
		//for (int node = 0; node < pnet->layers[layer].node_count; node++)
		//{
		//}

	}

	free(pnet->layers);

	// free network
	free(pnet);
}
