#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include "ann.h"

#if defined(_WIN32) && !defined(_WIN64)
	//#define R_MIN -0.05
	//#define R_MAX 0.05
#define R_MIN -1
#define R_MAX 1

#else
	#define R_MIN -1.0
	#define R_MAX 1.0
#endif

#ifdef __clang__
#	define CLANG_VECTORIZE 
//#pragma clang loop vectorize(enable)
#else
#	define CLANG_VECTORIZE
#endif

#define TENSOR_PATH 0

//
static const char *optimizers[] = {
	"Stochastic Gradient Descent",
	"Stochastic Gradient Descent with decay",
	"Adaptive SGD",
	"SGD with momentum",
	"RMSPROP",
	"ADAGRAD",
	"Adam",
	"SGD"
};

//
static const char *loss_types[] = {
	"Mean squared error",
	"Categorical cross-entropy",
	"Mean squared error"
};

//-----------------------------------------------
//
//-----------------------------------------------
static void ann_puts(const char *s)
{
	fputs(s, stdout);
}

//-----------------------------------------------
// output
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

//
static real no_activation(real x) { return x; }

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
// compute the tanh activation
//------------------------------
static real ann_tanh(real x)
{
	return (real)tanh(x);
}


//------------------------------
// compute the tanh activation
//------------------------------
static real softsign(real x)
{
	return (real)(x / (1.0 + fabs(x)));
}

//------------------------------
// compute the softmax
//------------------------------
static void softmax(PNetwork pnet)
{
	real sum = 0.0;

	// find the sum of the output node values, excluding the bias node
	int output_layer = pnet->layer_count - 1;
	int node_count = pnet->layers[output_layer].node_count;
	PNode pNode = pnet->layers[output_layer].nodes;

	for (int node = 1; node < node_count; node++)
	{
		sum += (real)exp(pNode[node].value);
	}

	for (int node = 1; node < node_count; node++)
	{
		pNode[node].value = (real)(exp(pNode[node].value) / sum);
	}
}

//-----------------------------------------------
//
//-----------------------------------------------
static void print_props(PNetwork pnet)
{
	ann_printf(pnet,	"Training ANN\n"
						"------------\n");
	ann_printf(pnet, "Network shape: ");
	for (int i = 0; i < pnet->layer_count; i++)
	{
		if (i != 0)
			putchar('-');
		ann_printf(pnet, "%d", pnet->layers[i].node_count - 1);
	}
	puts("");

	ann_printf(pnet, "Optimizer: %s\n", optimizers[pnet->optimizer]);
	ann_printf(pnet, "Loss function: %s\n", loss_types[pnet->loss_type]);
	ann_printf(pnet, "Batch size: %u\n", pnet->batchSize);
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

//------------------------------
// initialize the weights
//------------------------------
static void init_weights(PNetwork pnet)
{
	if (!pnet)
		return;

	for (int layer = 0; layer < pnet->layer_count; layer++)
	{
		// input layers don't have weights
		if (pnet->layers[layer].layer_type == LAYER_INPUT)
			continue;

		int weight_count	= pnet->layers[layer - 1].node_count;
		int node_count		= pnet->layers[layer].node_count;

//		real limit = (real)sqrt(6.0 / (weight_count + node_count));
//		real limit = (real)sqrt(1.0 / (weight_count));

		for (int node = 0; node < node_count; node++)
		{
			for (int weight = 0; weight < weight_count; weight++)
			{
				// if (weight == 0)
				// 	pnet->layers[layer].nodes[node].weights[weight] = pnet->init_bias;
				// else
					pnet->layers[layer].nodes[node].weights[weight] = get_rand((real)-pnet->weight_limit , (real)pnet->weight_limit);
				 //pnet->layers[layer].nodes[node].weights[weight] = get_rand((real)-limit, (real)limit);
			}
		}
	}

	pnet->weights_set = 1;
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
	PLayer poutput_layer = &pnet->layers[pnet->layer_count - 1];

	assert(poutput_layer->layer_type == LAYER_OUTPUT);

	real mse = (real)0.0, diff;

	CLANG_VECTORIZE
	for (int i = 1; i < poutput_layer->node_count; i++)
	{
		diff = outputs[i - 1] - poutput_layer->nodes[i].value;
		mse += diff * diff;
	}

	return mse;
}

//---------------------------------------------
// compute the categorical cross entropy error
// TODO this is buggy
//---------------------------------
static real compute_cross_entropy(PNetwork pnet, real *outputs)
{
	// get the output layer
	PLayer poutput_layer = &pnet->layers[pnet->layer_count - 1];

	assert(poutput_layer->layer_type == LAYER_OUTPUT);

	real xe = 0.0;

	CLANG_VECTORIZE
	for (int i = 1; i < poutput_layer->node_count; i++)
	{
		xe += (real)(outputs[i - 1] * log(poutput_layer->nodes[i].value));
	}

	return -xe;
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
			CLANG_VECTORIZE
			for (int prev_node = 0; prev_node < pnet->layers[layer - 1].node_count; prev_node++)
			{
				// accumulate sum of prev nodes value times this nodes weight for that value
				sum += pnet->layers[layer - 1].nodes[prev_node].value * pnet->layers[layer].nodes[node].weights[prev_node];
			}

			// update the nodes final value, using the correct activation function
			pnet->layers[layer].nodes[node].value = pnet->layers[layer].activation_func(sum);
		}
	}

	// apply softmax on output, if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax(pnet);
}

//-------------------------------------------
// compute the gradients via back propagation
//-------------------------------------------
static void back_propagate(PNetwork pnet, real *outputs)
{
	// for each node in the output layer, excluding output layer bias node
	int output_layer = pnet->layer_count - 1;
	real x, z, r, y, dl_dy, dl_dz;
	real gradient;
	real dl_dz_zomz;
	PNode pnode;

	//-------------------------------
	// output layer back-propagation
	//-------------------------------
	int output_nodes = pnet->layers[output_layer].node_count;
	for (int node = 1; node < output_nodes; node++)
	{
		// for each incoming input for this node, calculate the change in weight for that node
		int node_count = pnet->layers[output_layer - 1].node_count;
		pnode = &pnet->layers[output_layer].nodes[node];
		for (int prev_node = 0; prev_node < node_count; prev_node++)
		{
			z = pnet->layers[output_layer - 1].nodes[prev_node].value;
			r = *outputs;
			y = pnode->value;

			dl_dy = (r - y);
			gradient = dl_dy * z;

			pnode->gradients[prev_node] += gradient;
			pnode->dl_dz = dl_dy * pnode->weights[prev_node];
		}

		// get next expected output value
		outputs++;
	}

	//-------------------------------
	// hidden layer back-propagation
	// excluding the input layer
	//-------------------------------
	for (int layer = output_layer - 1; layer > 0; layer--)
	{
		// for each node of this layer
		int node_count = pnet->layers[layer].node_count;
		for (int node = 1; node < node_count; node++)
		{
			// for each incoming input to this node, calculate the weight change
			int prev_node_count = pnet->layers[layer - 1].node_count;
			for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
			{
				dl_dz = (real)0.0;

				// for each following node
				int next_node_count = pnet->layers[layer + 1].node_count;
				pnode = &pnet->layers[layer].nodes[node];
				for (int next_node = 1; next_node < next_node_count; next_node++)
				{
					dl_dz += pnet->layers[layer + 1].nodes[next_node].dl_dz;
				}

				x = pnet->layers[layer - 1].nodes[prev_node].value;
				z = pnode->value;
				dl_dz_zomz = dl_dz * z * ((real)1.0 - z);

				gradient = dl_dz_zomz * x;

				pnode->gradients[prev_node] += gradient;
				pnode->dl_dz = dl_dz_zomz * pnode->weights[prev_node];
			}
		}
	}
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

	// set the input values on the network, skipping the bias node
	int node_count = pnet->layers[0].node_count;
	PLayer player = &pnet->layers[0];
	for (int node = 1; node < node_count; node++)
	{
		player->nodes[node].value = *inputs++;
	}

#if TENSOR_PATH
	tensor_set_from_array(player->t_values, 1, node_count, inputs);
#endif

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

//-----------------------------------------------
//
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
// adaptive learning rate
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
//--------------------------------------------------------
static void optimize_sgd(PNetwork pnet)
{
	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		int node_count = pnet->layers[layer].node_count;
		for (int node = 1; node < node_count; node++)
		{
			// for each node in previous layer
			int prev_node_count = pnet->layers[layer - 1].node_count;
			for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
			{
				// update the weights by the change
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * pnet->layers[layer].nodes[node].gradients[prev_node];
			}
		}
	}
}

//-----------------------------------------------
// Gradient descent with momentum
//-----------------------------------------------
static void optimize_momentum(PNetwork pnet)
{
	real beta = (real)0.9, one_minus_beta = (real)0.1;
	real m;

	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		int node_count = pnet->layers[layer].node_count;
		for (int node = 1; node < node_count; node++)
		{
			// for each node in previous layer
			int prev_node_count = pnet->layers[layer - 1].node_count;
			for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
			{
				// update the weights by the change
				m = beta * pnet->layers[layer].nodes[node].m[prev_node] + one_minus_beta * pnet->layers[layer].nodes[node].gradients[prev_node];
				pnet->layers[layer].nodes[node].m[prev_node] = m;
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * m;
			}
		}
	}
}

//-----------------------------------------------
// Adaptive gradient descent
//-----------------------------------------------
static void optimize_adagrad(PNetwork pnet)
{

}

//-----------------------------------------------
//
//-----------------------------------------------
static void optimize_rmsprop(PNetwork pnet)
{
	real beta = (real)0.9, one_minus_beta = (real)0.1;
	real epsilon = (real)1e-6;
	real v, gradient;

	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		int node_count = pnet->layers[layer].node_count;
		for (int node = 1; node < node_count; node++)
		{
			// for each node in previous layer
			int prev_node_count = pnet->layers[layer - 1].node_count;
			for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
			{
				// update the weights by the change
				gradient = pnet->layers[layer].nodes[node].gradients[prev_node];
				v = beta * pnet->layers[layer].nodes[node].v[prev_node] + one_minus_beta * gradient * gradient;
				pnet->layers[layer].nodes[node].v[prev_node] = v;
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * gradient / (real)(sqrt(v) + epsilon);
			}
		}
	}
}

//-----------------------------------------------
// Adaptive moment estimation
//-----------------------------------------------
static void optimize_adam(PNetwork pnet)
{
	real beta1 = (real)0.9, one_minus_beta1 = (real)0.1;
	real beta2 = (real)0.999, one_minus_beta2 = (real)0.001;
	real epsilon = (real)1e-8;
	real m, v, mhat, vhat, gradient;

	pnet->train_iteration++;

	real one_minus_beta1_t = (real)1.0 / (real)(1.0 - pow(beta1, pnet->train_iteration));
	real one_minus_beta2_t = (real)1.0 / (real)(1.0 - pow(beta2, pnet->train_iteration));

	// for each layer after input
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		// for each node in the layer
		int node_count = pnet->layers[layer].node_count;
		for (int node = 1; node < node_count; node++)
		{
			// for each node in previous layer
			int prev_node_count = pnet->layers[layer - 1].node_count;
			for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
			{
				// update the weights by the change
				gradient = pnet->layers[layer].nodes[node].gradients[prev_node];
				m = beta1 * pnet->layers[layer].nodes[node].m[prev_node] + one_minus_beta1 * gradient;
				v = beta2 * pnet->layers[layer].nodes[node].v[prev_node] + one_minus_beta2 * gradient * gradient;
				mhat = m * one_minus_beta1_t;
				vhat = v * one_minus_beta2_t;

				pnet->layers[layer].nodes[node].m[prev_node] = m;
				pnet->layers[layer].nodes[node].v[prev_node] = v;
				pnet->layers[layer].nodes[node].weights[prev_node] += pnet->learning_rate * mhat / (real)(sqrt(vhat) + epsilon);
			}
		}
	}
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
		return E_FAIL;

	// check whether we've run out of layers
	pnet->layer_count++;
	if (pnet->layer_count > pnet->layer_size)
	{
		// need to allocate more layers
		pnet->layer_size <<= 1;
		pnet->layers = realloc(pnet->layers, pnet->layer_size * (sizeof(Layer)));
		if (NULL == pnet->layers)
			return E_FAIL;
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

	case ACTIVATION_NULL:
	case ACTIVATION_SOFTMAX:
		pnet->layers[cur_layer].activation_func = no_activation;
		// handled after full network is evaluated
		break;

	default:
		assert(0);
		break;
	}

	//--------------------
	// allocate the nodes
	//--------------------

	// add an extra for bias node
	node_count++;

	// create the node values tensor
	PTensor t = tensor_zeros(1, node_count);

	// set the bias node value to 1
	tensor_set(t, 0, 0, 1.0);
	pnet->layers[cur_layer].t_values = t;

	// create the weights tensor
	if (cur_layer > 0)
	{
		assert(pnet->layers[cur_layer - 1].t_weights == NULL);
		pnet->layers[cur_layer - 1].t_weights	= tensor_zeros(node_count, pnet->layers[cur_layer - 1].node_count);
		pnet->layers[cur_layer - 1].t_v			= tensor_zeros(1, node_count);
		pnet->layers[cur_layer - 1].t_m			= tensor_zeros(1, node_count);
		pnet->layers[cur_layer - 1].t_gradients = tensor_zeros(1, node_count);
	}

	// create the nodes
	PNode new_nodes = malloc(node_count * sizeof(Node));
	if (NULL == new_nodes)
		return E_FAIL;

	pnet->layers[cur_layer].nodes		= new_nodes;
	pnet->layers[cur_layer].node_count	= node_count;
	
	// bias node values are always 1
	pnet->layers[cur_layer].nodes[0].value = 1.0;

	// init the rest of the nodes
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
			pnet->layers[cur_layer].nodes[i].weights	= malloc(node_weights * sizeof(real));
			pnet->layers[cur_layer].nodes[i].m			= malloc(node_weights * sizeof(real));
			pnet->layers[cur_layer].nodes[i].v			= malloc(node_weights * sizeof(real));
			pnet->layers[cur_layer].nodes[i].gradients	= malloc(node_weights * sizeof(real));

			for (int j = 0; j < node_weights; j++)
			{
				pnet->layers[cur_layer].nodes[i].m[j] = (real)0.0;
				pnet->layers[cur_layer].nodes[i].v[j] = (real)0.0;
				pnet->layers[cur_layer].nodes[i].gradients[j] = (real)0.0;
			}
		}
	}

	return E_OK;
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
		pnet->layers[i].nodes 		= NULL;
	}

	for (int i = 0; i < DEFAULT_MSE_AVG; i++)
	{
		pnet->lastMSE[i] = (real)0.0;
	}

	ann_set_loss_function(pnet, loss_type);

	pnet->epochLimit	= 10000;
	pnet->train_iteration = 0;
	pnet->batchSize		= DEFAULT_BATCH_SIZE;

	pnet->print_func	= ann_puts;
	pnet->optimizer		= opt;

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
//
//-----------------------------------------------
void batch_eval_network(PNetwork pnet)
{
	if (!pnet)
		return;

	// loop over the non-input layers
	for (int layer = 1; layer < pnet->layer_count; layer++)
	{
		tensor_dot(pnet->layers[layer - 1].t_weights, pnet->layers[layer - 1].t_values, pnet->layers[layer].t_values);

		// update the nodes final value, using the correct activation function
//		pnet->layers[layer].nodes[node].value = pnet->layers[layer].activation_func(sum);
	}

	// apply softmax on output if requested
	if (pnet->layers[pnet->layer_count - 1].activation == ACTIVATION_SOFTMAX)
		softmax(pnet);
}

//-----------------------------------------------
// Train the network over a mini-batch
//-----------------------------------------------
real train_batch(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	if (!pnet || !inputs || !outputs)
		return 0.0;

	assert(pnet->layers[0].layer_type == LAYER_INPUT);
	assert(pnet->layers[pnet->layer_count - 1].layer_type == LAYER_OUTPUT);

	// set the input values on the network
	int node_count = pnet->layers[0].node_count;
	PLayer player = &pnet->layers[0];

	real loss = (real)0.0;

	for (size_t i = 0; i < pnet->batchSize; i++)
	{
		tensor_set_from_array(player->t_values, 1, node_count, inputs->values + i * node_count);

		// forward evaluate the network
		batch_eval_network(pnet);

		// compute the Loss function
		loss += pnet->loss_func(pnet, outputs->values);
	}

	// back propagate error through network and update weights
	pnet->optimize_func(pnet);

	return loss;
}

//-----------------------------------------------
// Train the network for a set of inputs/outputs
//-----------------------------------------------
real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, size_t rows)
{
	if (!pnet)
		return 0.0;

	print_props(pnet);
	ann_printf(pnet, "Training size: %u rows\n\n", rows);

	clock_t time_start = clock();

	pnet->train_iteration = 0;

	// initialize weights to random values if not already initialized
	init_weights(pnet);

	int converged = 0;
	real loss;
	unsigned epoch = 0;
	unsigned correct = 0;
	size_t row;

	// create indices for shuffling the inputs and outputs
	size_t *input_indices = alloca(rows * sizeof(size_t));
	for (size_t i = 0; i < rows; i++)
	{
		input_indices[i] = i;
	}

	size_t inc = max(1, rows / 20);
	size_t input_node_count = (pnet->layers[0].node_count - 1);
	size_t output_node_count = (pnet->layers[pnet->layer_count - 1].node_count - 1);

	// tensors to hold input/output batches
	PTensor input_batch		= tensor_create(pnet->batchSize, input_node_count);
	PTensor output_batch	= tensor_create(pnet->batchSize, output_node_count);

	size_t batch_count = rows / pnet->batchSize;

	// train over epochs until done
	while (!converged)
	{
		// re-shuffle the indices for this epoch
		shuffle_indices(input_indices, rows);
		
		// iterate over all sets of inputs in this epoch/minibatch
		ann_printf(pnet, "Epoch %u/%u\n[", ++epoch, pnet->epochLimit);
		loss = (real)0.0;

		// iterate over all batches
		for (size_t batch = 0; batch < batch_count; batch++)
		{
			// zero the gradients and dLdZ
			for (int layer = 1; layer < pnet->layer_count; layer++)
			{
				int node_count = pnet->layers[layer].node_count;
				for (int node = 0; node < node_count; node++)
				{
					int prev_node_count = pnet->layers[layer - 1].node_count;
					for (int prev_node = 0; prev_node < prev_node_count; prev_node++)
					{
						pnet->layers[layer].nodes[node].gradients[prev_node] = (real)0.0;
					}
				}
			}

			loss = (real)0.0;

			for (size_t batch_index = 0; batch_index < pnet->batchSize; batch_index++)
			{
				row = batch * pnet->batchSize + batch_index;

				real *ins = inputs->values + input_indices[row] * input_node_count;
				real *outs = outputs->values + input_indices[row] * output_node_count;

				loss += train_pass_network(pnet, ins, outs);

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

	clock_t time_end = clock();
	double diff_t = (time_end - time_start) / CLOCKS_PER_SEC;
	double per_step = 1000.0 * diff_t / (rows * epoch);

	ann_printf(pnet, "\nTraining time: %f seconds, %f ms/step\n", diff_t, per_step);

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
//
//------------------------------
static void free_node(PNode pnode)
{
	free(pnode->weights);
	free(pnode->m);
	free(pnode->v);
	free(pnode->gradients);
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
		}

		// free nodes
		for (int node = 0; node < pnet->layers[layer].node_count; node++)
		{
			if (pnet->layers[layer].layer_type != LAYER_INPUT)
			{
				free_node(&pnet->layers[layer].nodes[node]);
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
	PNetwork pnet = ann_make_network(OPT_SGD, LOSS_DEFAULT);

	FILE *fptr = fopen(filename, "wt");
	if (!fptr)
		return NULL;

	// TODO - load network props
	// TODO - create layers
	// TODO - set node weights

	fclose(fptr);
	return pnet;
}
