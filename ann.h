#ifndef __ANN_H
#define __ANN_H

#include "tensor.h"

//------------------------------
//
//------------------------------
#define E_FAIL -1
#define E_OK 0

//------------------------------
// Note: change to float if desired
//------------------------------
typedef FLOAT real;

//------------------------------
// Defines a node
//------------------------------
typedef struct
{
	real *weights;		// array of node weights
	real *dw;
	real value;			// node value
	real err;			// error term for this node
} Node, *PNode;

//------------------------------
// Layer types
//------------------------------
typedef enum { 
	LAYER_INPUT, 
	LAYER_HIDDEN, 
	LAYER_OUTPUT 
} Layer_type;

//------------------------------
// Activation types
//------------------------------
typedef enum { 
	ACTIVATION_NULL, 
	ACTIVATION_SIGMOID, 
	ACTIVATION_RELU,
	ACTIVATION_LEAKY_RELU,
	ACTIVATION_SOFTMAX 
} Activation_type;

//
//
//
typedef enum {
	LOSS_MSE,
	LOSS_CROSS_ENTROPY
} Loss_type;

//------------------------------
// Defines a layer in a network
//------------------------------
typedef struct
{
	int node_count;					// number of nodes in layer
	PNode nodes;					// array of nodes
	Layer_type layer_type;			// type of this layer
	Activation_type activation;		// type of activation, none, sigmoid, Relu
} Layer, *PLayer;

//------------------------------
// Defines a network
//------------------------------
typedef struct
{
	int layer_count;			// number of layers in network
	PLayer layers;				// array of layers
	real learning_rate;			// learning rate of network
	int size;					// number of layers allocated
	int weights_set;			// have the weights been initialized?
	real convergence_epsilon;	// threshold for convergence
	real lastMSE[4];
	unsigned mseCounter;
	int adaptiveLearning;
	unsigned epochLimit;
	Loss_type loss_type;
} Network, *PNetwork;

//------------------------------
// Configurable parameters
//------------------------------
#define DEFAULT_LAYERS			4
#define DEFAULT_CONVERGENCE 	0.01
#define DEFAULT_BUFFER_SIZE		1024
#define DEFAULT_LEARNING_RATE 	0.15	// pick a better default?
#define DEFAULT_LEARN_ADD		0.05
#define DEFAULT_LEARN_SUB		0.1

//------------------------------
//
//------------------------------
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork ann_make_network(void);
void ann_set_learning_rate(PNetwork pnet, real rate);
void ann_free_network(PNetwork pnet);
real ann_train_network(PNetwork pnet, real *inputs, size_t rows, size_t stride);
real ann_test_network(PNetwork pnet, real *inputs, real *outputs);
void ann_set_convergence(PNetwork pnet, real limit);
int ann_load_csv(const char *filename, real **data, size_t *rows, size_t *stride);

void print_network(PNetwork pnet);
void print_outputs(PNetwork pnet);
void softmax(PNetwork pnet);

#endif
