#ifndef __ANN_H
#define __ANN_H

//------------------------------
//
//------------------------------
#define E_FAIL -1
#define E_OK 0

//------------------------------
// Note: change to float if desired
//------------------------------
typedef double real;

//------------------------------
// Defines a node
//------------------------------
typedef struct
{
	real *weights;		// array of node weights
	real value;			// node value
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
} Network, *PNetwork;

//------------------------------
//
//------------------------------
#define DEFAULT_LAYERS	4
#define DEFAULT_CONVERGENCE 0.01

//------------------------------
//
//------------------------------
int add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork make_network(void);
void set_learning_rate(PNetwork pnet, real rate);
void free_network(PNetwork pnet);
real train_pass_network(PNetwork pnet, real *inputs, real *outputs);
real train_network(PNetwork pnet, real *inputs, int input_set_count, real *outputs);
real test_network(PNetwork pnet, real *inputs, real *outputs);
//void init_weights(PNetwork pnet);
void set_convergence(PNetwork pnet, real limit);

#endif
