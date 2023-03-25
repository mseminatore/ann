#ifndef __ANN_H
#define __ANN_H

//------------------------------
//
//------------------------------
#define E_FAIL -1
#define E_OK 0

//------------------------------
// Note: change to double if needed
//------------------------------
typedef double real;

//------------------------------
// Defines a node
//------------------------------
typedef struct
{
	real w;
	real value;
} Node, *PNode;

//
typedef enum { LAYER_INPUT, LAYER_HIDDEN, LAYER_OUTPUT } Layer_type;
typedef enum { ACTIVATION_NULL, ACTIVATION_SIGMOID, ACTIVATION_RELU } Activation_type;

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
	int layer_count;		// number of layers in network
	PLayer layers;			// array of layers
	real learning_rate;		// learning rate of network
	int size;				// number of layers allocated
} Network, *PNetwork;

//------------------------------
//
//------------------------------
#define DEFAULT_LAYERS	4
#define DEFAULT_NODES	8

//------------------------------
//
//------------------------------
int add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork make_network(void);
void set_learning_rate(PNetwork pnet, real rate);
void free_network(PNetwork pnet);

#endif
