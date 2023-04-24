#ifndef __ANN_H
#define __ANN_H

#include "tensor.h"

#ifdef _WIN32
#	include <malloc.h>
#	include <stdint.h>
#else
#	include <alloca.h>
#endif


//------------------------------
// Configurable parameters
//------------------------------
#define DEFAULT_LAYERS			4		// we pre-alloc this many layers
#define DEFAULT_CONVERGENCE 	0.01	// MSE <= 1% is default
#define DEFAULT_SMALL_BUF_SIZE	1024	// size use for small temp buffers
#define DEFAULT_BUFFER_SIZE		8192	// size used for large temp buffers
#define DEFAULT_LEARNING_RATE 	0.05	// base learning rate
#define DEFAULT_LEARNING_DECAY	0.95	// decay rate for learning rate
#define DEFAULT_LEARN_ADD		0.01	// adaptive learning rate factors
#define DEFAULT_LEARN_SUB		0.75
#define DEFAULT_MSE_AVG			4		// number of prior MSE's to average

//
#define CSV_HAS_HEADER 1
#define CSV_NO_HEADER 0

//------------------------------
// Error values
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
	real *dw;			// change in weights
	real *m;			// momentum
	real *v;			// velocity
	real value;			// node value
	real dl_dz;			// gradient term for this node
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
	ACTIVATION_TANH,
	ACTIVATION_SOFTSIGN,
	ACTIVATION_SOFTMAX 
} Activation_type;

//------------------------------
// Loss function types
//------------------------------
typedef enum {
	LOSS_MSE,
	LOSS_CROSS_ENTROPY
} Loss_type;

//------------------------------
// SGD optimization kernels
//------------------------------
typedef enum {
	OPT_SGD,
	OPT_SGD_WITH_DECAY,
	OPT_ADAPT,
	OPT_MOMENTUM,
	OPT_RMSPROP,
	OPT_ADAGRAD,
	OPT_ADAM,
	OPT_DEFAULT = OPT_SGD
} Optimizer_type;

//------------------------------
// Defines a layer in a network
//------------------------------
typedef struct
{
	int node_count;					// number of nodes in layer
	PNode nodes;					// array of nodes

	Layer_type layer_type;			// type of this layer
	Activation_type activation;		// type of activation, none, sigmoid, Relu

	// migration to tensor code path
	PTensor t_values;
	PTensor t_weights;
	PTensor t_v;
	PTensor t_m;
	PTensor t_dw;
} Layer, *PLayer;

// forward decls
typedef struct Network Network;
typedef struct Network *PNetwork;

// function pointers for Network
typedef real (*Err_func) (PNetwork pnet, real *outputs);
typedef void (*Output_func) (const char *);
typedef void (*Optimization_func) (PNetwork pnet, real *inputs, real *outputs);

//------------------------------
// Defines a network
//------------------------------
struct Network
{
	int layer_count;					// number of layers in network
	PLayer layers;						// array of layers

	real learning_rate;					// learning rate of network
	int layer_size;						// number of layers allocated
	int weights_set;					// have the weights been initialized?

	real convergence_epsilon;			// threshold for convergence

	real lastMSE[DEFAULT_MSE_AVG];		// for averaging the last X MSE values
	unsigned mseCounter;
	unsigned epochLimit;				// convergence epoch limit
	Loss_type loss_type;				// type of loss function used
	Optimizer_type optimizer;
	unsigned train_iteration;

	Err_func error_func;				// the error function
	Output_func print_func;				// print output function
	Optimization_func optimize_func;	// learning rate/weight optimizer
};

//------------------------------
// ANN function decls
//------------------------------

// building/freeing network model
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork ann_make_network(Optimizer_type opt);
void ann_free_network(PNetwork pnet);
int ann_load_csv(const char *filename, int has_header, real **data, size_t *rows, size_t *stride);
PNetwork ann_load_network(const char *filename);
int ann_save_network(PNetwork pnet, const char *filename);

// training/evaluating
real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, size_t rows);
void ann_set_convergence(PNetwork pnet, real limit);
int ann_predict(PNetwork pnet, real *inputs, real *outputs);
int ann_class_prediction(real *outputs, int classes);
real ann_evaluate(PNetwork pnet, PTensor inputs, PTensor outputs);

// get/set network properties
void ann_set_learning_rate(PNetwork pnet, real rate);
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type);

// debugging functions
// void print_network(PNetwork pnet);
void print_outputs(PNetwork pnet);
void softmax(PNetwork pnet);

#endif
