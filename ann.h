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

//------------------------------
// Loss function types
//------------------------------
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

	// migration to tensor code path
	PTensor values;
	PTensor weights;
} Layer, *PLayer;

// defines an error function
typedef struct Network Network;
typedef struct Network *PNetwork;

typedef real (*err_func) (PNetwork pnet, real *outputs);
typedef void (*output_func) (const char *);
typedef void (*optimize_func) (PNetwork pnet);

//------------------------------
// Defines a network
//------------------------------
struct Network
{
	int layer_count;			// number of layers in network
	PLayer layers;				// array of layers
	real learning_rate;			// learning rate of network
	int size;					// number of layers allocated
	int weights_set;			// have the weights been initialized?
	real convergence_epsilon;	// threshold for convergence
	real lastMSE[DEFAULT_MSE_AVG];			// we average the last 4 MSE values
	unsigned mseCounter;
	int adaptiveLearning;		// is adaptive learning enabled?
	unsigned epochLimit;		// convergence epoch limit
	Loss_type loss_type;		// type of loss function used
	err_func error_func;		// the error function
	output_func print_func;
	optimize_func opt_func;
};

//------------------------------
// ANN function decls
//------------------------------

// building/freeing network model
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork ann_make_network(void);
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
void print_network(PNetwork pnet);
void print_outputs(PNetwork pnet);
void softmax(PNetwork pnet);

#endif
