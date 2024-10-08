/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#pragma once

#ifndef __ANN_H
#define __ANN_H

#include "tensor.h"

#ifdef _WIN32
#	include <malloc.h>
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
#define DEFAULT_LEARN_ADD		0.005	// adaptive learning rate factors
#define DEFAULT_LEARN_SUB		0.75
#define DEFAULT_MSE_AVG			4		// number of prior MSE's to average
#define DEFAULT_BATCH_SIZE		32

//-----------------------------------------------
// flags for CSV reader
//-----------------------------------------------
#define CSV_HAS_HEADER 	1
#define CSV_NO_HEADER 	0

//------------------------------
// Error values
//------------------------------
#define ERR_FAIL 	-1
#define ERR_OK 		0

// validation helper
#define CHECK_OK(s) if ((s) != ERR_OK) return ERR_FAIL
#define CHECK_RESULT(fn, result, retval) if ((result) != (fn)) return (retval)

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
	LOSS_CATEGORICAL_CROSS_ENTROPY,
	LOSS_DEFAULT = LOSS_MSE
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


//-----------------------------------------------
// forward decls
//-----------------------------------------------
typedef struct Network Network;
typedef struct Network *PNetwork;
typedef struct Layer Layer;
typedef struct Layer *PLayer;

//-----------------------------------------------
// function pointers for Network
//-----------------------------------------------
typedef real(*Loss_func) (PNetwork pnet, PTensor outputs);
typedef void(*Output_func) (const char *);
typedef void(*Optimization_func) (PNetwork pnet);
typedef real(*Activation_func) (real);
typedef void(*BackPropagate_func)(PNetwork pnet, PLayer layer, PLayer prev_layer);

//------------------------------
// Defines a layer in a network
//------------------------------
struct Layer
{
	int node_count;						// number of nodes in layer

	Layer_type layer_type;				// type of this layer
	Activation_type activation;			// type of activation, none, sigmoid, Relu
	Activation_func activation_func;	// node activation function
	BackPropagate_func back_prop_func;	// back propagation function for this layer

	PTensor t_values;					// tensor of node values for the layer
	PTensor t_weights;					// tensor of weights for the layer
	PTensor t_v;						// tensor of velocities for optimizer
	PTensor t_m;						// tensor of momentums for optimizer
	PTensor t_gradients;				// tensor of gradients for back propagation
	PTensor t_dl_dz;					// tensor of dL_dz
	PTensor t_bias;						// bias vector
};

//------------------------------
// Defines a network
//------------------------------
struct Network
{
	int layer_count;					// number of layers in network
	PLayer layers;						// array of layers

	FILE *dbg;
	real learning_rate;					// learning rate of network
	int layer_size;						// number of layers allocated
	int weights_set;					// have the weights been initialized?
	unsigned batchSize;					// size of mini-batches

	real convergence_epsilon;			// threshold for convergence
	real weight_limit;					// range limit for initial weights
	real init_bias;						// initial bias node values

	real lastMSE[DEFAULT_MSE_AVG];		// for averaging the last X MSE values
	unsigned mseCounter;
	unsigned epochLimit;				// convergence epoch limit
	Loss_type loss_type;				// type of loss function used
	Optimizer_type optimizer;
	unsigned train_iteration;

	Loss_func loss_func;				// the error function
	Output_func print_func;				// print output function
	Optimization_func optimize_func;	// learning rate/weight optimizer
};

//------------------------------
// ANN public function decls
//------------------------------

// building/freeing network model
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
PNetwork ann_make_network(Optimizer_type opt, Loss_type loss_type);
void ann_free_network(PNetwork pnet);
int ann_load_csv(const char *filename, int has_header, real **data, int *rows, int *stride);
PNetwork ann_load_network(const char *filename);
int ann_save_network(PNetwork pnet, const char *filename);
int ann_save_network_binary(PNetwork pnet, const char *filename);
PNetwork ann_load_network_binary(const char *filename);

// training/evaluating
real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, int rows);
void ann_set_convergence(PNetwork pnet, real limit);
int ann_predict(PNetwork pnet, real *inputs, real *outputs);
int ann_class_prediction(real *outputs, int classes);
real ann_evaluate_accuracy(PNetwork pnet, PTensor inputs, PTensor outputs);

// get/set/show network properties
void ann_set_learning_rate(PNetwork pnet, real rate);
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type);
void ann_print_props(PNetwork pnet);

// debugging functions
// void print_network(PNetwork pnet);
void print_outputs(PNetwork pnet);

#endif
