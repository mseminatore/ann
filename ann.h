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
#define ERR_FAIL 		-1
#define ERR_OK 			0
#define ERR_NULL_PTR	-2
#define ERR_ALLOC		-3
#define ERR_INVALID		-4
#define ERR_IO			-5

// validation helper
#define CHECK_OK(s) if ((s) != ERR_OK) return ERR_FAIL
#define CHECK_RESULT(fn, result, retval) if ((result) != (fn)) return (retval)
#define CHECK_NULL(ptr) if ((ptr) == NULL) return ERR_NULL_PTR

//------------------------------
// Error logging callback type
//------------------------------
/**
 * Function pointer type for error logging callbacks.
 * 
 * Called when library errors occur (if callback is set via ann_set_error_log_callback).
 * Allows integration with custom logging systems, monitoring, or alerting.
 * 
 * @param error_code Numeric error code (ERR_NULL_PTR, ERR_ALLOC, etc.)
 * @param error_message Human-readable error message from ann_strerror()
 * @param function_name Name of the function where error occurred
 * 
 * Example callback:
 *   void my_error_handler(int code, const char *msg, const char *func) {
 *       fprintf(stderr, "[%s] Error %d: %s\n", func, code, msg);
 *       log_to_monitoring_system(code, msg, func);
 *   }
 *   ann_set_error_log_callback(my_error_handler);
 */
typedef void (*ErrorLogCallback)(int error_code, const char *error_message, const char *function_name);

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
	OPT_DEFAULT = OPT_ADAM
} Optimizer_type;

//------------------------------
// Weight initialization types
//------------------------------
typedef enum {
	WEIGHT_INIT_UNIFORM,       // Uniform distribution [-limit, limit] (current default)
	WEIGHT_INIT_XAVIER,        // Xavier/Glorot: std = sqrt(2 / (fan_in + fan_out)) - good for sigmoid/tanh
	WEIGHT_INIT_HE,            // He: std = sqrt(2 / fan_in) - good for ReLU variants
	WEIGHT_INIT_AUTO,          // Automatically choose based on activation function
	WEIGHT_INIT_DEFAULT = WEIGHT_INIT_AUTO
} Weight_init_type;


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

/**
 * Learning rate scheduler callback.
 * Called at the start of each epoch to compute the learning rate.
 *
 * @param epoch Current epoch number (1-based)
 * @param base_lr Initial/base learning rate
 * @param user_data User-provided context (scheduler parameters)
 * @return New learning rate for this epoch
 */
typedef real(*LRSchedulerFunc)(unsigned epoch, real base_lr, void *user_data);

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

	real dropout_rate;					// dropout rate for this layer (0.0 = disabled)
	PTensor t_dropout_mask;				// dropout mask tensor (for batched training)

	PTensor t_values;					// tensor of node values for the layer
	PTensor t_weights;					// tensor of weights for the layer
	PTensor t_v;						// tensor of velocities for optimizer
	PTensor t_m;						// tensor of momentums for optimizer
	PTensor t_gradients;				// tensor of gradients for back propagation
	PTensor t_dl_dz;					// tensor of dL_dz
	PTensor t_bias;						// bias vector
	PTensor t_bias_grad;				// tensor of gradients for bias
	PTensor t_bias_m;					// tensor of momentums for bias (Adam)
	PTensor t_bias_v;					// tensor of velocities for bias (AdaGrad/RMSProp/Adam)

	// Batched training tensors (batch_size × node_count)
	PTensor t_batch_values;				// batch activations for forward pass
	PTensor t_batch_dl_dz;				// batch gradients for backward pass
	PTensor t_batch_z;					// batch pre-activation values (for activation derivatives)
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
	real base_learning_rate;			// initial LR (for schedulers)
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

	real max_gradient;					// gradient clipping threshold (0 = disabled)
	Weight_init_type weight_init;		// weight initialization strategy

	real default_dropout;				// default dropout rate for hidden layers (0 = disabled)
	int is_training;					// 1 = training mode (apply dropout), 0 = inference mode

	Loss_func loss_func;				// the error function
	Output_func print_func;				// print output function
	Optimization_func optimize_func;	// learning rate/weight optimizer

	LRSchedulerFunc lr_scheduler;		// learning rate scheduler callback
	void *lr_scheduler_data;			// user data for scheduler

	unsigned current_batch_size;		// current allocated batch size for batch tensors
};

//------------------------------
// ANN public function decls
//------------------------------

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

/**
 * Create a new neural network with specified optimizer and loss function.
 * 
 * The network starts with no layers. Use ann_add_layer() to build the network
 * topology. Weights are initialized automatically when training begins.
 * 
 * @param opt Optimizer algorithm (OPT_SGD, OPT_MOMENTUM, OPT_ADAM, etc.)
 * @param loss_type Loss function (LOSS_MSE or LOSS_CATEGORICAL_CROSS_ENTROPY)
 * @return Pointer to newly allocated network, or NULL on allocation failure
 * 
 * @see ann_add_layer() to add layers to the network
 * @see ann_free_network() to free network resources
 */
PNetwork ann_make_network(Optimizer_type opt, Loss_type loss_type);

/**
 * Add a layer to the network.
 * 
 * Layers must be added in order from input to output. Weights and biases
 * are automatically allocated and initialized for each layer (except input).
 * Call this function for each layer you want in your network topology.
 * 
 * Network must have been created with ann_make_network() first.
 * 
 * @param pnet Network to add layer to (must not be NULL)
 * @param node_count Number of neurons in this layer (must be > 0)
 * @param layer_type Type: LAYER_INPUT, LAYER_HIDDEN, or LAYER_OUTPUT
 * @param activation_type Activation: ACTIVATION_NULL, ACTIVATION_SIGMOID, 
 *                        ACTIVATION_RELU, ACTIVATION_SOFTMAX, etc.
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network is NULL
 * @return ERR_INVALID if node_count <= 0 or invalid activation type
 * @return ERR_ALLOC if memory allocation fails
 * 
 * @see ann_make_network()
 * @see ann_train_network()
 */
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);

/**
 * Free all memory associated with a network.
 * 
 * Frees all layer tensors, weights, biases, and network structure.
 * Safe to call with NULL pointer.
 * 
 * @param pnet Network to free
 */
void ann_free_network(PNetwork pnet);

// ============================================================================
// DATA LOADING AND SAVING
// ============================================================================

/**
 * Load training data from CSV file.
 * 
 * CSV format: Each line is one training example, values comma-separated.
 * First line can be header (skip if has_header=CSV_HAS_HEADER).
 * 
 * @param filename Path to CSV file
 * @param has_header CSV_HAS_HEADER to skip first line, CSV_NO_HEADER otherwise
 * @param data Output pointer to allocated data array (caller must free)
 * @param rows Output: number of rows/examples in data
 * @param stride Output: number of columns per row (width of data)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if any parameter pointer is NULL
 * @return ERR_IO if file cannot be opened
 * 
 * Data is allocated in row-major order: data[row * stride + col]
 */
int ann_load_csv(const char *filename, int has_header, real **data, int *rows, int *stride);

/**
 * Save trained network to text file (human-readable).
 * 
 * Format includes version, optimizer, loss type, network topology,
 * and all weights/biases in text form. Can be loaded with ann_load_network().
 * 
 * @param pnet Network to save (must not be NULL)
 * @param filename Output file path
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or filename is NULL
 * @return ERR_IO if file cannot be created/written
 * 
 * @see ann_load_network()
 */
int ann_save_network(const PNetwork pnet, const char *filename);

/**
 * Save trained network to binary file (compact format).
 * 
 * Binary format is more compact than text but less human-readable.
 * Can be loaded with ann_load_network_binary().
 * 
 * @param pnet Network to save (must not be NULL)
 * @param filename Output file path
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or filename is NULL
 * @return ERR_IO if file cannot be created/written
 * 
 * @see ann_load_network_binary()
 */
int ann_save_network_binary(const PNetwork pnet, const char *filename);

/**
 * Export trained network to ONNX JSON format.
 * 
 * Exports the network as an ONNX model in JSON format, which can be used
 * for inference in ONNX-compatible runtimes or converted to binary ONNX.
 * 
 * Supported activations: Sigmoid, ReLU, LeakyReLU, Tanh, Softsign, Softmax.
 * Network is exported as a sequence of MatMul + Add + Activation operations.
 * 
 * @param pnet Network to export (must not be NULL, must have trained weights)
 * @param filename Output file path (typically .onnx.json extension)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or filename is NULL
 * @return ERR_INVALID if network has no layers or unsupported configuration
 * @return ERR_IO if file cannot be created/written
 */
int ann_export_onnx(const PNetwork pnet, const char *filename);

/**
 * Load trained network from text file.
 * 
 * @param filename Path to network file (saved with ann_save_network())
 * @return Pointer to loaded network, or NULL on error
 * 
 * @see ann_save_network()
 */
PNetwork ann_load_network(const char *filename);

/**
 * Load trained network from binary file.
 * 
 * @param filename Path to binary network file (saved with ann_save_network_binary())
 * @return Pointer to loaded network, or NULL on error
 * 
 * @see ann_save_network_binary()
 */
PNetwork ann_load_network_binary(const char *filename);

// ============================================================================
// TRAINING AND INFERENCE
// ============================================================================

/**
 * Train the network on a dataset using mini-batch stochastic gradient descent.
 * 
 * Performs iterative training with backpropagation until convergence or 
 * epoch limit. Updates learning rate and weights based on selected optimizer.
 * 
 * The network topology must be defined (layers added) before calling this.
 * Initial weights are randomized. Call multiple times to continue training
 * with same network instance.
 * 
 * @param pnet Network to train (must have layers defined)
 * @param inputs Training input data (rows x input_size matrix)
 * @param outputs Expected training outputs (rows x output_size matrix)
 * @param rows Number of training examples
 * @return Average loss on final epoch, or 0.0 on error
 * 
 * @see ann_set_convergence() to set convergence threshold
 * @see ann_predict() for inference
 */
real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, int rows);

/**
 * Run trained network on single input to produce output.
 * 
 * Forward-propagates input through all layers and returns the output
 * layer activations. Network must be trained before calling.
 * 
 * @param pnet Trained network (must not be NULL)
 * @param inputs Input feature vector (size = first layer node_count)
 * @param outputs Output buffer to fill with predictions 
 *                (size = last layer node_count)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or data pointers are NULL
 * @return ERR_INVALID if network state is invalid
 */
int ann_predict(const PNetwork pnet, const real *inputs, real *outputs);

/**
 * Determine predicted class from output activations.
 * 
 * Returns the index of the maximum output value, useful for classification
 * when outputs represent class probabilities.
 * 
 * @param outputs Output activation vector from network
 * @param classes Number of classes (length of outputs vector)
 * @return Index of maximum value (class prediction), or -1 on error
 */
int ann_class_prediction(const real *outputs, int classes);

/**
 * Evaluate network accuracy on a dataset.
 * 
 * Runs network on all examples and compares predicted class to expected class.
 * Uses ann_class_prediction() to determine class from outputs.
 * 
 * @param pnet Network to evaluate (must not be NULL)
 * @param inputs Test input data (rows x input_size matrix)
 * @param outputs Expected test outputs (rows x output_size matrix)
 * @return Accuracy as fraction [0.0..1.0], or -1.0 on error
 * 
 * @see ann_predict()
 * @see ann_class_prediction()
 */
real ann_evaluate_accuracy(const PNetwork pnet, const PTensor inputs, const PTensor outputs);

// ============================================================================
// CONFIGURATION AND PROPERTIES
// ============================================================================

/**
 * Set the learning rate for the network.
 * 
 * Learning rate controls step size of weight updates during training.
 * Default: DEFAULT_LEARNING_RATE (0.05)
 * Higher values: faster but less stable learning
 * Lower values: slower but more stable learning
 * 
 * @param pnet Network to configure
 * @param rate Learning rate (suggested range: 0.001 - 0.1)
 */
void ann_set_learning_rate(PNetwork pnet, real rate);

/**
 * Set the loss function used for training.
 * 
 * @param pnet Network to configure
 * @param loss_type LOSS_MSE (regression) or LOSS_CATEGORICAL_CROSS_ENTROPY (classification)
 */
void ann_set_loss_function(PNetwork pnet, Loss_type loss_type);

/**
 * Set the convergence criterion for training.
 * 
 * Training stops when average loss falls below this threshold.
 * Default: DEFAULT_CONVERGENCE (0.01)
 * 
 * @param pnet Network to configure
 * @param limit Loss threshold for convergence
 */
void ann_set_convergence(PNetwork pnet, real limit);

/**
 * Set gradient clipping threshold for training stability.
 * 
 * Clips gradient magnitudes to [-max_grad, max_grad] to prevent
 * exploding gradients, especially useful with ReLU activations.
 * Set to 0 to disable gradient clipping (default).
 * 
 * Recommended values: 1.0 - 5.0 for most networks.
 * 
 * @param pnet Network to configure
 * @param max_grad Maximum gradient magnitude (0 = disabled)
 */
void ann_set_gradient_clip(PNetwork pnet, real max_grad);

/**
 * Set weight initialization strategy.
 * 
 * Controls how initial weights are distributed:
 * - WEIGHT_INIT_UNIFORM: Uniform distribution [-limit, limit]
 * - WEIGHT_INIT_XAVIER: Xavier/Glorot init (good for sigmoid/tanh)
 * - WEIGHT_INIT_HE: He init (good for ReLU variants)
 * - WEIGHT_INIT_AUTO: Automatically choose based on layer activation (default)
 * 
 * Should be called before training. Has no effect after weights are initialized.
 * 
 * @param pnet Network to configure
 * @param init_type Weight initialization strategy
 */
void ann_set_weight_init(PNetwork pnet, Weight_init_type init_type);

/**
 * Set the mini-batch size for training.
 * 
 * Controls how many samples are processed before updating weights.
 * Larger batches: more stable gradients, less noise
 * Smaller batches: faster updates, more regularization effect
 * Default: DEFAULT_BATCH_SIZE (32)
 * 
 * @param pnet Network to configure
 * @param batch_size Number of samples per mini-batch (must be > 0)
 */
void ann_set_batch_size(PNetwork pnet, unsigned batch_size);

/**
 * Set the maximum number of training epochs.
 * 
 * Training stops after this many epochs even if convergence threshold
 * is not reached. Default: 10000
 * 
 * @param pnet Network to configure
 * @param limit Maximum number of epochs
 */
void ann_set_epoch_limit(PNetwork pnet, unsigned limit);

// ============================================================================
// LEARNING RATE SCHEDULERS
// ============================================================================

/**
 * Set a learning rate scheduler callback.
 * 
 * The scheduler is called at the start of each epoch to compute the learning rate.
 * Pass NULL to disable scheduling (use constant learning rate).
 * 
 * @param pnet Network to configure
 * @param scheduler Scheduler callback function, or NULL to disable
 * @param user_data User-provided data passed to scheduler (e.g., parameters)
 * 
 * @see lr_scheduler_step, lr_scheduler_exponential, lr_scheduler_cosine
 */
void ann_set_lr_scheduler(PNetwork pnet, LRSchedulerFunc scheduler, void *user_data);

/**
 * Parameters for step decay scheduler.
 * Reduces learning rate by factor every step_size epochs.
 */
typedef struct {
	unsigned step_size;    // Epochs between LR reductions
	real gamma;            // Multiplicative factor (e.g., 0.5 to halve LR)
} LRStepParams;

/**
 * Parameters for exponential decay scheduler.
 * LR = base_lr * (gamma ^ epoch)
 */
typedef struct {
	real gamma;            // Decay rate per epoch (e.g., 0.95)
} LRExponentialParams;

/**
 * Parameters for cosine annealing scheduler.
 * Smoothly decays LR from base_lr to min_lr over T_max epochs.
 */
typedef struct {
	unsigned T_max;        // Maximum number of epochs (full cycle)
	real min_lr;           // Minimum learning rate at end of cycle
} LRCosineParams;

/**
 * Step decay scheduler: halves LR every step_size epochs.
 * new_lr = base_lr * (gamma ^ (epoch / step_size))
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRStepParams
 * @return Scheduled learning rate
 */
real lr_scheduler_step(unsigned epoch, real base_lr, void *user_data);

/**
 * Exponential decay scheduler: multiplies LR by gamma each epoch.
 * new_lr = base_lr * (gamma ^ epoch)
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRExponentialParams
 * @return Scheduled learning rate
 */
real lr_scheduler_exponential(unsigned epoch, real base_lr, void *user_data);

/**
 * Cosine annealing scheduler: smooth decay to min_lr.
 * new_lr = min_lr + (base_lr - min_lr) * (1 + cos(pi * epoch / T_max)) / 2
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRCosineParams
 * @return Scheduled learning rate
 */
real lr_scheduler_cosine(unsigned epoch, real base_lr, void *user_data);

// ============================================================================
// DROPOUT REGULARIZATION
// ============================================================================

/**
 * Set the default dropout rate for all hidden layers.
 * 
 * Dropout randomly zeros out neurons during training to prevent overfitting.
 * Uses inverted dropout: activations are scaled by 1/(1-rate) during training
 * so no scaling is needed at inference time.
 * 
 * @param pnet Network to configure
 * @param rate Dropout rate (0.0 = disabled, 0.5 = 50% dropout). Must be in [0, 1).
 * 
 * @note Only applies to hidden layers; input and output layers are not affected.
 * @note Use ann_set_layer_dropout() to override the rate for specific layers.
 * @see ann_set_layer_dropout, ann_set_training_mode
 */
void ann_set_dropout(PNetwork pnet, real rate);

/**
 * Set the dropout rate for a specific layer.
 * 
 * Overrides the default dropout rate for this layer.
 * 
 * @param pnet Network to configure
 * @param layer Layer index (0 = input layer)
 * @param rate Dropout rate (0.0 = disabled, 0.5 = 50% dropout). Must be in [0, 1).
 * 
 * @note Setting dropout on input or output layers has no effect.
 */
void ann_set_layer_dropout(PNetwork pnet, int layer, real rate);

/**
 * Set training/inference mode for the network.
 * 
 * In training mode (is_training=1), dropout is applied.
 * In inference mode (is_training=0), dropout is disabled.
 * 
 * Training mode is automatically enabled during ann_train_network()
 * and disabled after training completes.
 * 
 * @param pnet Network to configure
 * @param is_training 1 for training mode, 0 for inference mode
 */
void ann_set_training_mode(PNetwork pnet, int is_training);

/**
 * Get the number of layers in the network.
 * 
 * @param pnet Network to query (must not be NULL)
 * @return Number of layers, or -1 if pnet is NULL
 */
int ann_get_layer_count(const PNetwork pnet);

/**
 * Get the number of nodes in a specific layer.
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return Number of nodes in the layer, or -1 on error
 */
int ann_get_layer_nodes(const PNetwork pnet, int layer);

/**
 * Get the activation type of a specific layer.
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return Activation type, or ACTIVATION_NULL on error
 */
Activation_type ann_get_layer_activation(const PNetwork pnet, int layer);

/**
 * Print network properties and configuration to stdout.
 * 
 * Displays: network topology, optimizer, loss function, mini-batch size, etc.
 * 
 * @param pnet Network to describe (must not be NULL)
 */
void ann_print_props(const PNetwork pnet);

// ============================================================================
// DEBUGGING AND INSPECTION
// ============================================================================

/**
 * Print the output layer activations to stdout.
 * 
 * Useful for debugging to see network predictions.
 * 
 * @param pnet Network whose outputs to print (must not be NULL)
 */
void print_outputs(const PNetwork pnet);

// ============================================================================
// ERROR HANDLING
// ============================================================================

/**
 * Convert an error code to a human-readable error message.
 * 
 * Maps error codes (ERR_OK, ERR_NULL_PTR, etc.) to descriptive strings
 * for better error reporting and debugging.
 * 
 * @param error_code Error code to convert (use error codes defined above)
 * @return Pointer to static error message string, never NULL
 * 
 * Usage:
 *   int result = ann_add_layer(net, 10, ACTIVATION_RELU);
 *   if (result != ERR_OK) {
 *       fprintf(stderr, "Error: %s\n", ann_strerror(result));
 *   }
 * 
 * @see ERR_OK ERR_NULL_PTR ERR_ALLOC ERR_INVALID ERR_IO ERR_FAIL
 */
const char* ann_strerror(int error_code);

/**
 * Set the error logging callback for library errors.
 * 
 * Installs a callback function that will be called whenever an error occurs
 * in the library. Enables integration with custom logging systems, monitoring,
 * alerting, or debugging frameworks.
 * 
 * The callback is called with:
 *   - error_code: Numeric error code (ERR_NULL_PTR, ERR_ALLOC, etc.)
 *   - error_message: Human-readable message from ann_strerror()
 *   - function_name: Name of the function that generated the error
 * 
 * To disable logging, pass NULL as the callback parameter.
 * 
 * @param callback Function pointer to error handler, or NULL to disable
 * 
 * Example:
 *   void error_handler(int code, const char *msg, const char *func) {
 *       fprintf(stderr, "[ANN Error in %s] %s\n", func, msg);
 *       // Could also log to file, send to monitoring service, etc.
 *   }
 *   ann_set_error_log_callback(error_handler);
 * 
 * @see ErrorLogCallback
 */
void ann_set_error_log_callback(ErrorLogCallback callback);

/**
 * Get the currently installed error logging callback.
 * 
 * @return Current callback function pointer, or NULL if none is set
 */
ErrorLogCallback ann_get_error_log_callback(void);

/**
 * Clear (disable) the error logging callback.
 * 
 * Equivalent to calling ann_set_error_log_callback(NULL).
 * After this call, library errors will not trigger callbacks.
 * 
 * @see ann_set_error_log_callback
 */
void ann_clear_error_log_callback(void);

#endif
