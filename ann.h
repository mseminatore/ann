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

// ANN_API is defined in tensor.h

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

#define MAX_HISTORY_SIZE 10000		// maximum epochs to store in learning curve history

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
// Layer kinds: what a layer computes (Layer_type is where it sits).
// New kinds are appended so saved model values stay stable.
//------------------------------
typedef enum {
	LAYER_KIND_DENSE,					// fully connected: y = act(W x + b)
	LAYER_KIND_CONV2D,					// 2-D convolution over a (C, H, W) input
	LAYER_KIND_MAXPOOL2D,				// 2-D max pooling, per channel
	LAYER_KIND_FLATTEN,					// reshape (C, H, W) to (C*H*W, 1, 1)
	LAYER_KIND_UPSAMPLE2D,				// nearest-neighbour upsampling by an integer scale, per channel
	LAYER_KIND_RESHAPE,					// give the same values a new (C, H, W) shape
	LAYER_KIND_GLOBAL_AVGPOOL2D,		// each channel's mean: (C, H, W) to (C, 1, 1)
	LAYER_KIND_BATCHNORM,				// normalize each channel over the batch, then scale and shift
	LAYER_KIND_EMBEDDING,				// token ids to learned vectors: L ids to the sequence (D, 1, L)
	LAYER_KIND_GLOBAL_MAXPOOL2D			// each channel's largest value: (C, H, W) to (C, 1, 1)
} Layer_kind;

//------------------------------
// Layer output shape: channels x height x width. Activations are stored
// row-major as C*H*W values per sample; dense layers are (nodes, 1, 1).
//------------------------------
typedef struct {
	int c, h, w;
} Layer_shape;

#define ANN_LAYER_MAX_PARAMS	4		// trainable tensors per layer (dense uses 2: W, b)
#define ANN_LAYER_CONFIG_MAX	8		// kind-specific settings (kernel size, stride, ...)
#define ANN_BATCHNORM_EPSILON	1e-5	// batch norm's epsilon: y = gamma * (x - mean) / sqrt(var + epsilon) + beta

//------------------------------
// One trainable tensor of a layer with its gradient and optimizer state.
// The layer owns these tensors.
//------------------------------
typedef struct {
	PTensor value;						// the parameter
	PTensor grad;						// accumulated gradient (-dL/dvalue)
	PTensor m;							// first moment (Momentum, Adam)
	PTensor v;							// second moment (AdaGrad, RMSProp, Adam)
	int regularize;						// 1 if L1/L2 apply (weights, not biases)
	int state;							// 1 for a running statistic (batch norm mean, variance):
										// saved, uploaded and restored like the others, never trained
} Layer_param;

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
	ACTIVATION_SOFTMAX,
	ACTIVATION_GELU						// tanh approximation; appended to keep saved model values stable
} Activation_type;

//------------------------------
// Loss function types
//------------------------------
typedef enum {
	LOSS_MSE,							// regression; use ACTIVATION_NULL on the output for unbounded targets
	LOSS_CATEGORICAL_CROSS_ENTROPY,		// softmax output: one class per sample; sigmoid output: binary/multi-label
	LOSS_BINARY_CROSS_ENTROPY = LOSS_CATEGORICAL_CROSS_ENTROPY,	// alias: use with a sigmoid output
	LOSS_DEFAULT = LOSS_MSE
} Loss_type;

//------------------------------
// Noise added to the inputs of training batches (e.g. for denoising
// autoencoders); see ann_set_input_noise()
//------------------------------
typedef enum {
	NOISE_NONE,							// no noise (default)
	NOISE_GAUSSIAN,						// add normal noise with standard deviation `amount`
	NOISE_MASKING,						// set each input to 0 with probability `amount`
	NOISE_SALT_PEPPER					// set each input to low or high (equally likely) with probability `amount`
} Noise_type;

//------------------------------
// SGD optimization kernels
//------------------------------
typedef enum {
	OPT_SGD,
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
typedef void(*Output_func) (const char *);
typedef void(*Optimization_func) (PNetwork pnet);
typedef real(*Activation_func) (real);

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
	int node_count;						// number of nodes in layer (shape.c * shape.h * shape.w)
	Layer_kind kind;					// what the layer computes
	Layer_shape shape;					// output shape
	Layer_shape in_shape;				// shape of the layer below (set when the layer is added)
	int config[ANN_LAYER_CONFIG_MAX];	// kind-specific settings (unused by dense)

	Layer_type layer_type;				// type of this layer
	Activation_type activation;			// type of activation, none, sigmoid, Relu
	Activation_func activation_func;	// node activation function

	real dropout_rate;					// dropout rate for this layer (0.0 = disabled)
	PTensor t_dropout_mask;				// dropout mask tensor (for batched training)

	PTensor t_scratch[2];				// kind work buffers (conv: im2col columns and their gradient)

	PTensor t_values;					// tensor of node values for the layer

	// Trainable parameters (none on the input layer). For dense layers the
	// named tensors below are the same tensors as params[0] (W) and params[1] (b).
	int param_count;
	Layer_param params[ANN_LAYER_MAX_PARAMS];

	// Parameters that produce this layer from the layer below (NULL on the input layer)
	PTensor t_weights;					// weights [node_count x previous layer's node_count]
	PTensor t_v;						// tensor of velocities for optimizer
	PTensor t_m;						// tensor of momentums for optimizer
	PTensor t_gradients;				// tensor of gradients for back propagation
	PTensor t_bias;						// bias vector [1 x node_count]
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
	int gpu_stale;						// 1 = CPU weights changed since the last GPU upload; GPU copy is not used

	real l2_lambda;						// L2 regularization (weight decay) coefficient (0 = disabled)
	real l1_lambda;						// L1 regularization (LASSO) coefficient (0 = disabled)

	Output_func print_func;				// print output function
	Optimization_func optimize_func;	// learning rate/weight optimizer

	LRSchedulerFunc lr_scheduler;		// learning rate scheduler callback
	void *lr_scheduler_data;			// user data for scheduler

	unsigned current_batch_size;		// current allocated batch size for batch tensors

	// Training history for learning curve export
	real *loss_history;					// loss at each epoch (dynamically allocated)
	real *lr_history;					// learning rate at each epoch
	unsigned history_count;				// number of epochs recorded
	unsigned history_capacity;			// allocated capacity
	real *val_loss_history;				// validation loss at each epoch (-1 = not measured)

	// Validation and early stopping
	PTensor val_inputs;					// validation inputs (borrowed, not owned)
	PTensor val_outputs;				// validation targets (borrowed, not owned)
	unsigned es_patience;				// epochs without improvement before stopping (0 = disabled)
	real es_min_delta;					// minimum loss decrease that counts as an improvement
	int es_restore_best;				// restore the best epoch's weights when training ends
	unsigned best_epoch;				// epoch with the lowest monitored loss in the last run (0 = none)
	unsigned epochs_trained;			// epochs run by the last ann_train_network() call

	// Input noise for training batches (see ann_set_input_noise)
	Noise_type noise_type;				// NOISE_NONE when disabled
	real noise_amount;					// standard deviation, or probability per input
	real noise_low, noise_high;			// clamp range when low < high; salt-and-pepper values
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
ANN_API PNetwork ann_make_network(Optimizer_type opt, Loss_type loss_type);

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
ANN_API int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);

/**
 * Add an input layer for image-like data of shape (channels, height, width).
 * 
 * Each sample is C*H*W values stored channel by channel, row by row
 * (C, H, W order, as in ONNX/PyTorch). Use it instead of
 * ann_add_layer(..., LAYER_INPUT, ...) when convolution or pooling layers follow.
 * 
 * @param pnet Network to add the layer to (must be empty)
 * @param channels Number of channels (e.g. 1 for grayscale)
 * @param height Image height
 * @param width Image width
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad size or not the first layer) or ERR_ALLOC
 */
ANN_API int ann_add_input(PNetwork pnet, int channels, int height, int width);

/**
 * Add a 2-D convolution layer with square kernels and zero padding.
 * 
 * Output shape is (filters, OH, OW) with OH = (H + 2*padding - kernel_size) / stride + 1,
 * and likewise for OW. Weights have shape [filters, C, kernel_size, kernel_size].
 * 
 * @param pnet Network to add the layer to
 * @param filters Number of output channels (> 0)
 * @param kernel_size Kernel height and width (> 0)
 * @param stride Step between windows (> 0)
 * @param padding Zeros added on each side (>= 0); (kernel_size - 1) / 2 keeps the size at stride 1
 * @param activation Activation applied to the output (not ACTIVATION_SOFTMAX)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings or the kernel does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_conv2d(PNetwork pnet, int filters, int kernel_size, int stride, int padding,
                           Activation_type activation);

/**
 * Add a grouped 2-D convolution layer: the input channels and the filters
 * are split into groups, and each group's filters see only that group's
 * channels.
 * 
 * Like ann_add_conv2d (groups = 1), but weights have shape
 * [filters, C / groups, kernel_size, kernel_size]: groups times fewer weights
 * and multiply-adds. groups = C is a depthwise convolution (see
 * ann_add_depthwise_conv2d). Maps to an ONNX Conv node with a group attribute.
 * 
 * @param pnet Network to add the layer to
 * @param filters Number of output channels (> 0, a multiple of groups)
 * @param kernel_size Kernel height and width (> 0)
 * @param stride Step between windows (> 0)
 * @param padding Zeros added on each side (>= 0)
 * @param groups Number of groups (> 0, dividing the input channels and filters)
 * @param activation Activation applied to the output (not ACTIVATION_SOFTMAX)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings or the kernel does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_conv2d_grouped(PNetwork pnet, int filters, int kernel_size, int stride, int padding,
                                   int groups, Activation_type activation);

/**
 * Add a depthwise 2-D convolution layer: each input channel is convolved
 * with its own depth_multiplier filters, and channels are not mixed.
 * 
 * The first half of a depthwise separable convolution (MobileNet): follow it
 * with a 1x1 ann_add_conv2d to mix the channels, with batch norm and an
 * activation after each. Output shape is (C * depth_multiplier, OH, OW).
 * The same as ann_add_conv2d_grouped with groups = C.
 * 
 * @param pnet Network to add the layer to
 * @param depth_multiplier Filters per input channel (> 0; usually 1)
 * @param kernel_size Kernel height and width (> 0)
 * @param stride Step between windows (> 0)
 * @param padding Zeros added on each side (>= 0)
 * @param activation Activation applied to the output (not ACTIVATION_SOFTMAX)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings, first layer, or the kernel does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_depthwise_conv2d(PNetwork pnet, int depth_multiplier, int kernel_size, int stride, int padding,
                                     Activation_type activation);

/**
 * Add a 2-D convolution layer with any window: separate kernel size, stride
 * and padding for the height and the width, and groups.
 * 
 * Output shape is (filters, OH, OW) with OH = (H + 2*padding_h - kernel_h) / stride_h + 1
 * and OW = (W + 2*padding_w - kernel_w) / stride_w + 1. Weights have shape
 * [filters, C / groups, kernel_h, kernel_w]. ann_add_conv2d, ann_add_conv2d_grouped
 * and ann_add_conv1d are special cases.
 * 
 * @param pnet Network to add the layer to
 * @param filters Number of output channels (> 0, a multiple of groups)
 * @param kernel_h Kernel height (> 0)
 * @param kernel_w Kernel width (> 0)
 * @param stride_h Step between windows down the height (> 0)
 * @param stride_w Step between windows along the width (> 0)
 * @param padding_h Zeros added above and below (>= 0)
 * @param padding_w Zeros added left and right (>= 0)
 * @param groups Number of groups (> 0, dividing the input channels and filters); 1 for an ordinary conv
 * @param activation Activation applied to the output (not ACTIVATION_SOFTMAX)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings or the kernel does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_conv2d_ex(PNetwork pnet, int filters, int kernel_h, int kernel_w, int stride_h, int stride_w,
                              int padding_h, int padding_w, int groups, Activation_type activation);

/**
 * Add a 1-D convolution layer, for sequences such as time series.
 * 
 * A sequence of L steps with C features (channels) per step is the image
 * (C, 1, L): ann_add_input(pnet, C, 1, L). The layer slides a window of
 * kernel_size steps along it; output shape is (filters, 1, OL) with
 * OL = (L + 2*padding - kernel_size) / stride + 1. The same as
 * ann_add_conv2d_ex with a 1 x kernel_size window; maps to an ONNX Conv node
 * with that window (and imports PyTorch's nn.Conv1d).
 * 
 * @param pnet Network to add the layer to
 * @param filters Number of output channels (> 0)
 * @param kernel_size Window length in steps (> 0)
 * @param stride Step between windows (> 0)
 * @param padding Zeros added at each end (>= 0); (kernel_size - 1) / 2 keeps the length at stride 1
 * @param activation Activation applied to the output (not ACTIVATION_SOFTMAX)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings or the kernel does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_conv1d(PNetwork pnet, int filters, int kernel_size, int stride, int padding,
                           Activation_type activation);

/**
 * Add a 2-D max pooling layer (no padding), applied to each channel.
 * 
 * Output shape is (C, OH, OW) with OH = (H - pool_size) / stride + 1, and likewise for OW.
 * 
 * @param pnet Network to add the layer to
 * @param pool_size Window height and width (> 0)
 * @param stride Step between windows (> 0); usually equal to pool_size
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings or the window does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_maxpool2d(PNetwork pnet, int pool_size, int stride);

/**
 * Add a 2-D max pooling layer with separate window sizes and strides for the
 * height and the width (no padding).
 * 
 * @param pnet Network to add the layer to
 * @param pool_h Window height (> 0)
 * @param pool_w Window width (> 0)
 * @param stride_h Step between windows down the height (> 0)
 * @param stride_w Step between windows along the width (> 0)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings, first layer, or the window does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_maxpool2d_ex(PNetwork pnet, int pool_h, int pool_w, int stride_h, int stride_w);

/**
 * Add a 1-D max pooling layer over sequences (C, 1, L): windows of pool_size
 * steps, stride steps apart. The same as ann_add_maxpool2d_ex with a
 * 1 x pool_size window.
 * 
 * @param pnet Network to add the layer to
 * @param pool_size Window length in steps (> 0)
 * @param stride Step between windows (> 0)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad settings, first layer, or the window does not fit) or ERR_ALLOC
 */
ANN_API int ann_add_maxpool1d(PNetwork pnet, int pool_size, int stride);

/**
 * Add a flatten layer: reshape (C, H, W) to (C*H*W, 1, 1).
 * 
 * Optional: a dense layer added after a convolution or pooling layer reads
 * its output as a flat vector anyway. It makes the model's structure explicit
 * and maps to an ONNX Flatten node.
 * 
 * @param pnet Network to add the layer to
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer) or ERR_ALLOC
 */
ANN_API int ann_add_flatten(PNetwork pnet);

/**
 * Add a 2-D nearest-neighbour upsampling layer, applied to each channel.
 * 
 * Output shape is (C, H*scale, W*scale): each input value is repeated in a
 * scale x scale block. The decoder's counterpart to max pooling, e.g. in
 * autoencoders. No parameters; maps to an ONNX Resize node.
 * 
 * @param pnet Network to add the layer to
 * @param scale Size factor for height and width (> 0)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad scale or first layer) or ERR_ALLOC
 */
ANN_API int ann_add_upsample2d(PNetwork pnet, int scale);

/**
 * Add a reshape layer: give the previous layer's values the shape (C, H, W).
 * 
 * Values are stored flat, so nothing moves; this turns, for example, a dense
 * layer of C*H*W nodes into an image that convolution, pooling and
 * upsampling layers can use. No parameters; maps to an ONNX Reshape node.
 * 
 * @param pnet Network to add the layer to
 * @param channels Number of channels
 * @param height Height
 * @param width Width
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer, or C*H*W differs from
 *         the previous layer's size) or ERR_ALLOC
 */
ANN_API int ann_add_reshape(PNetwork pnet, int channels, int height, int width);

/**
 * Add a global average pooling layer: each channel's mean over H x W.
 * 
 * Output shape is (C, 1, 1), so a dense layer can follow directly (no
 * flatten). Used in place of flatten + a large dense layer at the end of a
 * CNN, it removes most of the parameters, which keeps quantized and exported
 * models small. No parameters; maps to an ONNX GlobalAveragePool node.
 * 
 * @param pnet Network to add the layer to
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer) or ERR_ALLOC
 */
ANN_API int ann_add_global_avgpool2d(PNetwork pnet);

/**
 * Add a batch normalization layer, then an optional activation.
 * 
 * Normalizes each channel (each node after a dense layer) to zero mean and
 * unit variance, then scales and shifts it by learned gamma and beta:
 * y = act(gamma * (x - mean) / sqrt(var + 1e-5) + beta). Training uses the
 * batch's mean and variance and updates running averages of them (momentum
 * 0.1, as PyTorch's default); prediction and evaluation use the running
 * averages. The output has the input's shape.
 * 
 * Put it between a layer and its activation, e.g.
 * ann_add_conv2d(net, 16, 3, 1, 1, ACTIVATION_NULL); ann_add_batchnorm(net, ACTIVATION_RELU);
 * so deeper networks train faster and accept higher learning rates. Train
 * with batches of more than one sample. Maps to an ONNX BatchNormalization
 * node; int8 quantization and C export fold it into the dense or conv layer
 * before it.
 * 
 * @param pnet Network to add the layer to
 * @param activation Activation applied after normalizing (ACTIVATION_NULL for none)
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer or bad activation) or ERR_ALLOC
 */
ANN_API int ann_add_batchnorm(PNetwork pnet, Activation_type activation);

/**
 * Add an embedding layer: map token ids to learned vectors.
 * 
 * The layer below (usually the input layer) holds L token ids per sample,
 * as reals rounded to the nearest integer. The output is the sequence
 * (embed_dim, 1, L): channel d at step t is row id_t of the weights
 * [vocab_size x embed_dim], so ann_add_conv1d and the pooling layers follow
 * directly. Ids outside [0, vocab_size) and padding_idx give zero vectors and
 * train nothing; padding_idx's row stays zero. Weights start N(0, 1), as
 * PyTorch's nn.Embedding. No gradient reaches the layer below. Maps to ONNX
 * Cast, Gather, Transpose and Unsqueeze nodes (PyTorch's nn.Embedding imports).
 * Networks with an embedding cannot be quantized or exported as C headers yet,
 * and train on the CPU.
 * 
 * @param pnet Network to add the layer to
 * @param vocab_size Number of token ids (> 0)
 * @param embed_dim Length of each vector (> 0)
 * @param padding_idx Id that pads sequences (its vector is always zero), or -1 for none
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer, a layer below that is not flat, or bad
 *         settings) or ERR_ALLOC
 */
ANN_API int ann_add_embedding(PNetwork pnet, int vocab_size, int embed_dim, int padding_idx);

/**
 * Add a global max pooling layer: each channel's largest value over H x W.
 * 
 * Output shape is (C, 1, 1), so a dense layer can follow directly. Over the
 * steps of a sequence it keeps the strongest response of each feature
 * wherever it occurs, the usual pooling for text after an embedding and 1-D
 * convolutions. Gradients go to the input that was largest (the first on
 * ties). No parameters; maps to an ONNX GlobalMaxPool node.
 * 
 * @param pnet Network to add the layer to
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (first layer) or ERR_ALLOC
 */
ANN_API int ann_add_global_maxpool2d(PNetwork pnet);

/**
 * Free all memory associated with a network.
 * 
 * Frees all layer tensors, weights, biases, and network structure.
 * Safe to call with NULL pointer.
 * 
 * @param pnet Network to free
 */
ANN_API void ann_free_network(PNetwork pnet);

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
ANN_API int ann_load_csv(const char *filename, int has_header, real **data, int *rows, int *stride);

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
ANN_API int ann_save_network(const PNetwork pnet, const char *filename);

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
ANN_API int ann_save_network_binary(const PNetwork pnet, const char *filename);

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
ANN_API int ann_export_onnx(const PNetwork pnet, const char *filename);

/**
 * Import a network from ONNX JSON file.
 * 
 * Loads a neural network from the JSON format produced by ann_export_onnx().
 * Only supports sequential dense networks with supported activations.
 * 
 * @param filename Path to ONNX JSON file
 * @return Pointer to loaded network, or NULL on error
 * 
 * Supported ONNX operations:
 * - MatMul (dense layer weights)
 * - Add (bias)
 * - Sigmoid, Relu, LeakyRelu, Tanh, Softsign, Softmax
 * 
 * Unsupported operations (Conv, Pool, etc.) will cause import to fail.
 * 
 * @see ann_export_onnx()
 */
ANN_API PNetwork ann_import_onnx(const char *filename);

// ============================================================================
// INT8 QUANTIZATION AND C HEADER EXPORT (ann_quantize.c)
// ============================================================================

/**
 * An int8 version of a trained network (opaque). Weights are int8 with one
 * scale per output channel; activations are int8 with one scale and zero
 * point per tensor; sums are int32. See docs/C_EXPORT.md.
 */
typedef struct QuantizedNetwork QuantizedNetwork;
typedef struct QuantizedNetwork *PQuantizedNetwork;

/**
 * Quantize a trained network to int8.
 *
 * Runs the network over the calibration rows to find each layer's value
 * range; a few hundred representative training samples are usually enough.
 * Supports dense, conv2d, maxpool2d and flatten layers with any activation
 * (softmax only on the output layer). The network is not changed.
 *
 * @param pnet Trained network
 * @param calibration_inputs Input rows [samples x input nodes]
 * @return The int8 model (free with ann_free_quantized), or NULL on error
 */
ANN_API PQuantizedNetwork ann_quantize(const PNetwork pnet, const PTensor calibration_inputs);

/**
 * Free a model made by ann_quantize(). NULL is ignored.
 */
ANN_API void ann_free_quantized(PQuantizedNetwork q);

/**
 * Run the int8 model on real inputs: the inputs are quantized, the model
 * runs in integer math, and the outputs are dequantized (with softmax
 * applied if the network's output layer has it). Gives the same results as
 * the exported header's <prefix>_predict_float().
 *
 * @return ERR_OK, ERR_NULL_PTR or ERR_INVALID
 */
ANN_API int ann_quantized_predict(const PQuantizedNetwork q, const real *inputs, real *outputs);

/**
 * Run the int8 model on int8 inputs, giving int8 outputs (softmax outputs
 * are the int8 logits). Gives the same results as the exported header's
 * <prefix>_predict().
 *
 * @return ERR_OK, ERR_NULL_PTR or ERR_INVALID
 */
ANN_API int ann_quantized_predict_int8(const PQuantizedNetwork q, const signed char *inputs, signed char *outputs);

/**
 * The int8 model's input and output quantization: real = scale * (q - zero_point).
 * Any output pointer may be NULL.
 *
 * @return ERR_OK or ERR_NULL_PTR
 */
ANN_API int ann_quantized_get_io_params(const PQuantizedNetwork q, real *in_scale, int *in_zero_point,
                                        real *out_scale, int *out_zero_point);

/**
 * Classification accuracy of the int8 model (compare with ann_evaluate_accuracy).
 *
 * @return Accuracy in [0, 1], or -1 on bad arguments
 */
ANN_API real ann_quantized_evaluate_accuracy(const PQuantizedNetwork q, const PTensor inputs, const PTensor outputs);

/**
 * Export a model as a self-contained C header for embedded targets.
 *
 * With q, writes the int8 model; with q NULL, writes pnet's float32 weights.
 * The header holds the parameters as static const arrays and small inference
 * functions (<prefix>_predict, <prefix>_argmax) that run on ann_rt.h, a
 * header-only runtime with no allocation. Copy ann_rt.h next to the header.
 *
 * @param pnet Network to export as float32 (ignored when q is given)
 * @param q int8 model from ann_quantize(), or NULL
 * @param filename Output file, e.g. "model.h"
 * @param prefix C identifier used to name the header's symbols, e.g. "model"
 * @return ERR_OK, ERR_NULL_PTR, ERR_INVALID (bad prefix, unsupported network) or ERR_IO
 */
ANN_API int ann_export_c_header(const PNetwork pnet, const PQuantizedNetwork q, const char *filename,
                                const char *prefix);

/**
 * Export network architecture as PIKCHR diagram.
 * 
 * Generates a PIKCHR text file that can be rendered to SVG using
 * the pikchr tool: `pikchr network.pikchr > network.svg`
 * 
 * For small networks (≤10 nodes per layer), outputs detailed diagram
 * with individual node circles and connection lines.
 * For larger networks, outputs simplified box diagram with layer info.
 * 
 * @param pnet Network to export (must not be NULL)
 * @param filename Output file path (typically .pikchr extension)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or filename is NULL
 * @return ERR_INVALID if network has no layers
 * @return ERR_IO if file cannot be created/written
 */
ANN_API int ann_export_pikchr(const PNetwork pnet, const char *filename);

/**
 * Export training history as CSV for learning curve visualization.
 * 
 * Writes epoch, loss, and learning rate data collected during training.
 * When a validation set was used, a val_loss column is added (empty for
 * epochs trained without one).
 * Use external tools (gnuplot, matplotlib, Excel) to plot the curve.
 * 
 * @param pnet Network with training history (must have been trained)
 * @param filename Output CSV file path
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or filename is NULL
 * @return ERR_INVALID if no training history available
 * @return ERR_IO if file cannot be created/written
 */
ANN_API int ann_export_learning_curve(const PNetwork pnet, const char *filename);

/**
 * Clear training history to free memory or before retraining.
 * 
 * @param pnet Network to clear history from
 */
ANN_API void ann_clear_history(PNetwork pnet);

/**
 * Get the number of epochs in the training history.
 *
 * ann_train_network() appends one entry per epoch; the history accumulates
 * across runs until ann_clear_history() is called.
 *
 * @param pnet Network to query
 * @return Number of recorded epochs, or 0 if pnet is NULL or untrained
 *
 * @see ann_get_history()
 */
ANN_API unsigned ann_get_history_count(const PNetwork pnet);

/**
 * Copy the training history into caller-provided arrays.
 *
 * Copies the first min(max_count, ann_get_history_count()) epochs. Any of the
 * array pointers may be NULL to skip that series. val_loss entries are -1 for
 * epochs trained without validation data.
 *
 * @param pnet Network to query
 * @param loss Training loss per epoch, or NULL
 * @param lr Learning rate per epoch, or NULL
 * @param val_loss Validation loss per epoch, or NULL
 * @param max_count Capacity of each non-NULL array
 * @return Number of epochs copied (0 if pnet is NULL)
 *
 * @see ann_get_history_count(), ann_export_learning_curve()
 */
ANN_API unsigned ann_get_history(const PNetwork pnet, real *loss, real *lr,
								 real *val_loss, unsigned max_count);

/**
 * Load trained network from text file.
 * 
 * @param filename Path to network file (saved with ann_save_network())
 * @return Pointer to loaded network, or NULL on error
 * 
 * @see ann_save_network()
 */
ANN_API PNetwork ann_load_network(const char *filename);

/**
 * Load trained network from binary file.
 * 
 * @param filename Path to binary network file (saved with ann_save_network_binary())
 * @return Pointer to loaded network, or NULL on error
 * 
 * @see ann_save_network_binary()
 */
ANN_API PNetwork ann_load_network_binary(const char *filename);

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
ANN_API real ann_train_network(PNetwork pnet, PTensor inputs, PTensor outputs, int rows);

/**
 * Begin an online/incremental training session.
 * 
 * Prepares the network for step-by-step training without resetting
 * optimizer state (e.g. Adam momentum). Weights are initialized only 
 * if not already set (safe for loaded/pre-trained models).
 * 
 * Call ann_train_step() to train on individual mini-batches, then
 * ann_train_end() when finished.
 * 
 * @param pnet Network to train (must have layers defined)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if pnet is NULL
 * @return ERR_INVALID if network has no layers
 * @return ERR_ALLOC if batch tensor allocation fails
 * 
 * @see ann_train_step() to train on a single mini-batch
 * @see ann_train_end() to finish the training session
 */
ANN_API int ann_train_begin(PNetwork pnet);

/**
 * Train on a single mini-batch (online/incremental training).
 * 
 * Performs one forward pass, backward pass, and weight update on the
 * provided mini-batch. Does not reset optimizer state between calls,
 * enabling incremental learning on streaming data.
 * 
 * Must be called between ann_train_begin() and ann_train_end().
 * 
 * @param pnet Network being trained
 * @param inputs Input data (batch_size consecutive input vectors, each of
 *               size = first layer node_count)
 * @param targets Target data (batch_size consecutive target vectors, each of
 *                size = last layer node_count)
 * @param batch_size Number of samples in this mini-batch
 * @return Loss for this mini-batch, or 0.0 on error
 * 
 * @see ann_train_begin() to start a training session
 * @see ann_train_end() to finish the training session
 */
ANN_API real ann_train_step(PNetwork pnet, const real *inputs, const real *targets, int batch_size);

/**
 * End an online/incremental training session.
 * 
 * Disables training mode (stops dropout from being applied).
 * The network is ready for inference after this call.
 * 
 * @param pnet Network that was being trained
 * 
 * @see ann_train_begin() to start a training session
 */
ANN_API void ann_train_end(PNetwork pnet);

/**
 * Run trained network on single input to produce output.
 * 
 * Forward-propagates input through all layers and returns the output
 * layer activations. Safe to call during online training (between
 * ann_train_begin/ann_train_end) — dropout is automatically disabled.
 * 
 * @param pnet Trained network (must not be NULL)
 * @param inputs Input feature vector (size = first layer node_count)
 * @param outputs Output buffer to fill with predictions 
 *                (size = last layer node_count)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if network or data pointers are NULL
 * @return ERR_INVALID if network state is invalid
 */
ANN_API int ann_predict(const PNetwork pnet, const real *inputs, real *outputs);

// ============================================================================
// GPU ACCELERATION (requires USE_METAL, USE_CUDA or USE_VULKAN; see ann_config.h)
// ============================================================================

/**
 * Initialize the GPU compute context.
 *
 * Starts the first compiled-in GPU backend that initializes, trying Metal
 * (macOS), Vulkan (AMD, Intel, NVIDIA) and CUDA (NVIDIA) in that order; the
 * ANN_GPU environment variable ("metal", "vulkan", "cuda" or "none") limits
 * it to one backend. Creates the device, command queue, and compiles/loads kernels.
 * Must be called before any other ann_gpu_* function.
 *
 * @return 1 on success, 0 if no GPU backend is available or initialization fails
 */
ANN_API int ann_gpu_init(void);

/**
 * Upload network weights and biases to the GPU.
 *
 * Copies all layer weights and biases into GPU buffers.
 * Must be called after the network is fully trained or loaded.
 * After this call, ann_predict() and ann_predict_batch() will use the GPU path.
 *
 * @param pnet Trained network (must have weights initialized)
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if pnet is NULL or GPU not initialized
 * @return ERR_FAIL if any GPU buffer allocation fails
 */
ANN_API int ann_gpu_upload_network(PNetwork pnet);

/**
 * Release all GPU buffers associated with the network.
 *
 * Frees GPU buffers for all layer weights and biases. After this call,
 * ann_predict() falls back to CPU inference.
 * Safe to call even if the network was never uploaded to GPU.
 *
 * @param pnet Network whose GPU buffers should be freed
 */
ANN_API void ann_gpu_free_network(PNetwork pnet);

/**
 * Run inference on many samples at once.
 * 
 * Performs forward propagation for `batch_size` samples. When the network is
 * uploaded to a GPU (ann_gpu_init() and ann_gpu_upload_network()) the batch
 * runs there; otherwise it runs on the CPU's batched forward pass, in chunks
 * of up to 256 rows (the training batch size while training is in progress).
 * Either way it is much faster than calling ann_predict() per sample.
 * Dropout is never applied.
 * 
 * @param pnet       Trained network
 * @param inputs     Row-major input array [batch_size × input_node_count]
 * @param outputs    Row-major output array [batch_size × output_node_count] (written)
 * @param batch_size Number of samples to process simultaneously
 * @return ERR_OK on success
 * @return ERR_NULL_PTR if any pointer is NULL or batch_size <= 0
 * @return ERR_FAIL if GPU is not initialized or computation fails
 * @return ERR_ALLOC if GPU buffer allocation fails
 */
ANN_API int ann_predict_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size);

/**
 * Download updated weights and biases from GPU back to CPU.
 *
 * ann_train_network() already does this when it finishes, and the online
 * API keeps both copies current, so explicit calls are only needed after
 * driving the GPU some other way. Does nothing when the CPU weights are newer
 * than the GPU copy (e.g. mid online-training session), so it never
 * overwrites them with stale values.
 *
 * Safe to call even if no GPU training occurred.
 *
 * @param pnet Network whose weights should be downloaded from GPU
 */
ANN_API void ann_gpu_sync_weights(PNetwork pnet);

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
ANN_API int ann_class_prediction(const real *outputs, int classes);

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
ANN_API real ann_evaluate_accuracy(const PNetwork pnet, const PTensor inputs, const PTensor outputs);

/**
 * Evaluate the network's loss on a dataset.
 * 
 * Uses the network's loss function (MSE, or cross-entropy) averaged the same
 * way as the training loss, so the two are directly comparable.
 * Dropout is not applied.
 * 
 * @param pnet Network to evaluate (must not be NULL)
 * @param inputs Input data (rows x input_size matrix)
 * @param outputs Expected outputs (rows x output_size matrix)
 * @return Average loss, or -1.0 on error
 * 
 * @see ann_set_validation_data()
 */
ANN_API real ann_evaluate_loss(const PNetwork pnet, const PTensor inputs, const PTensor outputs);

/**
 * Compute binary confusion matrix and Matthews Correlation Coefficient.
 * 
 * For binary classification problems (2 output classes).
 * Class 0 = negative, Class 1 = positive.
 * 
 * @param pnet Trained network
 * @param inputs Input tensor [rows × input_features]
 * @param outputs Expected outputs tensor [rows × 2] (one-hot encoded)
 * @param tp Pointer to store true positives count (can be NULL)
 * @param fp Pointer to store false positives count (can be NULL)
 * @param tn Pointer to store true negatives count (can be NULL)
 * @param fn Pointer to store false negatives count (can be NULL)
 * @return Matthews Correlation Coefficient [-1, 1], or 0 on error
 */
ANN_API real ann_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs,
                          int *tp, int *fp, int *tn, int *fn);

/**
 * Print formatted binary confusion matrix with MCC.
 * 
 * Outputs a formatted confusion matrix and the Matthews Correlation
 * Coefficient to the network's output callback (or stdout).
 * 
 * @param pnet Trained network
 * @param inputs Input tensor [rows × input_features]
 * @param outputs Expected outputs tensor [rows × 2] (one-hot encoded)
 */
ANN_API void ann_print_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs);

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
ANN_API void ann_set_learning_rate(PNetwork pnet, real rate);

/**
 * Set the loss function used for training.
 * 
 * @param pnet Network to configure
 * @param loss_type LOSS_MSE (regression) or LOSS_CATEGORICAL_CROSS_ENTROPY (classification)
 *
 * LOSS_CATEGORICAL_CROSS_ENTROPY requires a softmax output (one class per
 * sample) or a sigmoid output (binary cross-entropy for each output). Training
 * any other output activation with it fails with ERR_INVALID.
 */
ANN_API void ann_set_loss_function(PNetwork pnet, Loss_type loss_type);

/**
 * Set the convergence criterion for training.
 * 
 * Training stops when average loss falls below this threshold.
 * Default: DEFAULT_CONVERGENCE (0.01)
 * 
 * @param pnet Network to configure
 * @param limit Loss threshold for convergence
 */
ANN_API void ann_set_convergence(PNetwork pnet, real limit);

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
ANN_API void ann_set_gradient_clip(PNetwork pnet, real max_grad);

/**
 * Set L2 regularization (weight decay) coefficient.
 * 
 * L2 regularization penalizes large weights by adding lambda * ||W||^2 to the loss.
 * This encourages smaller, more distributed weights and helps prevent overfitting.
 * Applied as: W = W * (1 - lr * lambda) during weight updates.
 * 
 * Default: 0 (disabled)
 * Common values: 1e-4 to 1e-2
 * 
 * @param pnet Network to configure
 * @param lambda L2 regularization strength (0 = disabled)
 */
ANN_API void ann_set_weight_decay(PNetwork pnet, real lambda);

/**
 * Set L1 regularization (LASSO) coefficient.
 * 
 * L1 regularization penalizes the absolute value of weights, encouraging sparsity.
 * Pushes small weights toward exactly zero, useful for feature selection.
 * Applied as: W = W - lr * lambda * sign(W) during weight updates.
 * 
 * Default: 0 (disabled)
 * Common values: 1e-5 to 1e-3
 * 
 * @param pnet Network to configure
 * @param lambda L1 regularization strength (0 = disabled)
 */
ANN_API void ann_set_l1_regularization(PNetwork pnet, real lambda);

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
ANN_API void ann_set_weight_init(PNetwork pnet, Weight_init_type init_type);

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
ANN_API void ann_set_batch_size(PNetwork pnet, unsigned batch_size);

/**
 * Set the maximum number of training epochs.
 * 
 * Training stops after this many epochs even if convergence threshold
 * is not reached. Default: 10000
 * 
 * @param pnet Network to configure
 * @param limit Maximum number of epochs
 */
ANN_API void ann_set_epoch_limit(PNetwork pnet, unsigned limit);

/**
 * Set a validation dataset that ann_train_network() evaluates after every epoch.
 * 
 * The validation loss is printed with each epoch, recorded in the training
 * history (see ann_export_learning_curve()), and monitored by early stopping.
 * The tensors are borrowed, not copied: keep them alive while training and
 * free them yourself. Pass NULL for both to clear the validation set.
 * 
 * @param pnet Network to configure
 * @param inputs Validation inputs (rows x input_size matrix), or NULL
 * @param outputs Validation targets (rows x output_size matrix), or NULL
 * @return ERR_OK, ERR_NULL_PTR if pnet is NULL or only one tensor is NULL,
 *         or ERR_INVALID if the row counts differ or are zero
 * 
 * @see ann_set_early_stopping(), ann_evaluate_loss()
 */
ANN_API int ann_set_validation_data(PNetwork pnet, PTensor inputs, PTensor outputs);

/**
 * Stop training when the monitored loss stops improving.
 * 
 * The validation loss is monitored when a validation set is configured,
 * otherwise the training loss. Training stops once `patience` epochs pass
 * without the loss dropping by more than `min_delta` below its best value.
 * 
 * @param pnet Network to configure
 * @param patience Epochs to wait for an improvement (0 = disabled, the default)
 * @param min_delta Minimum decrease that counts as an improvement (e.g. 0 or 1e-4)
 * @param restore_best Nonzero to restore the weights from the best epoch when
 *        training ends (whether it stopped early or reached the epoch limit)
 * 
 * @see ann_set_validation_data(), ann_get_best_epoch()
 */
ANN_API void ann_set_early_stopping(PNetwork pnet, unsigned patience, real min_delta, int restore_best);

/**
 * Get the epoch with the lowest monitored loss in the last ann_train_network() run.
 * 
 * @param pnet Trained network
 * @return 1-based epoch number, or 0 if early stopping was disabled or pnet is NULL
 */
ANN_API unsigned ann_get_best_epoch(const PNetwork pnet);

/**
 * Get the number of epochs the last ann_train_network() run trained.
 * 
 * Counts every epoch run, including ones after the best epoch when early
 * stopping restored earlier weights. Unlike ann_get_history_count(), it
 * covers only the last run and is not capped by MAX_HISTORY_SIZE.
 * 
 * @param pnet Trained network
 * @return Epochs trained, or 0 before training or if pnet is NULL
 * 
 * @see ann_get_best_epoch(), ann_get_history_count()
 */
ANN_API unsigned ann_get_epochs_trained(const PNetwork pnet);

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
 * @see ann_lr_scheduler_step, ann_lr_scheduler_exponential, ann_lr_scheduler_cosine
 */
ANN_API void ann_set_lr_scheduler(PNetwork pnet, LRSchedulerFunc scheduler, void *user_data);

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
 * Step decay scheduler: multiplies LR by gamma every step_size epochs.
 * new_lr = base_lr * (gamma ^ ((epoch - 1) / step_size)), so the first
 * step_size epochs run at base_lr. A step_size of 0 disables decay.
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRStepParams
 * @return Scheduled learning rate
 */
ANN_API real ann_lr_scheduler_step(unsigned epoch, real base_lr, void *user_data);

/**
 * Exponential decay scheduler: multiplies LR by gamma each epoch.
 * new_lr = base_lr * (gamma ^ (epoch - 1)), so epoch 1 runs at base_lr.
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRExponentialParams
 * @return Scheduled learning rate
 */
ANN_API real ann_lr_scheduler_exponential(unsigned epoch, real base_lr, void *user_data);

/**
 * Cosine annealing scheduler: smooth decay to min_lr.
 * new_lr = min_lr + (base_lr - min_lr) * (1 + cos(pi * epoch / T_max)) / 2
 * Epochs past T_max stay at min_lr. A T_max of 0 disables decay.
 * 
 * @param epoch Current epoch (1-based)
 * @param base_lr Initial learning rate
 * @param user_data Pointer to LRCosineParams
 * @return Scheduled learning rate
 */
ANN_API real ann_lr_scheduler_cosine(unsigned epoch, real base_lr, void *user_data);

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
ANN_API void ann_set_dropout(PNetwork pnet, real rate);

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
ANN_API void ann_set_layer_dropout(PNetwork pnet, int layer, real rate);

// ============================================================================
// INPUT NOISE
// ============================================================================

/**
 * Corrupt the inputs of every training batch, e.g. to train a denoising
 * autoencoder: the network sees noisy inputs and learns to output the clean
 * targets.
 * 
 * Noise is applied to a copy of each batch in ann_train_network() and
 * ann_train_step(), drawn fresh every time (so each epoch sees new noise),
 * on the CPU and with rand() like dropout, so GPU training and any thread
 * count give the same results. Predictions, evaluation and validation use
 * the inputs as given. The setting is not saved with the model.
 * 
 * @param pnet Network to configure
 * @param type NOISE_NONE, NOISE_GAUSSIAN, NOISE_MASKING or NOISE_SALT_PEPPER
 * @param amount Gaussian: standard deviation (>= 0). Masking and salt-and-pepper:
 *               probability that an input is changed, in [0, 1]
 * @param low, high When low < high, noisy values are clamped to [low, high] (e.g.
 *                  0 and 1 for pixel data); pass 0, 0 for no clamp. Salt-and-pepper
 *                  sets inputs to low or high, so it needs low < high.
 * @return ERR_OK, ERR_NULL_PTR, or ERR_INVALID for a bad type, amount or range
 */
ANN_API int ann_set_input_noise(PNetwork pnet, Noise_type type, real amount, real low, real high);

/**
 * Get the input noise settings.
 * 
 * @param pnet Network to query
 * @param type, amount, low, high Receive the settings (any may be NULL)
 * @return ERR_OK or ERR_NULL_PTR
 */
ANN_API int ann_get_input_noise(const PNetwork pnet, Noise_type *type, real *amount, real *low, real *high);

/**
 * Apply the network's input noise to values, as training does to each batch;
 * e.g. to make noisy test inputs for evaluating a denoising autoencoder.
 * Does nothing when the noise type is NOISE_NONE.
 * 
 * @param pnet Network whose noise settings to use
 * @param values Values to corrupt in place
 * @param count Number of values
 * @return ERR_OK, ERR_NULL_PTR, or ERR_INVALID for a negative count
 */
ANN_API int ann_apply_input_noise(const PNetwork pnet, real *values, int count);

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
ANN_API void ann_set_training_mode(PNetwork pnet, int is_training);

/**
 * Get the number of layers in the network.
 * 
 * @param pnet Network to query (must not be NULL)
 * @return Number of layers, or -1 if pnet is NULL
 */
ANN_API int ann_get_layer_count(const PNetwork pnet);

/**
 * Get the number of nodes in a specific layer.
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return Number of nodes in the layer, or -1 on error
 */
ANN_API int ann_get_layer_nodes(const PNetwork pnet, int layer);

/**
 * Get the activation type of a specific layer.
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return Activation type, or ACTIVATION_NULL on error
 */
ANN_API Activation_type ann_get_layer_activation(const PNetwork pnet, int layer);

/**
 * Get the kind of a specific layer (what it computes).
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return Layer kind, or LAYER_KIND_DENSE on error
 */
ANN_API Layer_kind ann_get_layer_kind(const PNetwork pnet, int layer);

/**
 * Get a layer's output shape (channels, height, width).
 * 
 * Dense and flatten layers report (nodes, 1, 1).
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @param channels, height, width Receive the shape (must not be NULL)
 * @return ERR_OK, ERR_NULL_PTR, or ERR_INVALID for a bad layer index
 */
ANN_API int ann_get_layer_shape(const PNetwork pnet, int layer, int *channels, int *height, int *width);

/**
 * Get a convolution, pooling or upsampling layer's window settings.
 * 
 * Conv2D: kernel size, stride and padding. MaxPool2D: pool size, stride and 0.
 * Upsample2D: the scale, the scale and 0. Other layers report 0 for all three.
 * For a non-square window these are the height's; see ann_get_layer_window.
 * Embedding: the vocabulary size, 0 and the padding id (-1 for none).
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @param size, stride, padding Receive the settings (must not be NULL)
 * @return ERR_OK, ERR_NULL_PTR, or ERR_INVALID for a bad layer index
 */
ANN_API int ann_get_layer_settings(const PNetwork pnet, int layer, int *size, int *stride, int *padding);

/**
 * Get a convolution, pooling or upsampling layer's window for the height and
 * the width.
 * 
 * Conv2D: kernel, stride and padding. MaxPool2D: pool size and stride, padding 0.
 * Upsample2D: the scale as kernel and stride, padding 0. Other layers report 0
 * for all six. (ann_get_layer_settings gives the height's values.)
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @param kernel_h, kernel_w Receive the window's height and width
 * @param stride_h, stride_w Receive the strides
 * @param padding_h, padding_w Receive the padding
 * @return ERR_OK, ERR_NULL_PTR or ERR_INVALID (layer out of range)
 */
ANN_API int ann_get_layer_window(const PNetwork pnet, int layer, int *kernel_h, int *kernel_w, int *stride_h,
                                 int *stride_w, int *padding_h, int *padding_w);

/**
 * Get a convolution layer's number of groups.
 * 
 * @param pnet Network to query (must not be NULL)
 * @param layer Layer index (0 = input layer)
 * @return The groups of a Conv2D layer (1 unless grouped or depthwise), 1 for
 *         other layers, or 0 if pnet is NULL or layer is out of range
 */
ANN_API int ann_get_layer_groups(const PNetwork pnet, int layer);

/**
 * Set how many CPU threads libann uses for training and batched inference.
 * 
 * libann splits its own loops (element-wise work, convolution and pooling,
 * and its built-in matrix multiply when built without BLAS) across this many
 * threads, the calling thread included. BLAS calls always stay on the calling
 * thread, so a threaded BLAS (OpenBLAS, Accelerate, MKL) keeps managing its
 * own threads and the two never compete. Results do not depend on the thread
 * count.
 * 
 * The setting is global. Call it while no training or prediction is running.
 * 
 * @param threads Thread count; 1 runs everything on the calling thread, 0
 *        restores the default (the ANN_NUM_THREADS environment variable if
 *        set, else the number of processors)
 * 
 * @see ann_get_num_threads()
 */
ANN_API void ann_set_num_threads(int threads);

/**
 * Get the number of CPU threads libann uses (1 when built with ANN_NO_THREADS).
 * 
 * @return Thread count, including the calling thread
 */
ANN_API int ann_get_num_threads(void);

//------------------------------
// What a libann build computes with; see ann_get_backend_info()
//------------------------------
typedef struct {
	const char *blas;					// BLAS library: "none" (libann's own loops), "Accelerate",
										// "OpenBLAS", "MKL", "CBLAS" or "BLAS" (another one)
	const char *gpu;					// GPU backend: the running one once ann_gpu_init() starts it, else
										// the first compiled in ("Metal", "Vulkan", "CUDA"), or "none"
	int gpu_active;						// 1 once ann_gpu_init() has started the GPU backend
	int threads;						// CPU threads for libann's own loops (ann_get_num_threads())
	int real_bytes;						// sizeof(real): 4 (float) or 8 (double)
	int optimized;						// 0 for an unoptimized (e.g. Debug, -O0) build of libann, which
										// runs much slower
} Backend_info;

/**
 * Report how this build of libann computes: its BLAS library, GPU backend,
 * CPU thread count and number type. Training prints the same as its
 * "Backend" line (see ann_print_props()).
 * 
 * @param info Receives the information (its strings are static)
 * @return ERR_OK, or ERR_NULL_PTR
 */
ANN_API int ann_get_backend_info(Backend_info *info);

/**
 * Set where a network's progress output goes (training progress, loading
 * messages, ann_print_props, ...).
 * 
 * The default writes to stdout. Errors are reported separately, through the
 * error log callback (see ann_set_error_log_callback).
 * 
 * @param pnet Network to configure
 * @param print_func Function receiving each piece of text, or NULL to silence the network
 * 
 * @see ann_get_print_function()
 */
ANN_API void ann_set_print_function(PNetwork pnet, Output_func print_func);

/**
 * Get a network's progress output function (NULL when it is silenced).
 * 
 * Save it before ann_set_print_function(pnet, NULL) to restore output later.
 * 
 * @param pnet Network to query
 * @return The print function, or NULL
 */
ANN_API Output_func ann_get_print_function(const PNetwork pnet);

/**
 * Print network properties and configuration to stdout.
 * 
 * Displays: network topology, optimizer, loss function, mini-batch size, etc.
 * 
 * @param pnet Network to describe (must not be NULL)
 */
ANN_API void ann_print_props(const PNetwork pnet);

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
ANN_API void ann_print_outputs(const PNetwork pnet);

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
ANN_API const char* ann_strerror(int error_code);

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
 * With no callback installed (the default), errors are printed to stderr.
 * Pass NULL to restore that default; install a callback that does nothing
 * to silence errors.
 * 
 * @param callback Function pointer to error handler, or NULL for the default
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
ANN_API void ann_set_error_log_callback(ErrorLogCallback callback);

/**
 * Get the currently installed error logging callback.
 * 
 * @return Current callback function pointer, or NULL if none is set
 */
ANN_API ErrorLogCallback ann_get_error_log_callback(void);

/**
 * Remove the error logging callback.
 * 
 * Equivalent to calling ann_set_error_log_callback(NULL).
 * After this call, library errors are printed to stderr again.
 * 
 * @see ann_set_error_log_callback
 */
ANN_API void ann_clear_error_log_callback(void);

#endif
