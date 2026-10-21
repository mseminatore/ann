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

//------------------------------------------------------------------------------
// Fashion-MNIST with a small convolutional network.
//
// Same data as examples/mnist.c, but each 784-pixel row is treated as a
// 1x28x28 image:
//
//   input 1x28x28 -> conv 8@3x3 relu -> maxpool 2 -> conv 16@3x3 relu
//   -> maxpool 2 -> flatten -> dense 64 relu -> dense 10 softmax
//
// Usage: fashion_cnn [-e epochs] [-b batch] [-g] [train.csv] [test.csv]
// -g trains on the GPU (needs a USE_METAL build). On the CPU training takes
// roughly 10 seconds per epoch; 10 epochs reach
// about 90% test accuracy (the MLP in examples/mnist.c gets about 88%).
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

static int batch_size = 32;
static int epoch_count = 3;
static int use_gpu = 0;     // train on the GPU (-g, needs a USE_METAL build)

//----------------------------------
// get options from the command line
//----------------------------------
static int get_options(int n, char *args[])
{
	int i;
	for (i = 1; (i < n) && (args[i][0] == '-'); i++)
	{
		if (args[i][1] == 'b' && i + 1 < n)
			batch_size = atoi(args[++i]);
		else if (args[i][1] == 'e' && i + 1 < n)
			epoch_count = atoi(args[++i]);
		else if (args[i][1] == 'g')
			use_gpu = 1;
	}
	return i;
}

//----------------------------------
// load a label-first CSV into image inputs scaled to [0, 1] and one-hot labels
//----------------------------------
static int load_images(const char *filename, PTensor *x, PTensor *y)
{
	real *data = NULL;
	int rows, stride;

	if (ERR_OK != ann_load_csv(filename, CSV_HAS_HEADER, &data, &rows, &stride))
	{
		printf("Error: unable to open data file - %s\n", filename);
		return ERR_FAIL;
	}

	PTensor labels = tensor_create_from_array(rows, stride, data);
	free(data);

	*x = tensor_slice_cols(labels, 1);	// pixels; labels keeps the class column
	*y = tensor_onehot(labels, 10);
	tensor_mul_scalar(*x, (real)(1.0 / 255.0));
	tensor_free(labels);
	return ERR_OK;
}

int main(int argc, char *argv[])
{
	int first_arg = get_options(argc, argv);
	const char *train_file = first_arg < argc ? argv[first_arg] : "fashion-mnist_train.csv";
	const char *test_file = first_arg + 1 < argc ? argv[first_arg + 1] : "fashion-mnist_test.csv";

#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
#endif

	PTensor x_train, y_train, x_test, y_test;
	if (load_images(train_file, &x_train, &y_train) != ERR_OK ||
		load_images(test_file, &x_test, &y_test) != ERR_OK)
		return ERR_FAIL;

	// define the network
	PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);

	ann_add_input(pnet, 1, 28, 28);
	ann_add_conv2d(pnet, 8, 3, 1, 0, ACTIVATION_RELU);		// 8x26x26
	ann_add_maxpool2d(pnet, 2, 2);							// 8x13x13
	ann_add_conv2d(pnet, 16, 3, 1, 0, ACTIVATION_RELU);		// 16x11x11
	ann_add_maxpool2d(pnet, 2, 2);							// 16x5x5
	ann_add_flatten(pnet);									// 400
	ann_add_layer(pnet, 64, LAYER_HIDDEN, ACTIVATION_RELU);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

	// set some hyper-parameters
	ann_set_epoch_limit(pnet, epoch_count);
	ann_set_convergence(pnet, (real)0.0);	// always train every epoch
	ann_set_batch_size(pnet, batch_size);

	static real lr_decay = 0.9f;
	ann_set_lr_scheduler(pnet, ann_lr_scheduler_exponential, &lr_decay);

#ifdef USE_GPU
	// GPU is opt-in at runtime: initialize it, then upload the network before training
	if (use_gpu && (!ann_gpu_init() || ERR_OK != ann_gpu_upload_network(pnet)))
		puts("GPU unavailable, training on CPU");
#else
	if (use_gpu)
		puts("GPU support not compiled in, using CPU");
#endif

	// train the network
	ann_train_network(pnet, x_train, y_train, x_train->rows);

	// evaluate the network against the test data
	real acc = ann_evaluate_accuracy(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

	ann_export_onnx(pnet, "fashion_cnn.onnx.json");
	ann_export_pikchr(pnet, "fashion_cnn.pikchr");

	// free memory
	ann_free_network(pnet);
	tensor_free(x_train);
	tensor_free(y_train);
	tensor_free(x_test);
	tensor_free(y_test);

	return 0;
}
