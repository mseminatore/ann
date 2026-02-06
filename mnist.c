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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ann.h"

#ifdef USE_BLAS
#	include <cblas.h>
#endif

#define EPSILON 1e-5

static int threads = -1;
static int batch_size = 8;
static int epoch_count = 5;
static int export_onnx = 0;

//----------------------------------
// get options from the command line
//----------------------------------
int getopt(int n, char *args[])
{
	int i;
	for (i = 1; (i < n) && (args[i][0] == '-'); i++)
	{
		// thread count
		if (args[i][1] == 't')
		{
			threads = atoi(args[i + 1]);
			i++;
		}

		// batch size
		if (args[i][1] == 'b')
		{
			batch_size = atoi(args[i + 1]);
			i++;
		}

		// epochs
		if (args[i][1] == 'e')
		{
			epoch_count = atoi(args[i + 1]);
			i++;
		}

		// export ONNX file
		if (args[i][1] == 'x')
		{
			export_onnx = 1;
			i++;
		}
	}

	return i;
}

//------------------------------
//
//------------------------------
void print_data(real *data, int rows, int stride)
{
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < stride; col++)
		{
			printf("%g, ", *data++);
		}

		puts("");
	}
}

//------------------------------------------
// get/print prediction from one-hot vector
//------------------------------------------
void print_class_prediction(real * data)
{
	int class = ann_class_prediction(data, 10);

	printf("\nClass prediction is: %d\n\n", class + 1);
}

//--------------------------------------
// display a 28x28 image from flat data
//--------------------------------------
void print_ascii_art(real *data, int rows, int cols)
{
	char c;
	char *pixels = " `. - ':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

	puts("\nInput image\n");

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col< cols; col++)
		{
			c = (int)(94.0  * data[row * cols + col]);
			putchar(pixels[c]);
		}
		puts("");
	}
}

//--------------------------------------
// print a histogram of tensor values
//--------------------------------------
void class_histogram(PTensor outputs)
{
	int pred;

	int classes = outputs->cols;
	int *histo = alloca(classes * sizeof(int));
	int sum = 0;

	memset(histo, 0, classes * sizeof(int));

	printf("\nClass Histogram\n");

	for (int row = 0; row < outputs->rows; row++)
	{
		pred = ann_class_prediction(&outputs->values[row * classes], classes);
		histo[pred]++;
	}

	for (int i = 0; i < classes; i++)
		sum += histo[i];

	for (int i = 0; i < classes; i++)
		histo[i] = 40 * histo[i] / sum;

	for (int i = 0; i < classes; i++)
	{
		printf("%3d|", i);
		for (int j = 0; j < histo[i]; j++)
			putchar('*');
		puts("");
	}

	printf("   +");
	for (int i = 0; i < 40; i++)
		putchar('-');
	puts("");
}

//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
	char *classes[] =
	{
		"T - shirt / top",
		"Trouser",
		"Pullover",
		"Dress",
		"Coat",
		"Sandal",
		"Shirt",
		"Sneaker",
		"Bag",
		"Ankle boot"
	};

	int iFirstArg = getopt(argc, argv);

#if defined(USE_BLAS) || defined(USE_CBLAS) || defined(CBLAS)

	#if defined(CBLAS) || defined(USE_CBLAS)
		cblas_init(CBLAS_DEFAULT_THREADS);
		if (threads != -1)
			cblas_set_num_threads(threads);
			
		printf( "%s\n", cblas_get_config());
		printf("      CPU uArch: %s\n", cblas_get_corename());
		printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
	#else
		if (threads != -1)
			openblas_set_num_threads(threads);

		printf( "%s\n", openblas_get_config());
		printf("      CPU uArch: %s\n", openblas_get_corename());
		printf("  Cores/Threads: %d/%d\n", openblas_get_num_procs(), openblas_get_num_threads());
	#endif
#else
	printf("\nBLAS not enabled, using scalar single-threaded C implementation\n");
#endif

	// make a new network
	PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);

	// define our network
	ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 32, LAYER_HIDDEN, ACTIVATION_SIGMOID);
//	ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_RELU);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

	real *data = NULL, *test_data = NULL;
	int rows, stride, test_rows, test_stride;
	char *training_data_file = "fashion-mnist_train.csv";
	char* testing_data_file = "fashion-mnist_test.csv";

	// load the training data
	if (iFirstArg < argc)
		training_data_file = argv[iFirstArg];
	
	if (ERR_OK != ann_load_csv(training_data_file, CSV_HAS_HEADER, &data, &rows, &stride))
	{
		printf("Error: unable to open training file - %s\n", training_data_file);
		return ERR_FAIL;
	}

	// load the test data
	if (iFirstArg + 1 < argc)
		testing_data_file = argv[iFirstArg + 1];

	if (ERR_OK != ann_load_csv(testing_data_file, CSV_HAS_HEADER, &test_data, &test_rows, &test_stride))
	{
		printf("Error: unable to open training file - %s\n", training_data_file);
		return ERR_FAIL;
	}

	// convert outputs to onehot code
	PTensor y_labels = tensor_create_from_array(rows, stride, data);
	free(data);

	PTensor x_train = tensor_slice_cols(y_labels, 1);
	PTensor y_train = tensor_onehot(y_labels, 10);

	PTensor y_test_labels = tensor_create_from_array(test_rows, test_stride, test_data);
	free(test_data);

	PTensor x_test = tensor_slice_cols(y_test_labels, 1);
	PTensor y_test = tensor_onehot(y_test_labels, 10);

	// normalize inputs
	tensor_mul_scalar(x_train, (real)(1.0 / 255.0));
	tensor_mul_scalar(x_test, (real)(1.0 / 255.0));

	// set some hyper-parameters
	pnet->epochLimit = epoch_count;
	pnet->convergence_epsilon = (real)EPSILON;
	pnet->batchSize = batch_size;

	// train the network
	ann_train_network(pnet, x_train, y_train, x_train->rows);
	
	// if export ONNX flag is set, export the trained network to an ONNX file
	if (export_onnx)
	{
		char *onnx_filename = "fashion_mnist.onnx";
		if (ERR_OK != ann_export_onnx(pnet, onnx_filename))
		{
			printf("Error: failed to export ONNX file - %s\n", onnx_filename);
			return ERR_FAIL;
		}
		else
		{
			printf("Exported trained network to ONNX file: %s\n", onnx_filename);
		}
	}

	// evaluate the network against the test data
	real acc = ann_evaluate_accuracy(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

	// free memory
	ann_free_network(pnet);

	tensor_free(y_labels);
	tensor_free(x_train);
	tensor_free(y_train);

	tensor_free(y_test_labels);
	tensor_free(x_test);
	tensor_free(y_test);

	return 0;
}
