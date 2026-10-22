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
// Quantize a Fashion-MNIST model to int8 and export it as C headers.
//
// Trains an MLP (784 -> 128 relu -> 10 softmax) or, with -c, the CNN from
// examples/fashion_cnn.c, or loads a saved model with -m. Then quantizes it,
// calibrating on 1000 training images, compares float and int8 test accuracy,
// and writes fashion_q.h (int8) and fashion_f.h (float32). Copy ann_rt.h next
// to them to build them into a firmware or any C program.
//
// Usage: export_model [-c] [-e epochs] [-m model.nna] [train.csv] [test.csv]
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"

#define CALIBRATION_ROWS 1000

static int use_cnn = 0;
static int epoch_count = 5;
static const char *model_file = NULL;

//----------------------------------
// an option's value: attached (-e2) or the next argument (-e 2)
//----------------------------------
static const char *option_value(int n, char *args[], int *i)
{
	if (args[*i][2])
		return &args[*i][2];
	if (*i + 1 < n)
		return args[++*i];
	return NULL;
}

//----------------------------------
// get options from the command line
//----------------------------------
static int get_options(int n, char *args[])
{
	int i;
	for (i = 1; (i < n) && (args[i][0] == '-'); i++)
	{
		const char *value;
		if (args[i][1] == 'c')
			use_cnn = 1;
		else if (args[i][1] == 'e' && (value = option_value(n, args, &i)))
			epoch_count = atoi(value);
		else if (args[i][1] == 'm' && (value = option_value(n, args, &i)))
			model_file = value;
	}
	return i;
}

//----------------------------------
// load a label-first CSV into inputs scaled to [0, 1] and one-hot labels
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

static PNetwork make_network(void)
{
	PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);

	if (use_cnn)
	{
		ann_add_input(pnet, 1, 28, 28);
		ann_add_conv2d(pnet, 8, 3, 1, 0, ACTIVATION_RELU);
		ann_add_maxpool2d(pnet, 2, 2);
		ann_add_conv2d(pnet, 16, 3, 1, 0, ACTIVATION_RELU);
		ann_add_maxpool2d(pnet, 2, 2);
		ann_add_flatten(pnet);
		ann_add_layer(pnet, 64, LAYER_HIDDEN, ACTIVATION_RELU);
	}
	else
	{
		ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
		ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_RELU);
	}
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

	ann_set_epoch_limit(pnet, epoch_count);
	ann_set_convergence(pnet, (real)0.0);
	return pnet;
}

static long file_size(const char *filename)
{
	FILE *f = fopen(filename, "rb");
	long size = -1;
	if (f && fseek(f, 0, SEEK_END) == 0)
		size = ftell(f);
	if (f)
		fclose(f);
	return size;
}

int main(int argc, char *argv[])
{
	int first_arg = get_options(argc, argv);
	const char *train_file = first_arg < argc ? argv[first_arg] : "fashion-mnist_train.csv";
	const char *test_file = first_arg + 1 < argc ? argv[first_arg + 1] : "fashion-mnist_test.csv";
	PTensor x_train, y_train, x_test, y_test;
	PNetwork pnet;

	if (load_images(train_file, &x_train, &y_train) != ERR_OK ||
		load_images(test_file, &x_test, &y_test) != ERR_OK)
		return ERR_FAIL;

	if (model_file)
	{
		pnet = ann_load_network(model_file);
		if (!pnet)
		{
			printf("Error: unable to load model - %s\n", model_file);
			return ERR_FAIL;
		}
	}
	else
	{
		pnet = make_network();
		ann_train_network(pnet, x_train, y_train, x_train->rows);
	}

	// calibrate on the first training images
	int rows = x_train->rows < CALIBRATION_ROWS ? x_train->rows : CALIBRATION_ROWS;
	PTensor calibration = tensor_create_from_array(rows, x_train->cols, x_train->values);
	PQuantizedNetwork q = ann_quantize(pnet, calibration);
	if (!q)
		return ERR_FAIL;

	real acc_float = ann_evaluate_accuracy(pnet, x_test, y_test);
	real acc_int8 = ann_quantized_evaluate_accuracy(q, x_test, y_test);
	printf("\nTest accuracy: float32 %.2f%%, int8 %.2f%% (%+.2f points)\n",
	       acc_float * 100, acc_int8 * 100, (acc_int8 - acc_float) * 100);

	if (ann_export_c_header(NULL, q, "fashion_q.h", "fashion_q") != ERR_OK ||
		ann_export_c_header(pnet, NULL, "fashion_f.h", "fashion_f") != ERR_OK)
		return ERR_FAIL;
	printf("Wrote fashion_q.h (%ld bytes of source) and fashion_f.h (%ld bytes of source)\n",
	       file_size("fashion_q.h"), file_size("fashion_f.h"));
	printf("Each header's comment gives its parameter memory; copy ann_rt.h next to them to build.\n");

	ann_free_quantized(q);
	ann_free_network(pnet);
	tensor_free(calibration);
	tensor_free(x_train);
	tensor_free(y_train);
	tensor_free(x_test);
	tensor_free(y_test);
	return 0;
}
