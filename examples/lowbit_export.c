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
// Low-bit weights for small exported models (quantization-aware fine-tuning).
//
// Trains a float MLP (784 -> 128 relu -> 10 softmax) on Fashion-MNIST, then
// for each low-bit format (int4, ternary, binary) gives a copy of it that
// format on the hidden layer, fine-tunes it with the online API, which keeps
// the weights, and quantizes it. Prints test accuracy and parameter memory
// for each, and writes the binary model as fashion_lowbit_q.h. The output
// layer stays int8: it is small, and the most sensitive to rounding.
//
// Usage: lowbit_export [-e float_epochs] [-f finetune_epochs] [train.csv] [test.csv]
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"

#define CALIBRATION_ROWS 1000
#define BATCH 32

static int epoch_count = 5;
static int finetune_epochs = 3;

static const char *option_value(int n, char *args[], int *i)
{
	if (args[*i][2])
		return &args[*i][2];
	if (*i + 1 < n)
		return args[++*i];
	return NULL;
}

static int get_options(int n, char *args[])
{
	int i;
	for (i = 1; (i < n) && (args[i][0] == '-'); i++)
	{
		const char *value;
		if (args[i][1] == 'e' && (value = option_value(n, args, &i)))
			epoch_count = atoi(value);
		else if (args[i][1] == 'f' && (value = option_value(n, args, &i)))
			finetune_epochs = atoi(value);
	}
	return i;
}

// load a label-first CSV into inputs scaled to [0, 1] and one-hot labels
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
	ann_free_csv(data);

	*x = tensor_slice_cols(labels, 1);	// pixels; labels keeps the class column
	*y = tensor_onehot(labels, 10);
	tensor_mul_scalar(*x, (real)(1.0 / 255.0));
	tensor_free(labels);
	return ERR_OK;
}

static PNetwork make_network(void)
{
	PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
	ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_RELU);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
	ann_set_epoch_limit(pnet, epoch_count);
	ann_set_convergence(pnet, (real)0.0);
	return pnet;
}

// a new network with src's parameters
static PNetwork copy_network(PNetwork src)
{
	PNetwork dst = make_network();
	for (int layer = 1; layer < ann_get_layer_count(src); layer++)
		for (int p = 0; p < ann_get_layer_param_count(src, layer); p++)
		{
			int rows, cols;
			ann_get_layer_param_shape(src, layer, p, &rows, &cols);
			real *values = malloc((size_t)rows * cols * sizeof(real));
			ann_get_layer_param(src, layer, p, values, rows * cols);
			ann_set_layer_param(dst, layer, p, values, rows * cols);
			free(values);
		}
	return dst;
}

// fine-tune with the online API, which keeps the weights: shuffled batches,
// a lower learning rate than training from scratch
static void finetune(PNetwork pnet, PTensor x, PTensor y)
{
	int rows = x->rows, in = x->cols, out = y->cols;
	int *order = malloc(rows * sizeof(int));
	real *bx = malloc((size_t)BATCH * in * sizeof(real)), *by = malloc((size_t)BATCH * out * sizeof(real));

	for (int i = 0; i < rows; i++)
		order[i] = i;
	ann_set_learning_rate(pnet, (real)0.0005);
	ann_train_begin(pnet);
	for (int epoch = 0; epoch < finetune_epochs; epoch++)
	{
		for (int i = rows - 1; i > 0; i--)
		{
			int j = rand() % (i + 1), t = order[i];
			order[i] = order[j];
			order[j] = t;
		}
		for (int b = 0; b + BATCH <= rows; b += BATCH)
		{
			for (int k = 0; k < BATCH; k++)
			{
				memcpy(&bx[k * in], &x->values[(size_t)order[b + k] * in], in * sizeof(real));
				memcpy(&by[k * out], &y->values[(size_t)order[b + k] * out], out * sizeof(real));
			}
			ann_train_step(pnet, bx, by, BATCH);
		}
	}
	ann_train_end(pnet);
	free(order);
	free(bx);
	free(by);
}

int main(int argc, char *argv[])
{
	static const struct { Weight_format format; const char *name; } formats[] = {
		{ WEIGHT_FORMAT_FLOAT, "int8" },
		{ WEIGHT_FORMAT_INT4, "int4" },
		{ WEIGHT_FORMAT_TERNARY, "ternary" },
		{ WEIGHT_FORMAT_BINARY, "binary" },
	};
	int first_arg = get_options(argc, argv);
	const char *train_file = first_arg < argc ? argv[first_arg] : "fashion-mnist_train.csv";
	const char *test_file = first_arg + 1 < argc ? argv[first_arg + 1] : "fashion-mnist_test.csv";
	PTensor x_train, y_train, x_test, y_test;

	if (load_images(train_file, &x_train, &y_train) != ERR_OK ||
		load_images(test_file, &x_test, &y_test) != ERR_OK)
		return ERR_FAIL;

	ann_set_seed(1);
	PNetwork base = make_network();
	ann_train_network(base, x_train, y_train, x_train->rows);
	real acc_float = ann_evaluate_accuracy(base, x_test, y_test);

	int cal_rows = x_train->rows < CALIBRATION_ROWS ? x_train->rows : CALIBRATION_ROWS;
	PTensor calibration = tensor_create_from_array(cal_rows, x_train->cols, x_train->values);

	printf("\nTest accuracy, float32: %.2f%% (%d parameters, %ld KB as float32)\n\n", acc_float * 100,
	       784 * 128 + 128 + 128 * 10 + 10, (long)((784 * 128 + 128 + 128 * 10 + 10) * 4 / 1024));
	printf("hidden layer   rounded   fine-tuned   int8 model   parameters\n");

	for (int f = 0; f < 4; f++)
	{
		PNetwork pnet = copy_network(base);
		real acc_rounded = acc_float, acc_tuned = acc_float;

		if (formats[f].format != WEIGHT_FORMAT_FLOAT)
		{
			ann_set_weight_format(pnet, formats[f].format);		// every layer but the output
			acc_rounded = ann_evaluate_accuracy(pnet, x_test, y_test);
			finetune(pnet, x_train, y_train);
			acc_tuned = ann_evaluate_accuracy(pnet, x_test, y_test);
		}

		PQuantizedNetwork q = ann_quantize(pnet, calibration);
		if (!q)
			return ERR_FAIL;
		real acc_q = ann_quantized_evaluate_accuracy(q, x_test, y_test);

		if (formats[f].format == WEIGHT_FORMAT_FLOAT)
			printf("%-12s   %6s    %9s    %8.2f%%   %7.1f KB\n", formats[f].name, "-", "-", acc_q * 100,
			       ann_quantized_param_bytes(q) / 1024.0);
		else
			printf("%-12s   %6.2f%%   %8.2f%%    %8.2f%%   %7.1f KB\n", formats[f].name, acc_rounded * 100,
			       acc_tuned * 100, acc_q * 100, ann_quantized_param_bytes(q) / 1024.0);

		if (formats[f].format == WEIGHT_FORMAT_BINARY)
		{
			if (ann_export_c_header(NULL, q, "fashion_lowbit_q.h", "fashion_lowbit_q") != ERR_OK)
				return ERR_FAIL;
			printf("\nWrote fashion_lowbit_q.h (the binary model); copy ann_rt.h next to it to build.\n");
		}
		ann_free_quantized(q);
		ann_free_network(pnet);
	}

	ann_free_network(base);
	tensor_free(calibration);
	tensor_free(x_train);
	tensor_free(y_train);
	tensor_free(x_test);
	tensor_free(y_test);
	return 0;
}
