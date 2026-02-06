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
#include "ann.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

#ifdef _WIN32
#	define DIR_FIX "..\\"
#else
#	define DIR_FIX
#endif

//---------------------------------------
// print the Ci with highest probability
//---------------------------------------
void print_class_pred(real * data)
{
	int class = ann_class_prediction(data, 10);
	printf("\n5x7 character prediction is: %d\n\n", class + 1);
}

//------------------------------
// display a 5x7 char from flat data
//------------------------------
void print5x7(real *data)
{
	char c;

	puts("\nInput vector\n");

	for (int row = 0 ; row < 7; row++)
	{
		for (int col = 0; col < 5; col++)
		{
			c = data[row * 5 + col] == 1.0 ? '+' : ' ';
			putchar(c);
		}
		puts("");
	}
}

//------------------------------
// add noise to image
//------------------------------
void add_noise(real *data, int size, int amount)
{
	for (int i = 0; i < amount; i++)
	{
		int index = rand() % size;

		data[index] = (data[index] == 1.0) ? (real)0.0 : (real)1.0;
	}
}

//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

	char *filename = DIR_FIX "num5x7.csv";

	PNetwork pnet = ann_make_network(OPT_DEFAULT, LOSS_DEFAULT);
	if (!pnet)
		return ERR_FAIL;

	real *data;
	int rows, stride;

	// load the data
	if (argc > 1)
		filename = argv[1];
		
	printf("Loading %s...", filename);
	CHECK_OK(ann_load_csv(filename, CSV_NO_HEADER, &data, &rows, &stride));
	puts("done.");

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	if (!x_train)
		return ERR_FAIL;

	PTensor y_train = tensor_slice_cols(x_train, 35);
	if (!y_train)
		return ERR_FAIL;

	pnet->learning_rate = (real)0.35;
	
	// define our network
	CHECK_OK(ann_add_layer(pnet, 35, LAYER_INPUT, ACTIVATION_NULL));
	CHECK_OK(ann_add_layer(pnet, 48, LAYER_HIDDEN, ACTIVATION_SIGMOID));
	CHECK_OK(ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID));

	ann_train_network(pnet, x_train, y_train, x_train->rows);

	int correct = 0;

	for (int i = 0; i < 10; i++)
	{
		add_noise(&data[i * 45], 35, 3);

		real outputs[10];
		CHECK_OK(ann_predict(pnet, &data[i * 45], outputs));

		print5x7(&data[i * 45]);
		print_class_pred(outputs);
		int pred_class = ann_class_prediction(outputs, 10);
		int class = ann_class_prediction(&y_train->values[i * 10], 10);

		if (pred_class == class)
			correct++;

		printf("Actual class: %d\n", class + 1);
	}

	real acc = ann_evaluate_accuracy(pnet, x_train, y_train);
	printf("Train accuracy: %g%%\n", acc * 100);
	printf("Test accuracy: %g%%\n", (real)correct * 10);

	ann_free_network(pnet);

	tensor_free(x_train);
	tensor_free(y_train);

	free(data);
	return ERR_OK;
}
