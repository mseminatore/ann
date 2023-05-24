#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ann.h"
#include <cblas.h>

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

//------------------------------
//
//------------------------------
void print_class_pred(real * data)
{
	int class = ann_class_prediction(data, 10);

	printf("\nClass prediction is: %d\n\n", class + 1);
}

//------------------------------
// display a 28x28 image from flat data
//------------------------------
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

//
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
//
//------------------------------
void hypertune(PNetwork pnet, PTensor x_train, PTensor y_train, int rows)
{
	real loss, minLoss;
	real opt_lr, opt_weights;

	pnet->epochLimit = 1;

	// optimize for learning rate
	puts("Optimizing LR\n");

	minLoss = 100.0;
	real dl_dr = (real)0.1;
	pnet->learning_rate = (real)0.01;
	loss = ann_train_network(pnet, x_train, y_train, rows);

	while (dl_dr > 1e-3)
	{
		loss -= ann_train_network(pnet, x_train, y_train, rows);
//		dl_dr = loss / 

		pnet->learning_rate -= dl_dr;
	}

	for (real lr = (real)0.001; lr < 1.1; lr += 0.0)
	{
		pnet->learning_rate = lr;

		// train the network
		loss = ann_train_network(pnet, x_train, y_train, rows);
		if (loss < minLoss)
		{
			minLoss = loss;
			opt_lr = lr;
		}
	}

//	printf("Optimal LR was: %f\n", opt_lr);

	// optimize for initial weights
	puts("Optimizing weights\n");

	minLoss = 100.0;
	for (real w = (real)0.001; w < 1.1; w *= 10.0)
	{
		pnet->weight_limit = w;
		pnet->learning_rate = opt_lr;

		// train the network
		loss = ann_train_network(pnet, x_train, y_train, rows);
		if (loss < minLoss)
		{
			minLoss = loss;
			opt_weights = w;
		}
	}

	printf("Optimal LR was: %f\n", opt_lr);
	printf("Optimal weight was: %f\n", opt_weights);

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

	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_CATEGORICAL_CROSS_ENTROPY);

#ifdef USE_BLAS
	printf( "%s\n", openblas_get_config());
	printf("      CPU uArch: %s\n", openblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", openblas_get_num_procs(), openblas_get_num_threads());
#endif

	// define our network
	ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

	real *data, *test_data;
	int rows, stride, test_rows, test_stride;

	// load the training data
	if (argc > 1)
		ann_load_csv(argv[1], CSV_HAS_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("fashion-mnist_train.csv", CSV_HAS_HEADER, &data, &rows, &stride);

	// load the test data
	if (argc > 2)
		ann_load_csv(argv[2], CSV_HAS_HEADER, &test_data, &test_rows, &test_stride);
	else
		ann_load_csv("fashion-mnist_test.csv", CSV_HAS_HEADER, &test_data, &test_rows, &test_stride);

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

	pnet->epochLimit = 5;
	pnet->convergence_epsilon = (real)1e-5;
	pnet->batchSize = 8;

	// train the network
	ann_train_network(pnet, x_train, y_train, x_train->rows /20);
	
	// evaluate the network against the test data
	real acc = ann_evaluate(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

//	ann_save_network(pnet, "mnist-fashion.nna");
//	ann_save_network_binary(pnet, "mnist-fashion.bnn");

	int i = 0;
//	for (; i < 5; i++)
//		print_ascii_art(&x_train->values[i * 784], 28, 28);

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
