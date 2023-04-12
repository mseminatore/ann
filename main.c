#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//
void print_data(real *data, size_t rows, size_t stride)
{
	for (size_t row = 0; row < rows; row++)
	{
		for (size_t col = 0; col < stride; col++)
		{
			printf("%g, ", *data++);
		}

		puts("");
	}
}

#define ACTIVATION ACTIVATION_SIGMOID
//#define ACTIVATION ACTIVATION_RELU

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	PNetwork pnet = ann_make_network();

	real *data;
	size_t rows, stride;

	// load the data
	if (argc > 1)
		ann_load_csv(argv[1], &data, &rows, &stride);
	else
		ann_load_csv("num5x7.csv", &data, &rows, &stride);

	// print_data(data, rows, stride);

	PTensor t = tensor_create_from_array(rows, stride, data);
	tensor_print(t);

	PTensor o = tensor_slice_cols(t, 2);
	tensor_print(t);
	tensor_print(o);

	tensor_free(t);
	tensor_free(o);

	// define our network
//	pnet->loss_type = LOSS_CROSS_ENTROPY;

	ann_add_layer(pnet, 35, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 48, LAYER_HIDDEN, ACTIVATION);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION);

	ann_train_network(pnet, data, rows, stride);
	
//	ann_test_network(pnet, inputs, outputs);

//	softmax(pnet);
	print_outputs(pnet);

	ann_free_network(pnet);

	free(data);
	return 0;
}
