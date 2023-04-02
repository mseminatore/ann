#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

// AND function
real inputs[] = {
	0.0, 0.0, 0.0,
	0.0, 1.0, 0.0,
	1.0, 0.0, 0.0,
	1.0, 1.0, 1.0
};

// OR function
// real inputs[] = {
// 	0.0, 0.0, 0.0,
// 	0.0, 1.0, 1.0,
// 	1.0, 0.0, 1.0,
// 	1.0, 1.0, 1.0
// };

// XOR function
//real inputs[] = {
//	0.0, 0.0, 0.0,
//	0.0, 1.0, 1.0,
//	1.0, 0.0, 1.0,
//	1.0, 1.0, 0,0
//};

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

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	PNetwork pnet = ann_make_network();

	real *data;
	size_t rows, stride;

	// load the data
	ann_load_csv("or.csv", &data, &rows, &stride);

//	print_data(data, rows, stride);

	// define our network
	ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

//	ann_set_learning_rate(pnet, 0.2);

	ann_train_network(pnet, inputs, rows, stride);
	
//	ann_test_network(pnet, inputs, outputs);

	ann_free_network(pnet);

	free(data);
	return 0;
}
