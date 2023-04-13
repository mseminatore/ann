#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	real *data;
	size_t rows, stride;

	// load the data
	if (argc > 1)
		ann_load_csv(argv[1], CSV_NO_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("and.csv", CSV_NO_HEADER, &data, &rows, &stride);

	PNetwork pnet = ann_make_network();

	// define our network
	ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	ann_train_network(pnet, data, rows, stride);
	
	real outputs[1];
	ann_predict(pnet, &data[0], outputs);

	print_outputs(pnet);

	ann_free_network(pnet);

	free(data);
	return 0;
}
