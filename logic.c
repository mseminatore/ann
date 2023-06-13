#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	real *data;
	int rows, stride;

	// load the data
	if (argc > 1)
		ann_load_csv(argv[1], CSV_NO_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("and.csv", CSV_NO_HEADER, &data, &rows, &stride);

	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	PTensor y_train = tensor_slice_cols(x_train, 2);

	// define our network
	ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	pnet->batchSize = 1;
	
	ann_train_network(pnet, x_train, y_train, x_train->rows);
	
	real outputs[1];
	ann_predict(pnet, &data[0], outputs);

	print_outputs(pnet);

	ann_free_network(pnet);

	free(data);
	return 0;
}
