#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

#ifdef _WIN32
#	define DIR_FIX "..\\"
#else
#	define DIR_FIX
#endif

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
		ann_load_csv(DIR_FIX "pima-indians-diabetes.csv", CSV_NO_HEADER, &data, &rows, &stride);

	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	PTensor x_test = tensor_slice_rows(x_train, 576);

	PTensor y_train = tensor_slice_cols(x_train, 8);
	PTensor y_test = tensor_slice_cols(x_test, 8);

	// normalize inputs
	PTensor t_max = tensor_max(x_train);
	tensor_div(x_train, t_max);
	tensor_div(x_test, t_max);

	// define our network
	ann_add_layer(pnet, 8, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 8, LAYER_OUTPUT, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 4, LAYER_OUTPUT, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	pnet->epochLimit = 10;

	ann_train_network(pnet, x_train, y_train, x_train->rows);

	// evaluate the network against the training data
	real acc = ann_evaluate_accuracy(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

	real outputs[1];
	ann_predict(pnet, x_test->values, outputs);

	print_outputs(pnet);

	ann_free_network(pnet);

	tensor_free(x_train);
	tensor_free(y_train);
	tensor_free(x_test);
	tensor_free(y_test);

	free(data);
	return 0;
}
