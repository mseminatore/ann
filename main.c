#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//------------------------------
//
//------------------------------
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
void print_class_pred(real * data)
{
	int num = -1;
	real prob = 0.0;

	for (int i = 0; i < 10; i++)
	{
		if (data[i] > prob)
		{
			prob = data[i];
			num = i;
		}
	}

	printf("\nClass prediction is: %d\n\n", num + 1);
}

//------------------------------
//
//------------------------------
void test_tensor()
{

}

//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
	PNetwork pnet = ann_make_network();

	real *data;
	size_t rows, stride;

	// load the data
	if (argc > 1)
		ann_load_csv(argv[1], CSV_HAS_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("fashion-mnist_train.csv", CSV_HAS_HEADER, &data, &rows, &stride);

	// print_data(data, rows, stride);

	PTensor y_labels = tensor_create_from_array(rows, stride, data);
	//tensor_print(t);

	PTensor x_train = tensor_slice_cols(y_labels, 1);
	PTensor x_test = tensor_slice_rows(x_train, 50000);

	// convert outputs to onehot code
	PTensor y_train = tensor_onehot(y_labels, 10);
	PTensor y_test = tensor_slice_rows(y_train, 50000);

	// normalize inputs
	tensor_mul_scalar(x_train, (real)(1.0 / 255.0));
	tensor_mul_scalar(x_test, (real)(1.0 / 255.0));

	//tensor_print(t);
	//tensor_print(o);

	//tensor_free(t);
	//tensor_free(o);

	// define our network
	ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	pnet->epochLimit = 5;

	ann_train_network(pnet, x_train->values, y_train->values, rows/48);
	
//	ann_test_network(pnet, inputs, outputs);

	printf("Actual class is %g\n", y_labels->values[50002]);

	real outputs[10];
	ann_predict(pnet, &x_test->values[784], outputs);
	//	softmax(pnet);

	print_class_pred(outputs);

	// print_outputs(pnet);

	// free memory
	ann_free_network(pnet);

	tensor_free(y_labels);
	tensor_free(x_train);
	tensor_free(y_train);

	free(data);
	return 0;
}
