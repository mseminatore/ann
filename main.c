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
	int class = ann_class_prediction(data, 10);

	printf("\nClass prediction is: %d\n\n", class + 1);
}

//------------------------------
// display a 28x28 image from flat data
//------------------------------
void print28x28(real *data)
{
	char c;

	puts("\nInput image\n");

	for (int col = 0; col < 28; col++)
	{
		for (int row = 0; row < 28; row++)
		{
			real val = data[row * 28 + col];
			if (val == 0.0)
				c = ' ';
			//else if (val < 0.25)
			//	c = '.';
			//else if (val < 0.5)
			//	c = '+';
			else
				c = '*';

			putchar(c);
		}
		puts("");
	}
}

//------------------------------
// evaluate the accuracy 
//------------------------------
real ann_evaluate(PNetwork pnet, PTensor inputs, PTensor outputs)
{
	size_t correct = 0;

	if (!pnet || !inputs || !outputs)
	{
		return -1.0;
	}

	real *pred = alloca(outputs->cols * sizeof(real));
	int pred_class, act_class;

	for (size_t i = 0; i < inputs->rows; i++)
	{
		ann_predict(pnet, &inputs->values[i * inputs->cols], pred);
		pred_class = ann_class_prediction(pred, 10);
		act_class = ann_class_prediction(&outputs->values[i * outputs->cols], 10);

		if (pred_class == act_class)
			correct++;
	}

//	printf("Actual class is %s (%g)\n", classes[(int)y_labels->values[50001]], y_labels->values[50001]);
	//print28x28(&x_test->values[0]);
	//print_class_pred(outputs);

	return (real)correct / inputs->rows;
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

	PNetwork pnet = ann_make_network();

	real *data, *test_data;
	size_t rows, stride, test_rows, test_stride;

	// load the data
	if (argc > 1)
		ann_load_csv(argv[1], CSV_HAS_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("fashion-mnist_train.csv", CSV_HAS_HEADER, &data, &rows, &stride);

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

	// define our network
	ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 912, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	pnet->epochLimit = 5;

	// train the network
	ann_train_network(pnet, x_train->values, y_train->values, x_train->rows);
	
	// evaluate the network against the test data
	real acc = ann_evaluate(pnet, x_test, y_test);
	printf("Test accuracy: %g%%\n", acc * 100);

	// print_outputs(pnet);

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
