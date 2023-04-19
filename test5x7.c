#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

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

// add noise
void add_noise(real *data, size_t size, int amount)
{
	for (int i = 0; i < amount; i++)
	{
		int index = rand() % size;

		data[index] = (data[index] == 1.0) ? (real)0.0 : (real)1.0;
	}
}

#ifdef _WIN32
#define DIR_FIX "..\\"
#endif

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
		ann_load_csv(argv[1], CSV_NO_HEADER, &data, &rows, &stride);
	else
		ann_load_csv(DIR_FIX "num5x7.csv", CSV_NO_HEADER, &data, &rows, &stride);

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	PTensor y_train = tensor_slice_cols(x_train, 35);
	
	pnet->adaptiveLearning = 0;
	pnet->learning_rate = 0.35;

	// define our network
	ann_add_layer(pnet, 35, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 48, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	ann_train_network(pnet, x_train, y_train, x_train->rows);

	int correct = 0;

	for (int i = 0; i < 10; i++)
	{
		add_noise(&data[i * 45], 35, 2);

		real outputs[10];
		ann_predict(pnet, &data[i * 45], outputs);

		print5x7(&data[i * 45]);
		print_class_pred(outputs);
		int pred_class = ann_class_prediction(outputs, 10);
		int class = ann_class_prediction(&y_train->values[i * 10], 10);

		if (pred_class == class)
			correct++;

		printf("Actual class: %d\n", class + 1);
	}

	real acc = ann_evaluate(pnet, x_train, y_train);
	printf("Train accuracy: %g%%\n", acc * 100);
	printf("Test accuracy: %g%%\n", (real)correct * 10);

	ann_free_network(pnet);

	tensor_free(x_train);
	tensor_free(y_train);

	free(data);
	return 0;
}
