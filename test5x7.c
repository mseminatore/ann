#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//---------------------------------------
// print the Ci with highest probability
//---------------------------------------
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

	printf("\n5x7 character prediction is: %d\n\n", num + 1);
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

		data[index] = (data[index] == 1.0) ? 0.0 : 1.0;
	}
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
		ann_load_csv(argv[1], CSV_NO_HEADER, &data, &rows, &stride);
	else
		ann_load_csv("num5x7.csv", CSV_NO_HEADER, &data, &rows, &stride);

	// define our network
	ann_add_layer(pnet, 35, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(pnet, 48, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	ann_train_network(pnet, data, rows, stride);

	for (int i = 0; i < 10; i++)
	{
		add_noise(&data[i * 45], 35, 5);

		real outputs[10];
		ann_predict(pnet, &data[i * 45], outputs);

		print5x7(&data[i * 45]);
		print_class_pred(outputs);
	}

	ann_free_network(pnet);

	free(data);
	return 0;
}
