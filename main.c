#include <stdio.h>
#include "ann.h"

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	PNetwork pnet = make_network();

	// define our network
	add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	add_layer(pnet, 2, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	set_learning_rate(pnet, 0.5);

	train_network(pnet);
	
	test_network(pnet);

	free_network(pnet);
	return 0;
}
