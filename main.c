#include <stdio.h>
#include "ann.h"

// AND function
//real inputs[] = {
//	0.0, 0.0, 
//	0.0, 1.0,
//	1.0, 0.0,
//	1.0, 1.0
//};
//
//real outputs[] = {
//	0.0,
//	0.0,
//	0.0,
//	1.0
//};

// OR function
real inputs[] = {
	0.0, 0.0, 
	0.0, 1.0,
	1.0, 0.0,
	1.0, 1.0
};

real outputs[] = {
	0.0,
	1.0,
	1.0,
	1.0
};

// XOR function
//real inputs[] = {
//	0.0, 0.0,
//	0.0, 1.0,
//	1.0, 0.0,
//	1.0, 1.0
//};
//
//real outputs[] = {
//	0.0,
//	1.0,
//	1.0,
//	0.0
//};

//------------------------------
//
//------------------------------
int main(int argc, char *argv[])
{
	PNetwork pnet = make_network();

	// define our network
	add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

	set_learning_rate(pnet, 0.2);

	init_weights(pnet);

	train_network(pnet, inputs, 4, outputs);
	
	test_network(pnet, inputs, outputs);

	free_network(pnet);
	return 0;
}
