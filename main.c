#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

// AND function
real inputs[] = {
	0.0, 0.0, 
	0.0, 1.0,
	1.0, 0.0,
	1.0, 1.0
};

real outputs[] = {
	0.0,
	0.0,
	0.0,
	1.0
};

// OR function
// real inputs[] = {
// 	0.0, 0.0, 
// 	0.0, 1.0,
// 	1.0, 0.0,
// 	1.0, 1.0
// };

// real outputs[] = {
// 	0.0,
// 	1.0,
// 	1.0,
// 	1.0
// };

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

	real *data;
	int count;

	load_csv("data.csv", &data, &count);

	//for (int i = 1; i <= count; i++)
	//{
	//	printf("%g, ", data[i-1]);
	//	if (i % 3 == 0)
	//		puts("");
	//}

	// define our network
	add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
	add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

//	set_learning_rate(pnet, 0.2);

	train_network(pnet, inputs, 4, outputs);
	
	test_network(pnet, inputs, outputs);

	free_network(pnet);

	free(data);
	return 0;
}
