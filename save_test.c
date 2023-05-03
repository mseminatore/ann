#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ann.h"


//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
	real *test_data;
	size_t test_rows, test_stride;

    PNetwork pnet = ann_load_network("mnist-fashion.nn");

	// load the test data
	ann_load_csv("fashion-mnist_test.csv", CSV_HAS_HEADER, &test_data, &test_rows, &test_stride);
	
    PTensor y_test_labels = tensor_create_from_array(test_rows, test_stride, test_data);
	free(test_data);

	PTensor x_test = tensor_slice_cols(y_test_labels, 1);
	PTensor y_test = tensor_onehot(y_test_labels, 10);

	// normalize inputs
	tensor_mul_scalar(x_test, (real)(1.0 / 255.0));
	
	// evaluate the network against the test data
	real acc = ann_evaluate(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

	// free memory
	ann_free_network(pnet);
    
	tensor_free(y_test_labels);
	tensor_free(x_test);
	tensor_free(y_test);

	return 0;
}
