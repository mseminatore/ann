/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

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
	char *network_filename = "mnist-fashion.nna";
	char *test_filename = "fashion-mnist_test.csv";

	real *test_data;
	int test_rows, test_stride;

	if (argc > 1)
		network_filename = argv[1];

	if (argc > 2)
		test_filename = argv[2];

    PNetwork pnet = ann_load_network(network_filename);
	if (!pnet)
		return ERR_FAIL;

	// load the test data
	printf("Loading %s...", test_filename);
	CHECK_OK(ann_load_csv(test_filename, CSV_HAS_HEADER, &test_data, &test_rows, &test_stride));
	puts("done.");

    PTensor y_test_labels = tensor_create_from_array(test_rows, test_stride, test_data);
	free(test_data);

	PTensor x_test = tensor_slice_cols(y_test_labels, 1);
	if (!x_test)
		return ERR_FAIL;

	PTensor y_test = tensor_onehot(y_test_labels, 10);
	if (!y_test)
		return ERR_FAIL;

	// normalize inputs
	tensor_mul_scalar(x_test, (real)(1.0 / 255.0));
	
	// evaluate the network against the test data
	real acc = ann_evaluate_accuracy(pnet, x_test, y_test);
	printf("\nTest accuracy: %g%%\n", acc * 100);

	// free memory
	ann_free_network(pnet);
    
	tensor_free(y_test_labels);
	tensor_free(x_test);
	tensor_free(y_test);

	return ERR_OK;
}
