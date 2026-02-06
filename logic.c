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
#include "ann.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

	real *data;
	int rows, stride;
	char *filename = "and.csv";

	// load the data
	if (argc > 1)
		filename = argv[1];

	printf("Loading %s...", filename);
	CHECK_OK(ann_load_csv(filename, CSV_NO_HEADER, &data, &rows, &stride));
	puts("done.");

	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);
	if (!pnet)
		return ERR_FAIL;

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	if (!x_train)
		return ERR_FAIL;

	PTensor y_train = tensor_slice_cols(x_train, 2);
	if (!y_train)
		return ERR_FAIL;

	// define our network
	CHECK_OK(ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL));
	CHECK_OK(ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID));

	ann_train_network(pnet, x_train, y_train, x_train->rows);
	
	real outputs[1];
	CHECK_OK(ann_predict(pnet, &data[0], outputs));

	print_outputs(pnet);

	ann_free_network(pnet);

	free(data);
	return ERR_OK;
}
