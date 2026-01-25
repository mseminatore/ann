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

#pragma once

#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdlib.h>
#include <stdint.h>
#include "ann_config.h"

//----------------------------------
// Note: change to double if desired
//----------------------------------
typedef float real;

#ifndef max
#	define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

//------------------------------
// tensor structure
//------------------------------
typedef struct
{
	int rows, cols;
	int stride;			// not currently used
	real *values;
	int rank;			// not currently used
} Tensor, *PTensor;

//------------------------------
// flags for matvec operation
//------------------------------
typedef enum TENSOR_TRANSPOSE {
	Tensor_NoTranspose, 
	Tensor_Transpose
} TENSOR_TRANSPOSE;

//------------------------------
// function decls
//------------------------------

// creation/destruction
PTensor tensor_create(int rows, int cols);
PTensor tensor_create_from_array(int rows, int cols, const real *vals);
void tensor_set_from_array(PTensor t, int rows, int cols, const real *array);
void tensor_free(PTensor t);
PTensor tensor_ones(int rows, int cols);
PTensor tensor_zeros(int rows, int cols);
PTensor tensor_create_random_uniform(int rows, int cols, real min, real max);
PTensor tensor_onehot(const PTensor t, int classes);
PTensor tensor_copy(const PTensor t);

// math ops
PTensor tensor_add_scalar(PTensor t, real val);
PTensor tensor_add(const PTensor a, const PTensor b);
PTensor tensor_sub(const PTensor a, const PTensor b);
PTensor tensor_mul_scalar(PTensor t, real val);
PTensor tensor_mul(const PTensor a, const PTensor b);
PTensor tensor_div(const PTensor a, const PTensor b);
PTensor tensor_matvec(TENSOR_TRANSPOSE trans, real alpha, const PTensor mtx, real beta, const PTensor v, PTensor dest);
PTensor tensor_square(PTensor t);
PTensor tensor_exp(PTensor t);
PTensor tensor_argmax(const PTensor t);
PTensor tensor_max(const PTensor t);
real tensor_sum(const PTensor t);
PTensor tensor_axpy(real alpha, const PTensor x, PTensor y);
PTensor tensor_gemm(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);
PTensor tensor_axpby(real alpha, const PTensor x, real beta, PTensor y);
PTensor tensor_outer(real alpha, const PTensor a, const PTensor b, PTensor dest);
PTensor tensor_heaviside(const PTensor a);

// manipulation
real tensor_get_element(const PTensor t, int row, int col);
void tensor_set_element(PTensor t, int row, int col, real val);

PTensor tensor_slice_rows(const PTensor t, int rows);
PTensor tensor_slice_cols(const PTensor t, int cols);
void tensor_fill(PTensor t, real val);
void tensor_random_uniform(PTensor t, real min, real max);

// IO
void tensor_print(const PTensor t);
int tensor_save_to_file(const PTensor t, const char *filename);

#endif
