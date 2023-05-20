#pragma once

#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdlib.h>
#include "ann_config.h"

//----------------------------------
// Note: change to double if desired
//----------------------------------
typedef float real;

#ifdef _M_AMD64
	typedef int tsize;
#else
	typedef int tsize;
#endif

#ifndef max
#	define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

//------------------------------
// tensor structure
//------------------------------
typedef struct
{
	int rows, cols, stride;
	real *values;
	int rank;
} Tensor, *PTensor;

//------------------------------
//
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
PTensor tensor_create_from_array(int rows, int cols, real *vals);
void tensor_set_from_array(PTensor t, int rows, int cols, real *array);
void tensor_free(PTensor t);
PTensor tensor_ones(int rows, int cols);
PTensor tensor_zeros(int rows, int cols);
PTensor tensor_create_random_uniform(int rows, int cols, real min, real max);
PTensor tensor_onehot(PTensor t, int classes);

// math ops
PTensor tensor_add_scalar(PTensor t, real val);
PTensor tensor_add(PTensor a, PTensor b);
PTensor tensor_sub(PTensor a, PTensor b);
PTensor tensor_mul_scalar(PTensor t, real val);
PTensor tensor_mul(PTensor a, PTensor b);
PTensor tensor_div(PTensor a, PTensor b);
PTensor tensor_matvec(TENSOR_TRANSPOSE trans, PTensor mtx, PTensor v, PTensor dest);
PTensor tensor_square(PTensor t);
PTensor tensor_exp(PTensor t);
PTensor tensor_argmax(PTensor t);
PTensor tensor_max(PTensor t);
PTensor tensor_exp(PTensor t);
real tensor_sum(PTensor t);
PTensor tensor_axpy(real alpha, PTensor x, PTensor y);
PTensor tensor_gemm(real alpha, PTensor A, PTensor B, real beta, PTensor C);
PTensor tensor_axpby(real alpha, PTensor x, real beta, PTensor y);
PTensor tensor_outer(PTensor a, PTensor b, PTensor dest);

// manipulation
real tensor_get_element(PTensor t, int row, int col);
void tensor_set_element(PTensor t, int row, int col, real val);

PTensor tensor_slice_rows(PTensor t, int rows);
PTensor tensor_slice_cols(PTensor t, int cols);
void tensor_fill(PTensor t, real val);
void tensor_random_uniform(PTensor t, real min, real max);

void tensor_print(PTensor t);

#endif
