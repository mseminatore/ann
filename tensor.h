#pragma once

#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdlib.h>
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
	size_t rows, cols, stride;
	real *values;
	size_t rank;
} Tensor, *PTensor;

//------------------------------
// function decls
//------------------------------

// creation/destruction
PTensor tensor_create(size_t rows, size_t cols);
PTensor tensor_create_from_array(size_t rows, size_t cols, real *vals);
void tensor_set_from_array(PTensor t, size_t rows, size_t cols, real *array);
void tensor_free(PTensor t);
PTensor tensor_ones(size_t rows, size_t cols);
PTensor tensor_zeros(size_t rows, size_t cols);
PTensor tensor_create_random_uniform(size_t rows, size_t cols, real min, real max);
PTensor tensor_onehot(PTensor t, size_t classes);

// math ops
PTensor tensor_add_scalar(PTensor t, real val);
PTensor tensor_add(PTensor a, PTensor b);
PTensor tensor_sub(PTensor a, PTensor b);
PTensor tensor_mul_scalar(PTensor t, real val);
PTensor tensor_mul(PTensor a, PTensor b);
PTensor tensor_div(PTensor a, PTensor b);
PTensor tensor_dot(PTensor a, PTensor b, PTensor c);
PTensor tensor_square(PTensor t);
PTensor tensor_exp(PTensor t);
PTensor tensor_argmax(PTensor t);
PTensor tensor_max(PTensor t);
PTensor tensor_exp(PTensor t);
real tensor_sum(PTensor t);
PTensor tensor_axpy(real a, PTensor x, PTensor y);
PTensor tensor_gemm(real alpha, PTensor A, PTensor B, real beta, PTensor C);

// manipulation
real tensor_get_element(PTensor t, size_t row, size_t col);
void tensor_set_element(PTensor t, size_t row, size_t col, real val);
void tensor_set_all(PTensor, real val);

PTensor tensor_slice_rows(PTensor t, size_t rows);
PTensor tensor_slice_cols(PTensor t, size_t cols);
void tensor_fill(PTensor t, real val);
void tensor_random_uniform(PTensor t, real min, real max);

void tensor_print(PTensor t);

#endif
