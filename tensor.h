#pragma once

#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdlib.h>

#define FLOAT double

//------------------------------
// tensor structure
//------------------------------
typedef struct
{
	size_t rows, cols;
	FLOAT *values;
	size_t rank;
} Tensor, *PTensor;

//------------------------------
// function decls
//------------------------------
PTensor tensor_create(size_t rows, size_t cols);
PTensor tensor_create_from_array(size_t rows, size_t cols, FLOAT *vals);
void tensor_free(PTensor t);
FLOAT tensor_get(PTensor t, size_t row, size_t col);
void tensor_set(PTensor t, size_t row, size_t col, FLOAT val);
void tensor_fill(PTensor t, FLOAT val);
void tensor_randomize(PTensor t);
PTensor tensor_ones(size_t rows, size_t cols);
PTensor tensor_zeros(size_t rows, size_t cols);
PTensor tensor_rand(size_t rows, size_t cols);
PTensor tensor_add_scalar(PTensor t, FLOAT val);
PTensor tensor_add(PTensor a, PTensor b);
PTensor tensor_mul(PTensor a, PTensor b);
FLOAT tensor_get(PTensor t, size_t row, size_t col);
void tensor_set(PTensor t, size_t row, size_t col, FLOAT val);
PTensor tensor_slice_rows(PTensor t, size_t rows);
PTensor tensor_slice_cols(PTensor t, size_t cols);
void tensor_print(PTensor t);
PTensor tensor_dot(PTensor a, PTensor b);

#endif
