#pragma once

#ifndef __TENSOR_H
#define __TENSOR_H

#include <stdlib.h>

//------------------------------
// tensor structure
//------------------------------
typedef struct
{
	size_t rows, cols;
	double *values;
	size_t rank;
} Tensor, *PTensor;

//------------------------------
// function decls
//------------------------------
PTensor tensor_create(size_t rows, size_t cols);
void tensor_free(PTensor t);
double tensor_get(PTensor t, size_t row, size_t col);
void tensor_set(PTensor t, size_t row, size_t col, double val);
void tensor_fill(PTensor t, double val);
PTensor tensor_ones(size_t rows, size_t cols);
PTensor tensor_zeros(size_t rows, size_t cols);
PTensor tensor_rand(size_t rows, size_t cols);
PTensor tensor_add_scalar(PTensor t, double val);
PTensor tensor_add(PTensor a, PTensor b);
PTensor tensor_mul(PTensor a, PTensor b);
double tensor_get(PTensor t, size_t row, size_t col);
void tensor_set(PTensor t, size_t row, size_t col, double val);
PTensor tensor_slice(PTensor t, size_t rows);

#endif
