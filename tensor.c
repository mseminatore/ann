#include <assert.h>
#include <stdio.h>

#ifdef _WIN32
#	include <xmmintrin.h>
#endif

#include "tensor.h"

//------------------------------
// create a new tensor
//
// Note: tensors are col major
//------------------------------
PTensor tensor_create(size_t rows, size_t cols)
{
	PTensor t = NULL;

	t = malloc(sizeof(Tensor));
	if (!t)
		return NULL;

	t->rows = rows;
	t->cols = cols;
	t->rank = 2;

	t->values = malloc(rows * cols * sizeof(FLOAT));
	if (!t->values)
	{
		free(t);
		t = NULL;
	}

	return t;
}

//------------------------------
//
//------------------------------
PTensor tensor_create_from_array(size_t rows, size_t cols, FLOAT *vals)
{
	PTensor t = tensor_create(rows, cols);

	assert(t);
	if (!t)
		return NULL;

	int limit = t->rows * t->cols;

	for (int i = 0; i < limit; i++)
		t->values[i] = *vals++;

	return t;
}

//------------------------------
// free tensor memory
//------------------------------
void tensor_free(PTensor t)
{
	assert(t);
	if (!t)
		return;

	t->rows = t->cols = -1;
	free(t->values);
	free(t);
}

//------------------------------
// fill tensor with given value
//------------------------------
void tensor_fill(PTensor t, FLOAT val)
{
	assert(t);
	if (!t)
		return;

	int limit = t->rows * t->cols;

	for (int i = 0; i < limit; i++)
		t->values[i] = val;
}

//------------------------------
// create new tensor of ones
//------------------------------
PTensor tensor_ones(size_t rows, size_t cols)
{
	PTensor t = tensor_create(rows, cols);
	if (!t)
		return NULL;

	tensor_fill(t, 1.0);
	return t;
}

//------------------------------
// create new tensor of zeros
//------------------------------
PTensor tensor_zeros(size_t rows, size_t cols)
{
	PTensor t = tensor_create(rows, cols);
	if (!t)
		return NULL;

	tensor_fill(t, 0.0);
	return t;
}

//----------------------------------
// create new tensor of rand values
//----------------------------------
PTensor tensor_rand(size_t rows, size_t cols)
{
	PTensor t = tensor_create(rows, cols);
	if (!t)
		return NULL;

	int limit = t->rows * t->cols;
	for (int i = 0; i < limit; i++)
		t->values[i] = (FLOAT)rand() / (FLOAT)RAND_MAX;

	return t;
}

//------------------------------
// add scalar to tensor
//------------------------------
PTensor tensor_add_scalar(PTensor t, FLOAT val)
{
	if (!t)
		return NULL;

	int limit = t->rows * t->cols;
	for (int i = 0; i < limit; i++)
		t->values[i] += val;

	return t;
}

//------------------------------
// add two tensors
//------------------------------
PTensor tensor_add(PTensor a, PTensor b)
{
	if (!a || !b)
		return NULL;

	// shape must be the same
	if (a->rows != b->rows || a->cols != b->cols)
	{
		puts("err: tensor_add shapes not equal");
		return NULL;
	}

	int limit = a->rows * a->cols;
	for (int i = 0; i < limit; i++)
		a->values[i] += b->values[i];

	return a;
}

//------------------------------
// multiply two tensors
//------------------------------
PTensor tensor_mul(PTensor a, PTensor b)
{
	if (!a || !b)
		return NULL;

	// shape must be the same
	if (a->rows != b->rows || a->cols != b->cols)
	{
		puts("err: tensor_mul shapes not equal");
		return NULL;
	}

	return a;
}

//------------------------------
// get a tensor component
//------------------------------
FLOAT tensor_get(PTensor t, size_t row, size_t col)
{
	if (row > t->rows || col > t->cols)
		return 0.0;

	return t->values[row * t->cols + col];
}

//------------------------------
// set a tensor component
//------------------------------
void tensor_set(PTensor t, size_t row, size_t col, FLOAT val)
{
	if (row > t->rows || col > t->cols)
		return;

	t->values[row * t->cols + col] = val;
}

//-------------------------------------------
// slice out rows from the end of the tensor
//-------------------------------------------
PTensor tensor_slice_rows(PTensor t, size_t row_start)
{
	PTensor r;

	assert(t);
	if (!t || row_start > t->rows)
		return NULL;

	r = tensor_create(t->rows - row_start, t->cols);
	if (!r)
		return NULL;

	// copy the elements
	FLOAT *v = &(t->values[row_start * t->cols]);
	for (size_t i = 0; i < (t->rows - row_start) * t->cols; i++)
		r->values[i] = *v++;

	// adjust size of t to remove sliced rows
	// NOTE: we don't release t's extra memory
	t->rows -= row_start;
	
	return r;
}

//-------------------------------------------
// slice out cols from the end of the tensor
//-------------------------------------------
PTensor tensor_slice_cols(PTensor t, size_t col_start)
{
	PTensor r;

	assert(t);
	if (!t || col_start > t->cols)
		return NULL;

	// create result tensor
	r = tensor_create(t->rows, t->cols - col_start);
	if (!r)
		return NULL;

	// copy the elements
	for (size_t row = 0; row < t->rows; row++)
	{
		for (size_t col = col_start; col < t->cols; col++)
			tensor_set(r, row, col, tensor_get(t, row, col));
	}

	// fixup t

	// adjust size of t to remove sliced cols
	// NOTE: we don't release t's extra memory
	t->cols -= col_start;
	
	return r;
}

//------------------------------
//
//------------------------------
PTensor tensor_dot(PTensor a, PTensor b)
{
	PTensor t = NULL;
	
	return t;
}

//------------------------------
// print the tensor
//------------------------------
void tensor_print(PTensor t)
{
	assert(t);
	if (!t)
		return;

	FLOAT *v = t->values;
	putchar('[');
	for (size_t row = 0; row < t->rows; row++)
	{
		putchar('[');
		for (size_t col = 0; col < t->cols; col++)
		{
			if (col != 0)
				putchar(',');

			printf(" %g", *v++);
		}

		putchar(']');
		if (row + 1 != t->rows)
			puts("");
	}
	puts("]\n");
}
