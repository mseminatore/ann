#include <assert.h>
#include <stdio.h>
#include <math.h>

#if defined(_WIN32) || defined(__x86_64__)
#	include <immintrin.h>
#	define TENSOR_SSE
#endif

#if defined(__aarch64__)
#	define TENSOR_SSE
#	include "sse2neon.h"
#endif

#include "tensor.h"

#if defined(USE_BLAS)
#	include <cblas.h>
#endif

#ifdef __AVX__
#	define TENSOR_AVX
#endif

#if _M_IX86_FP == 2
#	define TENSOR_SSE
#	define TENSOR_AVX
#endif

#ifdef TENSOR_AVX
#	define TENSOR_ALIGN 32
#else
#	define TENSOR_ALIGN 16
#endif

//------------------------------
// aligned alloc where needed
//------------------------------
static void *tmalloc(size_t size)
{
	#if defined(_WIN32) && !defined(_WIN64)
		return _aligned_malloc(size, TENSOR_ALIGN);
	#else
		return malloc(size);
	#endif
}

//------------------------------
// aligned free where needed
//------------------------------
static void tfree(void *block)
{
	#if defined(_WIN32) && !defined(_WIN64)
		_aligned_free(block);
	#else
		free(block);
	#endif
}

//------------------------------
// aligned realloc where needed
//------------------------------
static void *trealloc(void *block, size_t size)
{
	#if defined(_WIN32) && !defined(_WIN64)
		return _aligned_realloc(block, size, TENSOR_ALIGN);
	#else
		return realloc(block, size);
	#endif
}

//------------------------------
// create a new tensor
//
// Note: tensors are ROW major
//------------------------------
PTensor tensor_create(size_t rows, size_t cols)
{
	PTensor t = NULL;

	t = malloc(sizeof(Tensor));
	if (!t)
		return NULL;

	t->rows = rows;
	t->cols = cols;

	// make sure each row of tensor is properly aligned
	t->stride = (cols + TENSOR_ALIGN - 1) & (SIZE_MAX ^ 0x1f);

	// only rank 2 tensors supported now
	t->rank = 2;

	t->values = tmalloc(rows * cols * sizeof(real));
	if (!t->values)
	{
		free(t);
		t = NULL;
	}

	return t;
}

//------------------------------
// create new tensor from array
//------------------------------
PTensor tensor_create_from_array(size_t rows, size_t cols, real *array)
{
	PTensor t = tensor_create(rows, cols);

	assert(t);
	assert(array);
	if (!t || !array)
		return NULL;

	tensor_set_from_array(t, rows, cols, array);

	return t;
}

//------------------------------
// set tensor values from array
//------------------------------
void tensor_set_from_array(PTensor t, size_t rows, size_t cols, real *array)
{
	assert(t);
	assert(array);
	if (!t || !array)
		return;

	if (t->rows != rows || t->cols != cols)
	{
		assert(0 && "tensor: invalid shape");
		return;
	}

	// TODO - check values alignment?

	size_t limit = t->rows * t->cols;

	for (size_t i = 0; i < limit; i++)
		t->values[i] = *array++;
}

//------------------------------
// free tensor memory
//------------------------------
void tensor_free(PTensor t)
{
	if (!t)
	{
		assert(t);
		return;
	}

	t->rows = t->cols = t->stride = -1;
	tfree(t->values);
	free(t);
}

//------------------------------
// fill tensor with given value
//------------------------------
void tensor_fill(PTensor t, real val)
{
	assert(t);
	if (!t)
		return;

	size_t limit = t->rows * t->cols;
	size_t i = 0;

	for (; i < limit; i++)
		t->values[i] = val;
}

//------------------------------
// fill tensor with given value
//------------------------------
void tensor_random_uniform(PTensor t, real min, real max)
{
	assert(t);
	if (!t)
		return;

	size_t limit = t->rows * t->cols;

	for (size_t i = 0; i < limit; i++)
	{
		real r = (real)rand() / (real)RAND_MAX;
		real scale = max - min;

		r *= scale;
		r += min;

		t->values[i] = r;
	}
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
PTensor tensor_create_random_uniform(size_t rows, size_t cols, real min, real max)
{
	PTensor t = tensor_create(rows, cols);
	if (!t)
		return NULL;

	tensor_random_uniform(t, min, max);

	return t;
}

//------------------------------
// returns y = a * x + b * y
//------------------------------
PTensor tensor_axpy(real a, PTensor x, PTensor y)
{
	if (!x || !y || x->rows != y->rows || x->cols != y->cols)
	{
		assert(0 && "tensor: invalid shape.");
		return NULL;
	}

	size_t limit = x->rows * x->cols;

//#ifdef USE_BLAS
//	cblas_saxpy(limit, a, x->values, 1, y->values, 1);
//#else

	size_t i = 0;

	if (a == 1.0)
	{
		for (; i < limit; i++)
			y->values[i] += x->values[i];
	}
	else
	{
		for (; i < limit; i++)
			y->values[i] += a * x->values[i];
	}
//#endif

	return y;
}

//------------------------------
// add scalar to tensor
//------------------------------
PTensor tensor_add_scalar(PTensor t, real val)
{
	if (!t)
		return NULL;

	size_t limit = t->rows * t->cols;
	size_t i = 0;

	for (; i < limit; i++)
		t->values[i] += val;

	return t;
}

//------------------------------
// add two tensors (a = a + b)
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

	size_t limit = a->rows * a->cols;
	size_t i = 0;

	for (; i < limit; i++)
		a->values[i] += b->values[i];

	return a;
}

//------------------------------
// sub two tensors (a = a - b)
//------------------------------
PTensor tensor_sub(PTensor a, PTensor b)
{
	if (!a || !b)
		return NULL;

	// shape must be the same
	if (a->rows != b->rows || a->cols != b->cols)
	{
		puts("err: tensor_add shapes not equal");
		return NULL;
	}

	size_t limit = a->rows * a->cols;
	size_t i = 0;
	
	for (; i < limit; i++)
		a->values[i] -= b->values[i];

	return a;
}

//------------------------------
// square the tensor
//------------------------------
PTensor tensor_square(PTensor t)
{
	if (!t)
		return NULL;

	size_t limit = t->rows * t->cols;
	size_t i = 0;

	for (; i < limit; i++)
		t->values[i] *= t->values[i];

	return t;
}

//------------------------------
// multiply tensor by a scalar
//------------------------------
PTensor tensor_mul_scalar(PTensor t, real val)
{
	if (!t)
		return NULL;

	size_t limit = t->rows * t->cols;
	size_t i = 0;

	for (; i < limit; i++)
		t->values[i] *= val;

	return t;

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
		assert(0 && "tensor: invalid shape");
		return NULL;
	}

	int limit = a->rows * a->cols;
	int i = 0;

	for (; i < limit; i++)
		a->values[i] *= b->values[i];

	return a;
}

//------------------------------
// divide tensor a by b
//------------------------------
PTensor tensor_div(PTensor a, PTensor b)
{
	if (!a || !b)
		return NULL;

	if (a->cols != b->cols || b->rows != 1)
	{
		assert(0 && "tensor: invalid shape");
		return NULL;
	}

	int limit = a->rows * a->cols;
	int i = 0;

	for (; i < limit; i++)
		a->values[i] /= b->values[i];

	return a;
}

//-------------------------------
// return a tensor containing max
// col values from each row of t
//-------------------------------
PTensor tensor_max(PTensor t)
{
	if (!t)
		return NULL;

	PTensor r = tensor_zeros(1, t->cols);

	for (size_t row = 0; row < t->rows; row++)
	{
		for (size_t col = 0; col < t->cols; col++)
		{
			real a = tensor_get_element(r, 0, col);
			real b = tensor_get_element(t, row, col);
			real val = max(a, b);
			tensor_set_element(r, 0, col, val);
		}
	}

	return r;
}

//------------------------------
// get a tensor component
//------------------------------
real tensor_get_element(PTensor t, size_t row, size_t col)
{
	if (!t || row > t->rows || col > t->cols)
		return 0.0;

	return t->values[row * t->cols + col];
}

//------------------------------
// set a tensor component
//------------------------------
void tensor_set_element(PTensor t, size_t row, size_t col, real val)
{
	if (!t || row > t->rows || col > t->cols)
		return;

	t->values[row * t->cols + col] = val;
}

//------------------------------
// set all tensor elements to val
//------------------------------
void tensor_set_all(PTensor t, real val)
{
	if (!t)
		return;

	int limit = t->rows * t->cols;
	int i = 0;

	for (; i < limit; i++)
		t->values[i] = val;
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
	real *v = &(t->values[row_start * t->cols]);
	for (size_t i = 0; i < (t->rows - row_start) * t->cols; i++)
		r->values[i] = *v++;

	// adjust size of t to remove sliced rows
	t->rows -= (t->rows - row_start);

	// release t's extra memory
	t->values = trealloc(t->values, t->rows * t->cols * sizeof(real));
	assert(t->values);

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
		{
			real v = tensor_get_element(t, row, col);
			tensor_set_element(r, row, col - col_start, v);
		}
	}

	// fixup t to remove the cols
	real *values = t->values;
	for (size_t row = 0; row < t->rows; row++)
	{
		for (size_t col = 0; col < t->cols; col++)
		{
			if (col >= col_start)
				continue;

			*values++ = t->values[row * t->cols + col];
		}
	}

	// adjust size of t to remove sliced cols
	t->cols -= (t->cols - col_start);

	// release t's extra memory
	t->values = trealloc(t->values, t->rows * t->cols * sizeof(real));
	assert(t->values);

	return r;
}

//-------------------------------
// turn int vector to onehot
//-------------------------------
PTensor tensor_onehot(PTensor t, size_t classes)
{
	if (t->cols > 1)
	{
		assert(t->cols == 1);
		return NULL;
	}

	PTensor r = tensor_zeros(t->rows, classes);

	for (size_t row = 0; row < t->rows; row++)
	{
		tensor_set_element(r, row, (size_t)tensor_get_element(t, row, 0), (real)1.0);
	}

	return r;
}

//------------------------------
// return the horizontal sum of t
//------------------------------
real tensor_sum(PTensor t)
{
	real sum = (real)0.0;

	if (!t || t->rows > 1)
	{
		assert(0 && "tensor: invalid tensor");
		return (real)0.0;
	}

	int limit = t->rows * t->cols;

#ifdef USE_BLAS
	sum = cblas_sasum(limit, t->values, 1);
#else
	int i = 0;
	for (; i < limit; i++)
		sum += t->values[i];
#endif

	return sum;
}

//------------------------------
// return exp of tensor t 
//------------------------------
PTensor tensor_exp(PTensor t)
{
	if (!t)
		return NULL;

	int limit = t->rows * t->cols;
	int i = 0;

	for (; i < limit; i++)
		t->values[i] = (real)exp(t->values[i]);

	return t;
}

//-------------------------------
// return tensor containing argmax of t
//-------------------------------
PTensor tensor_argmax(PTensor t)
{
	assert(0);
	if (!t)
		return NULL;

	return NULL;
}

//-------------------------------
// compute the tensor dot product
//-------------------------------
PTensor tensor_dot(PTensor a, PTensor b, PTensor dest)
{
	if (a->cols != b->cols)
	{
		assert(0 && "tensor: invalid shape.");
		return NULL;
	}

#ifdef USE_BLAS
	cblas_sgemv(CblasRowMajor, CblasNoTrans, a->rows, a->cols, 1.0, a->values, a->cols, b->values, 1, 0.0, dest->values, 1);
#else

	real sum;
	for (size_t a_row = 0; a_row < a->rows; a_row++)
	{
		for (size_t b_row = 0; b_row < b->rows; b_row++)
		{
			sum = (real)0.0;

			for (size_t a_col = 0; a_col < a->cols; a_col++)
			{
				sum += a->values[a_row * a->cols + a_col] * b->values[b_row * b->cols + a_col];
			}

			dest->values[b_row * b->cols + a_row] = sum;
		}
	}
#endif

	return dest;
}

//-------------------------------
//
//-------------------------------
PTensor tensor_gemm(real alpha, PTensor A, PTensor B, real beta, PTensor C)
{
	return C;
}

//------------------------------
// print the tensor
//------------------------------
void tensor_print(PTensor t)
{
	assert(t);
	if (!t)
		return;

	real *v = t->values;
	putchar('[');
	for (size_t row = 0; row < t->rows; row++)
	{
		putchar('[');
		for (size_t col = 0; col < t->cols; col++)
		{
			if (col != 0)
				putchar(',');

			printf(" %3.2g", *v++);
		}

		putchar(']');
		if (row + 1 != t->rows)
			puts("");
	}
	puts("]\n");
}
