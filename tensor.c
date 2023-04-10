#include <assert.h>
#include <stdio.h>

#if defined(_WIN32) || defined(__x86_64__)
#include <xmmintrin.h>
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#	define TENSOR_SSE
//#	define TENSOR_AVX
#	include "sse2neon.h"
#endif

#include "tensor.h"

#ifdef __AVX__
#	define TENSOR_AVX
#endif

#if _M_IX86_FP == 2
#	define TENSOR_SSE
#endif

#ifdef TENSOR_AVX
#	define TENSOR_ALIGN 32
#else
#	define TENSOR_ALIGN 16
#endif

//------------------------------
//
//------------------------------
void *tmalloc(size_t size)
{
	#if defined(_WIN32) && !defined(_WIN64)
		return _aligned_malloc(size, TENSOR_ALIGN);
	#else
		return malloc(size);
	#endif
}

//------------------------------
//
//------------------------------
void tfree(void *block)
{
	#if defined(_WIN32) && !defined(_WIN64)
		_aligned_free(block);
	#else
		free(block);
	#endif
}

//------------------------------
//
//------------------------------
void* trealloc(void *block, size_t size)
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

	t->values = tmalloc(rows * cols * sizeof(FLOAT));
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
	tfree(t->values);
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
	int i = 0;

#ifdef TENSOR_AVX
	__m256 va = _mm256_set1_ps(val);
	for (; i + 8 < limit; i += 8)
		_mm256_store_ps(&(t->values[i]), va);
#endif

#ifdef TENSOR_SSE
	__m128 a = _mm_set1_ps(val);
	for (;i + 4 < limit; i += 4)
		_mm_store_ps(&(t->values[i]), a);
#endif

	for (; i < limit; i++)
		t->values[i] = val;
}

//------------------------------
// fill tensor with given value
//------------------------------
void tensor_randomize(PTensor t)
{
	assert(t);
	if (!t)
		return;

	int limit = t->rows * t->cols;

	for (int i = 0; i < limit; i++)
		t->values[i] = (FLOAT)rand() / (FLOAT)RAND_MAX;
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
	int i = 0;

#ifdef TENSOR_AVX
	__m256 va = _mm256_set1_ps(val);
	for (; i + 8 < limit; i += 8)
	{
		__m256 vb = _mm256_load_ps(&t->values[i]);
		__m256 vc = _mm256_add_ps(vb, va);
		_mm256_store_ps(&t->values[i], vc);
	}

#endif

#ifdef TENSOR_SSE
	__m128 a = _mm_set1_ps(val);

	for (; i + 4 < limit; i += 4)
	{
		__m128 b = _mm_load_ps(&t->values[i]);
		__m128 c = _mm_add_ps(b, a);
		_mm_store_ps(&t->values[i], c);
	}
#endif

	for (; i < limit; i++)
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
	int i = 0;

#ifdef TENSOR_AVX
	for (; i + 8 < limit; i += 8)
	{
		__m256 a256 = _mm256_load_ps(&a->values[i]);
		__m256 b256 = _mm256_load_ps(&b->values[i]);
		__m256 c256 = _mm256_add_ps(a256, b256);
		_mm256_store_ps(&a->values[i], c256);
	}
#endif

#ifdef TENSOR_SSE
	for (; i + 4 < limit; i += 4)
	{
		__m128 av = _mm_load_ps(&a->values[i]);
		__m128 bv = _mm_load_ps(&b->values[i]);
		__m128 c = _mm_add_ps(av, bv);
		_mm_store_ps(&a->values[i], c);
	}

#endif

	for (; i < limit; i++)
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

	int limit = a->rows * a->cols;
	int i = 0;

#ifdef TENSOR_AVX
#endif

#ifdef TENSOR_SSE
	for (; i + 4 < limit; i += 4)
	{
		__m128 av = _mm_load_ps(&a->values[i]);
		__m128 bv = _mm_load_ps(&b->values[i]);
		__m128 c = _mm_mul_ps(av, bv);
		_mm_store_ps(&a->values[i], c);
	}

#endif

	for (; i < limit; i++)
		a->values[i] *= b->values[i];

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
	t->rows -= row_start;

	// TODO: we don't release t's extra memory

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
			FLOAT v = tensor_get(t, row, col);
			tensor_set(r, row, col - col_start, v);
		}
	}

	// fixup t to remove the cols
	FLOAT *values = t->values;
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

	// TODO: we don't release t's extra memory

	return r;
}

//-------------------------------
// compute the tensor dot product
//-------------------------------
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
