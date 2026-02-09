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

// ============================================================================
// TENSOR CREATION AND DESTRUCTION
// ============================================================================

/**
 * Create an empty tensor of specified dimensions.
 * @param rows Number of rows (must be > 0)
 * @param cols Number of columns (must be > 0)
 * @return Pointer to new tensor, or NULL on allocation failure
 */
PTensor tensor_create(int rows, int cols);

/**
 * Create a tensor initialized with values from array (row-major order).
 * @param rows Number of rows
 * @param cols Number of columns
 * @param vals Array of row-major values (rows*cols elements)
 * @return Pointer to new initialized tensor, or NULL on failure
 */
PTensor tensor_create_from_array(int rows, int cols, const real *vals);

/**
 * Initialize existing tensor with values from array.
 * @param t Tensor to initialize
 * @param rows Number of rows (should match t->rows)
 * @param cols Number of columns (should match t->cols)
 * @param array Array of values in row-major order
 */
void tensor_set_from_array(PTensor t, int rows, int cols, const real *array);

/**
 * Free tensor and all allocated memory.
 * Safe to call with NULL pointer.
 * @param t Tensor to free
 */
void tensor_free(PTensor t);

/**
 * Create a tensor filled with ones.
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new tensor filled with 1.0
 */
PTensor tensor_ones(int rows, int cols);

/**
 * Create a tensor filled with zeros.
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new tensor filled with 0.0
 */
PTensor tensor_zeros(int rows, int cols);

/**
 * Create a tensor with random uniform values.
 * @param rows Number of rows
 * @param cols Number of columns
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return Pointer to new tensor with random values
 */
PTensor tensor_create_random_uniform(int rows, int cols, real min, real max);

/**
 * Convert a tensor of class indices to one-hot encoding.
 * Input should be a column vector of integer class indices.
 * @param t Input tensor (rows x 1) containing class indices
 * @param classes Number of classes
 * @return Tensor of shape (rows x classes) with one-hot encoding
 */
PTensor tensor_onehot(const PTensor t, int classes);

/**
 * Create a deep copy of a tensor.
 * @param t Tensor to copy
 * @return Pointer to new tensor with same values
 */
PTensor tensor_copy(const PTensor t);

// ============================================================================
// ELEMENT-WISE AND SCALAR OPERATIONS
// ============================================================================

/**
 * Add scalar value to all tensor elements (in-place).
 * @param t Tensor to modify
 * @param val Scalar value to add
 * @return Pointer to modified tensor (same as t)
 */
PTensor tensor_add_scalar(PTensor t, real val);

/**
 * Add two tensors element-wise (in-place): a = a + b.
 * Tensors must have identical shape.
 * @param a First operand (modified in-place)
 * @param b Second operand (unchanged)
 * @return Pointer to modified tensor (same as a), or NULL on shape mismatch
 */
PTensor tensor_add(const PTensor a, const PTensor b);

/**
 * Subtract two tensors element-wise (in-place): a = a - b.
 * Tensors must have identical shape.
 * @param a First operand (modified in-place)
 * @param b Second operand (unchanged)
 * @return Pointer to modified tensor (same as a), or NULL on shape mismatch
 */
PTensor tensor_sub(const PTensor a, const PTensor b);

/**
 * Multiply tensor by scalar (in-place): t = t * alpha.
 * @param t Tensor to modify
 * @param val Scalar multiplier
 * @return Pointer to modified tensor (same as t)
 */
PTensor tensor_mul_scalar(PTensor t, real val);

/**
 * Element-wise multiply two tensors (in-place): a = a * b.
 * Tensors must have identical shape.
 * @param a First operand (modified in-place)
 * @param b Second operand (unchanged)
 * @return Pointer to modified tensor (same as a), or NULL on shape mismatch
 */
PTensor tensor_mul(const PTensor a, const PTensor b);

/**
 * Element-wise divide two tensors (in-place): a = a / b.
 * Tensors must have identical shape. Warning: no divide-by-zero protection.
 * @param a First operand (modified in-place)
 * @param b Second operand (unchanged)
 * @return Pointer to modified tensor (same as a), or NULL on shape mismatch
 */
PTensor tensor_div(const PTensor a, const PTensor b);

// ============================================================================
// MATRIX AND ADVANCED OPERATIONS
// ============================================================================

/**
 * Matrix-vector product: dest = alpha * A*x + beta * dest.
 * Implements: y = alpha * Ax + beta*y (with optional transpose)
 * @param trans Tensor_NoTranspose or Tensor_Transpose for matrix
 * @param alpha Scalar multiplier for product term
 * @param mtx Matrix (MxN when not transposed, NxM when transposed)
 * @param beta Scalar multiplier for destination term
 * @param v Vector (N elements)
 * @param dest Destination vector (M elements, modified in-place)
 * @return Pointer to modified dest tensor, or NULL on shape mismatch
 */
PTensor tensor_matvec(TENSOR_TRANSPOSE trans, real alpha, const PTensor mtx, real beta, const PTensor v, PTensor dest);

/**
 * Square each element (in-place): t = t * t.
 * @param t Tensor to modify
 * @return Pointer to modified tensor (same as t)
 */
PTensor tensor_square(PTensor t);

/**
 * Compute exponential of each element (in-place): t = e^t.
 * @param t Tensor to modify
 * @return Pointer to modified tensor (same as t)
 */
PTensor tensor_exp(PTensor t);

/**
 * Find index of maximum value in each column.
 * Returns a 1xN tensor containing column indices.
 * @param t Input tensor (MxN)
 * @return Tensor of shape 1xN with max indices, or NULL on error
 */
PTensor tensor_argmax(const PTensor t);

/**
 * Find maximum value in each column.
 * Returns a 1xN tensor containing column max values.
 * @param t Input tensor (MxN)
 * @return Tensor of shape 1xN with max values, or NULL on error
 */
PTensor tensor_max(const PTensor t);

/**
 * Sum all elements in a tensor (expects 1xN vector).
 * @param t Input tensor (should be 1xN row vector)
 * @return Sum of all elements
 */
real tensor_sum(const PTensor t);

/**
 * AXPY operation: y = alpha * x + y (in-place).
 * Tensors must have identical shape.
 * @param alpha Scalar multiplier for x
 * @param x Addend vector (unchanged)
 * @param y Destination vector (modified in-place)
 * @return Pointer to modified y, or NULL on shape mismatch
 */
PTensor tensor_axpy(real alpha, const PTensor x, PTensor y);

/**
 * General matrix multiplication: C = alpha * A*B + beta * C
 * Supports both BLAS (cblas_sgemm) and scalar fallback implementations.
 * @param alpha Scalar multiplier for product
 * @param A First matrix operand (M×K)
 * @param B Second matrix operand (K×N)
 * @param beta Scalar multiplier for C
 * @param C Result matrix (M×N, modified in-place)
 * @return Pointer to modified C, or NULL on dimension mismatch
 */
PTensor tensor_gemm(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);

/**
 * General matrix multiplication with B transposed.
 * Computes: C = alpha * A * B^T + beta * C
 * For batched forward pass: Y = X * W^T where X[batch×in], W[out×in], Y[batch×out]
 * @param alpha Scalar multiplier for product
 * @param A First matrix operand (M×K)
 * @param B Second matrix operand (N×K, will be transposed to K×N)
 * @param beta Scalar multiplier for C
 * @param C Result matrix (M×N, modified in-place)
 * @return Pointer to modified C, or NULL on error
 */
PTensor tensor_gemm_transB(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);

/**
 * General matrix multiplication with A transposed.
 * Computes: C = alpha * A^T * B + beta * C
 * For batched gradient computation: dW = delta^T * A
 * @param alpha Scalar multiplier for product
 * @param A First matrix operand (K×M, will be transposed to M×K)
 * @param B Second matrix operand (K×N)
 * @param beta Scalar multiplier for C
 * @param C Result matrix (M×N, modified in-place)
 * @return Pointer to modified C, or NULL on error
 */
PTensor tensor_gemm_transA(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);

/**
 * AXPBY operation: y = alpha * x + beta * y (in-place).
 * Tensors must have identical shape.
 * @param alpha Scalar multiplier for x
 * @param x First vector (unchanged)
 * @param beta Scalar multiplier for y
 * @param y Second vector (modified in-place)
 * @return Pointer to modified y, or NULL on shape mismatch
 */
PTensor tensor_axpby(real alpha, const PTensor x, real beta, PTensor y);

/**
 * Outer product: dest = alpha * a * b^T + dest (in-place).
 * For column vector a (Mx1) and row vector b (1xN), produces MxN result.
 * @param alpha Scalar multiplier
 * @param a Column vector (Mx1)
 * @param b Row vector (1xN)
 * @param dest Destination matrix MxN (modified in-place)
 * @return Pointer to modified dest, or NULL on shape mismatch
 */
PTensor tensor_outer(real alpha, const PTensor a, const PTensor b, PTensor dest);

/**
 * Heaviside step function (in-place): t = (t > 0) ? 1 : 0.
 * Used as ReLU derivative approximation.
 * @param a Tensor to modify
 * @return Pointer to modified tensor (same as a)
 */
PTensor tensor_heaviside(const PTensor a);

// ============================================================================
// ELEMENT ACCESS AND MANIPULATION
// ============================================================================

/**
 * Get single element from tensor at [row, col].
 * @param t Tensor to access
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return Element value, or 0.0 if indices out of bounds
 */
real tensor_get_element(const PTensor t, int row, int col);

/**
 * Set single element in tensor at [row, col].
 * @param t Tensor to modify
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @param val Value to set
 */
void tensor_set_element(PTensor t, int row, int col, real val);

/**
 * Create new tensor from specified rows (bottom rows of input).
 * Returns rows from index 'rows' to end of tensor.
 * @param t Input tensor
 * @param rows Number of rows to remove from top
 * @return New tensor containing rows [rows..t->rows)
 */
PTensor tensor_slice_rows(const PTensor t, int rows);

/**
 * Create new tensor from specified columns (rightmost columns of input).
 * Returns columns from index 'cols' to end of tensor.
 * @param t Input tensor
 * @param cols Number of columns to remove from left
 * @return New tensor containing cols [cols..t->cols)
 */
PTensor tensor_slice_cols(const PTensor t, int cols);

/**
 * Fill entire tensor with a constant value (in-place).
 * @param t Tensor to fill
 * @param val Value to fill with
 */
void tensor_fill(PTensor t, real val);

/**
 * Fill tensor with random uniform values (in-place).
 * @param t Tensor to fill
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 */
void tensor_random_uniform(PTensor t, real min, real max);

/**
 * Fill tensor with random normal (Gaussian) values (in-place).
 * Uses Box-Muller transform to generate normally distributed values.
 * @param t Tensor to fill
 * @param mean Mean of the distribution
 * @param std Standard deviation of the distribution
 */
void tensor_random_normal(PTensor t, real mean, real std);

/**
 * Clip tensor values to a specified range (in-place).
 * Values below min_val are set to min_val, values above max_val are set to max_val.
 * @param t Tensor to clip
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 */
void tensor_clip(PTensor t, real min_val, real max_val);

// ============================================================================
// INPUT/OUTPUT
// ============================================================================

/**
 * Print tensor to stdout in matrix format.
 * @param t Tensor to print
 */
void tensor_print(const PTensor t);

/**
 * Save tensor to CSV file (comma-separated values).
 * Each row on separate line, values comma-delimited.
 * @param t Tensor to save
 * @param filename Output file path
 * @return ERR_OK on success, error code on failure
 */
int tensor_save_to_file(const PTensor t, const char *filename);

#endif
