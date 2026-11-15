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
#ifndef ANN_LOWBIT_H
#define ANN_LOWBIT_H

//================================================================================================
// Low-bit weight rounding (internal), shared by training (ann.c), which computes with the
// rounded weights, and quantization (ann_quantize.c), which packs the same integers, so an
// exported model holds exactly the weights libann trained and predicted with.
//
// Each row of a weight matrix (one output channel) gets a scale s and integers q with
// w ~ s * q (see Weight_format in ann.h). Everything is computed in double from the row's
// values, the same way on both sides.
//================================================================================================

#include <math.h>
#include "ann.h"

typedef struct {
	double scale;		// s: the row's weights are s * q
	double threshold;	// ternary: |w| above it is kept
} Lowbit_row;

#if defined(_MSC_VER) && !defined(__cplusplus)
#	define ANN_LOWBIT_FUNC static __inline
#else
#	define ANN_LOWBIT_FUNC static inline
#endif

//------------------------------
// bits a weight takes in the format (8 for WEIGHT_FORMAT_FLOAT, which quantizes to int8)
//------------------------------
ANN_LOWBIT_FUNC int lowbit_bits(Weight_format format)
{
	switch (format)
	{
	case WEIGHT_FORMAT_INT4:	return 4;
	case WEIGHT_FORMAT_TERNARY:	return 2;
	case WEIGHT_FORMAT_BINARY:	return 1;
	default:					return 8;
	}
}

//------------------------------
// the scale (and ternary threshold) of a row of n weights. A row of zeros gets scale 0,
// so its weights stay 0 (binary ones too).
//------------------------------
ANN_LOWBIT_FUNC Lowbit_row lowbit_row(Weight_format format, const real *w, int n)
{
	Lowbit_row r = { 0.0, 0.0 };
	double sum = 0.0, max_abs = 0.0;
	int i;

	for (i = 0; i < n; i++)
	{
		double a = fabs((double)w[i]);
		sum += a;
		if (a > max_abs)
			max_abs = a;
	}

	switch (format)
	{
	case WEIGHT_FORMAT_INT4:
		r.scale = max_abs / 7.0;
		break;

	case WEIGHT_FORMAT_TERNARY:
		{
			double kept = 0.0;
			int count = 0;

			r.threshold = n > 0 ? 0.7 * sum / n : 0.0;
			for (i = 0; i < n; i++)
			{
				double a = fabs((double)w[i]);
				if (a > r.threshold)
				{
					kept += a;
					count++;
				}
			}
			r.scale = count > 0 ? kept / count : 0.0;
		}
		break;

	case WEIGHT_FORMAT_BINARY:
		r.scale = n > 0 ? sum / n : 0.0;
		break;

	default:
		break;
	}
	return r;
}

//------------------------------
// a weight's integer in its row (see lowbit_row): -7..7, -1..1, or -1/+1
//------------------------------
ANN_LOWBIT_FUNC int lowbit_code(Weight_format format, Lowbit_row r, real w)
{
	double x = (double)w;

	switch (format)
	{
	case WEIGHT_FORMAT_INT4:
		{
			double q;
			if (r.scale <= 0.0)
				return 0;
			q = floor(x / r.scale + 0.5);
			return q > 7.0 ? 7 : q < -7.0 ? -7 : (int)q;
		}

	case WEIGHT_FORMAT_TERNARY:
		return fabs(x) > r.threshold ? (x > 0.0 ? 1 : -1) : 0;

	case WEIGHT_FORMAT_BINARY:
		return x >= 0.0 ? 1 : -1;

	default:
		return 0;
	}
}

#endif /* ANN_LOWBIT_H */
