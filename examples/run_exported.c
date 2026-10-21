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

//------------------------------------------------------------------------------
// Run the C headers written by export_model on the Fashion-MNIST test set.
//
// Companion to examples/export_model.c. Built without libann: it needs only
// fashion_q.h, fashion_f.h and ann_rt.h on the include path. Reads the test
// CSV, runs every image through the int8 header (int8 in and out) and the
// float32 header, and prints each one's accuracy and speed. The accuracies
// should match the ones export_model printed.
//
// Usage: run_exported [test.csv]
//------------------------------------------------------------------------------

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#	define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fashion_q.h"
#include "fashion_f.h"

#define LINE_SIZE 8192

static float input_f[FASHION_F_INPUT_SIZE];
static int8_t input_q[FASHION_Q_INPUT_SIZE];
static float ws_f[FASHION_F_WORKSPACE_SIZE];
static int8_t ws_q[FASHION_Q_WORKSPACE_SIZE];
static char line[LINE_SIZE];

//----------------------------------
// parse "label,p0,p1,..." into the label and pixels scaled to [0, 1],
// the same preprocessing export_model trained and calibrated with
//----------------------------------
static int parse_row(char *s, int *label)
{
	char *end;
	int i;

	*label = (int)strtol(s, &end, 10);
	if (end == s)
		return 0;

	for (i = 0; i < FASHION_F_INPUT_SIZE; i++)
	{
		if (*end != ',')
			return 0;
		s = end + 1;
		input_f[i] = (float)strtol(s, &end, 10) / 255.0f;
		if (end == s)
			return 0;
	}
	return 1;
}

int main(int argc, char *argv[])
{
	const char *test_file = argc > 1 ? argv[1] : "fashion-mnist_test.csv";
	long rows = 0, correct_q = 0, correct_f = 0;
	clock_t ticks_q = 0, ticks_f = 0, t;
	int label, i;
	FILE *f;

	if (FASHION_Q_INPUT_SIZE != FASHION_F_INPUT_SIZE)
	{
		printf("Error: fashion_q.h and fashion_f.h have different input sizes\n");
		return 1;
	}

	f = fopen(test_file, "r");
	if (!f)
	{
		printf("Error: unable to open data file - %s\n", test_file);
		return 1;
	}

	// skip the header row
	if (!fgets(line, LINE_SIZE, f))
	{
		fclose(f);
		return 1;
	}

	while (fgets(line, LINE_SIZE, f))
	{
		if (!parse_row(line, &label))
			continue;

		// int8: quantize the input with the header's input parameters
		{
			int8_t out[FASHION_Q_OUTPUT_SIZE];

			for (i = 0; i < FASHION_Q_INPUT_SIZE; i++)
				input_q[i] = ann_rt_quantize(input_f[i], FASHION_Q_INPUT_SCALE, FASHION_Q_INPUT_ZERO_POINT);

			t = clock();
			if (fashion_q_predict(input_q, out, ws_q) != 0)
			{
				printf("Error: fashion_q_predict failed\n");
				fclose(f);
				return 1;
			}
			correct_q += fashion_q_argmax(out) == label;
			ticks_q += clock() - t;
		}

		// float32
		{
			float out[FASHION_F_OUTPUT_SIZE];

			t = clock();
			if (fashion_f_predict(input_f, out, ws_f) != 0)
			{
				printf("Error: fashion_f_predict failed\n");
				fclose(f);
				return 1;
			}
			correct_f += fashion_f_argmax(out) == label;
			ticks_f += clock() - t;
		}

		rows++;
	}
	fclose(f);

	if (rows == 0)
	{
		printf("Error: no test rows read from %s\n", test_file);
		return 1;
	}

	printf("%ld test images from %s\n", rows, test_file);
	printf("  int8    (fashion_q.h): accuracy %.2f%%, %.1f us/image\n",
	       100.0 * correct_q / rows, 1e6 * ticks_q / CLOCKS_PER_SEC / rows);
	printf("  float32 (fashion_f.h): accuracy %.2f%%, %.1f us/image\n",
	       100.0 * correct_f / rows, 1e6 * ticks_f / CLOCKS_PER_SEC / rows);
	printf("  int8 - float32: %+.2f points\n", 100.0 * (correct_q - correct_f) / rows);
	return 0;
}
