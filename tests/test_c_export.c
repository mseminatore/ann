//-----------------------------------------------------------
// Exported C headers, compiled on their own.
//
// Built without libann from the headers tests/gen_c_export.c writes, plus
// ann_rt.h, with strict C99 warnings. Checks that each int8 header gives
// exactly libann's int8 results (ann_quantized_predict_int8 and
// ann_quantized_predict) and that each float header matches ann_predict.
// Built a second time with ANN_RT_NO_FLOAT, which leaves only the int8 path.
//-----------------------------------------------------------

#include <stdio.h>
#include <string.h>

#include "dense_q.h"
#include "cnn_q.h"
#include "ae_q.h"
#ifndef ANN_RT_NO_FLOAT
#	include "dense_f.h"
#	include "cnn_f.h"
#	include "ae_f.h"
#endif
#include "expected.h"

static int g_failures;

static void check(const char *what, int ok)
{
	printf("\t%-60s %s\n", what, ok ? "ok" : "FAILED");
	if (!ok)
		g_failures++;
}

#ifndef ANN_RT_NO_FLOAT
static float absf(float v) { return v < 0.0f ? -v : v; }
#endif

int main(void)
{
	static int8_t ws_q[DENSE_Q_WORKSPACE_SIZE + CNN_Q_WORKSPACE_SIZE + AE_Q_WORKSPACE_SIZE];
	int8_t out_q[8];
	int same, argmax;
	int r;

#ifdef ANN_RT_NO_FLOAT
	printf("C header export, int8 only (ANN_RT_NO_FLOAT)\n");
#else
	printf("C header export, int8 and float\n");
#endif

	// int8 in and out: exactly libann's ann_quantized_predict_int8
	same = 1;
	argmax = 1;
	for (r = 0; r < ROWS; r++)
	{
		same &= dense_q_predict(dense_x_q[r], out_q, ws_q) == 0 &&
		        memcmp(out_q, dense_y_int8[r], DENSE_Q_OUTPUT_SIZE) == 0;
		argmax &= dense_q_argmax(out_q) == ann_rt_argmax_q(dense_y_int8[r], DENSE_Q_OUTPUT_SIZE);
	}
	check("dense int8 header matches libann bit for bit", same);
	check("dense int8 argmax", argmax);

	same = 1;
	for (r = 0; r < ROWS; r++)
		same &= cnn_q_predict(cnn_x_q[r], out_q, ws_q) == 0 &&
		        memcmp(out_q, cnn_y_int8[r], CNN_Q_OUTPUT_SIZE) == 0;
	check("CNN int8 header matches libann bit for bit", same);

	same = 1;
	for (r = 0; r < ROWS; r++)
		same &= ae_q_predict(ae_x_q[r], out_q, ws_q) == 0 &&
		        memcmp(out_q, ae_y_int8[r], AE_Q_OUTPUT_SIZE) == 0;
	check("reshape/upsample int8 header matches libann bit for bit", same);

#ifndef ANN_RT_NO_FLOAT
	{
		static float ws_f[DENSE_F_WORKSPACE_SIZE + CNN_F_WORKSPACE_SIZE + AE_F_WORKSPACE_SIZE];
		float out_f[8], worst;
		int k;

		// float in and out through the int8 model: exactly ann_quantized_predict
		same = 1;
		for (r = 0; r < ROWS; r++)
			same &= dense_q_predict_float(dense_x[r], out_f, ws_q) == 0 &&
			        memcmp(out_f, dense_y_q[r], sizeof(float) * DENSE_Q_OUTPUT_SIZE) == 0;
		check("dense int8 header with float I/O matches libann exactly", same);

		same = 1;
		for (r = 0; r < ROWS; r++)
			same &= cnn_q_predict_float(cnn_x[r], out_f, ws_q) == 0 &&
			        memcmp(out_f, cnn_y_q[r], sizeof(float) * CNN_Q_OUTPUT_SIZE) == 0;
		check("CNN int8 header with float I/O matches libann exactly", same);

		same = 1;
		for (r = 0; r < ROWS; r++)
			same &= ae_q_predict_float(ae_x[r], out_f, ws_q) == 0 &&
			        memcmp(out_f, ae_y_q[r], sizeof(float) * AE_Q_OUTPUT_SIZE) == 0;
		check("reshape/upsample int8 header with float I/O matches libann exactly", same);

		// the float headers: ann_predict up to float summation order
		worst = 0.0f;
		argmax = 1;
		for (r = 0; r < ROWS; r++)
		{
			dense_f_predict(dense_x[r], out_f, ws_f);
			for (k = 0; k < DENSE_F_OUTPUT_SIZE; k++)
				if (absf(out_f[k] - dense_y_float[r][k]) > worst)
					worst = absf(out_f[k] - dense_y_float[r][k]);
			argmax &= dense_f_argmax(out_f) == ann_rt_argmax_f(dense_y_float[r], DENSE_F_OUTPUT_SIZE);
		}
		check("dense float header matches ann_predict (1e-5)", worst <= 1e-5f);
		check("dense float argmax", argmax);

		worst = 0.0f;
		for (r = 0; r < ROWS; r++)
		{
			cnn_f_predict(cnn_x[r], out_f, ws_f);
			for (k = 0; k < CNN_F_OUTPUT_SIZE; k++)
				if (absf(out_f[k] - cnn_y_float[r][k]) > worst)
					worst = absf(out_f[k] - cnn_y_float[r][k]);
		}
		check("CNN float header matches ann_predict (1e-5)", worst <= 1e-5f);

		worst = 0.0f;
		for (r = 0; r < ROWS; r++)
		{
			ae_f_predict(ae_x[r], out_f, ws_f);
			for (k = 0; k < AE_F_OUTPUT_SIZE; k++)
				if (absf(out_f[k] - ae_y_float[r][k]) > worst)
					worst = absf(out_f[k] - ae_y_float[r][k]);
		}
		check("reshape/upsample float header matches ann_predict (1e-5)", worst <= 1e-5f);
	}
#endif

	printf("%d failure(s)\n", g_failures);
	return g_failures ? 1 : 0;
}
