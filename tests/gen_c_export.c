//-----------------------------------------------------------
// Generator for tests/test_c_export.c (run by the build).
//
// Trains a small dense network and a small CNN, quantizes them, and writes
// into the output directory:
//   dense_q.h, cnn_q.h, ae_q.h, gap_q.h   int8 headers (ann_export_c_header with the int8 model)
//   dense_f.h, cnn_f.h, ae_f.h, gap_f.h   float32 headers (ann_export_c_header without it)
//   expected.h           test inputs and libann's outputs for them:
//                        ann_quantized_predict_int8, ann_quantized_predict and
//                        ann_predict
// test_c_export.c compiles the headers without libann and checks they give
// the same results.
//
// Usage: gen_c_export <output directory>
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"

#define ROWS 16		// test rows per model
#define TRAIN 128	// training rows per model

static void quiet(const char *s) { (void)s; }

static void write_float(FILE *f, double v)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%.9g", (float)v);
	fprintf(f, "%s%sf", buf, strpbrk(buf, ".e") ? "" : ".0");
}

static void write_reals(FILE *f, const char *name, const real *v, int rows, int cols)
{
	fprintf(f, "static const float %s[%d][%d] = {\n", name, rows, cols);
	for (int r = 0; r < rows; r++)
	{
		fprintf(f, "\t{ ");
		for (int c = 0; c < cols; c++)
		{
			write_float(f, v[r * cols + c]);
			fprintf(f, c + 1 < cols ? ", " : " },\n");
		}
	}
	fprintf(f, "};\n");
}

static void write_int8s(FILE *f, const char *name, const signed char *v, int rows, int cols)
{
	fprintf(f, "static const int8_t %s[%d][%d] = {\n", name, rows, cols);
	for (int r = 0; r < rows; r++)
	{
		fprintf(f, "\t{ ");
		for (int c = 0; c < cols; c++)
			fprintf(f, "%d%s", v[r * cols + c], c + 1 < cols ? ", " : " },\n");
	}
	fprintf(f, "};\n");
}

// a target class from the inputs, so there is something to learn
static void make_data(real *x, real *y, int rows, int in, int out, int seed)
{
	for (int r = 0; r < rows; r++)
	{
		double best = -1e9;
		int cls = 0;
		for (int i = 0; i < in; i++)
			x[r * in + i] = (real)(0.5 + 0.5 * sin(0.37 * (r * in + i) + seed));
		for (int k = 0; k < out; k++)
		{
			double score = 0.0;
			for (int i = k; i < in; i += out)
				score += x[r * in + i];
			if (score > best) { best = score; cls = k; }
			y[r * out + k] = 0.0f;
		}
		y[r * out + cls] = 1.0f;
	}
}

// train, quantize and export one model, and write its expected outputs
static int export_model(PNetwork net, const char *dir, const char *name, FILE *expected, int seed)
{
	int in = ann_get_layer_nodes(net, 0), out = ann_get_layer_nodes(net, ann_get_layer_count(net) - 1);
	real *x = malloc(TRAIN * in * sizeof(real)), *y = malloc(TRAIN * out * sizeof(real));
	real *yq = malloc(ROWS * out * sizeof(real)), *yf = malloc(ROWS * out * sizeof(real));
	signed char *xi = malloc(ROWS * in), *yi = malloc(ROWS * out);
	char path[1024], prefix[64], array[64];
	real in_scale;
	int in_zp, err = ERR_OK;

	make_data(x, y, TRAIN, in, out, seed);
	PTensor tx = tensor_create_from_array(TRAIN, in, x);
	PTensor ty = tensor_create_from_array(TRAIN, out, y);

	net->print_func = quiet;
	ann_set_convergence(net, 0.0f);
	ann_set_epoch_limit(net, 20);
	ann_train_network(net, tx, ty, TRAIN);

	PQuantizedNetwork q = ann_quantize(net, tx);
	if (!q)
		return ERR_FAIL;

	snprintf(prefix, sizeof(prefix), "%s_q", name);
	snprintf(path, sizeof(path), "%s/%s.h", dir, prefix);
	err |= ann_export_c_header(NULL, q, path, prefix);
	snprintf(prefix, sizeof(prefix), "%s_f", name);
	snprintf(path, sizeof(path), "%s/%s.h", dir, prefix);
	err |= ann_export_c_header(net, NULL, path, prefix);

	// the test rows are the first training rows, quantized like the device would
	ann_quantized_get_io_params(q, &in_scale, &in_zp, NULL, NULL);
	for (int i = 0; i < ROWS * in; i++)
	{
		double v = floor(x[i] / (float)in_scale + 0.5) + in_zp;
		xi[i] = (signed char)(v < -128 ? -128 : v > 127 ? 127 : v);
	}
	for (int r = 0; r < ROWS; r++)
	{
		err |= ann_quantized_predict_int8(q, &xi[r * in], &yi[r * out]);
		err |= ann_quantized_predict(q, &x[r * in], &yq[r * out]);
		err |= ann_predict(net, &x[r * in], &yf[r * out]);
	}

	snprintf(array, sizeof(array), "%s_x", name);
	write_reals(expected, array, x, ROWS, in);
	snprintf(array, sizeof(array), "%s_x_q", name);
	write_int8s(expected, array, xi, ROWS, in);
	snprintf(array, sizeof(array), "%s_y_int8", name);
	write_int8s(expected, array, yi, ROWS, out);
	snprintf(array, sizeof(array), "%s_y_q", name);
	write_reals(expected, array, yq, ROWS, out);
	snprintf(array, sizeof(array), "%s_y_float", name);
	write_reals(expected, array, yf, ROWS, out);

	ann_free_quantized(q);
	tensor_free(tx);
	tensor_free(ty);
	free(x); free(y); free(yq); free(yf); free(xi); free(yi);
	return err;
}

int main(int argc, char *argv[])
{
	char path[1024];
	int err = ERR_OK;

	if (argc != 2)
	{
		fprintf(stderr, "usage: gen_c_export <output directory>\n");
		return 1;
	}

	snprintf(path, sizeof(path), "%s/expected.h", argv[1]);
	FILE *expected = fopen(path, "w");
	if (!expected)
	{
		fprintf(stderr, "gen_c_export: cannot write %s\n", path);
		return 1;
	}
	fprintf(expected, "/* generated by gen_c_export for test_c_export.c */\n#include <stdint.h>\n");
	fprintf(expected, "#define ROWS %d\n", ROWS);

	srand(5);

	// dense: ReLU, GELU (lookup table) and softmax
	PNetwork dense = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
	ann_add_layer(dense, 12, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(dense, 16, LAYER_HIDDEN, ACTIVATION_RELU);
	ann_add_layer(dense, 8, LAYER_HIDDEN, ACTIVATION_GELU);
	ann_add_layer(dense, 4, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
	err |= export_model(dense, argv[1], "dense", expected, 1);
	ann_free_network(dense);

	// CNN: padded ReLU conv, pooling, strided sigmoid conv, flatten, tanh dense output
	PNetwork cnn = ann_make_network(OPT_ADAM, LOSS_MSE);
	ann_add_input(cnn, 2, 8, 8);
	ann_add_conv2d(cnn, 4, 3, 1, 1, ACTIVATION_RELU);
	ann_add_maxpool2d(cnn, 2, 2);
	ann_add_conv2d(cnn, 5, 3, 2, 1, ACTIVATION_SIGMOID);
	ann_add_flatten(cnn);
	ann_add_layer(cnn, 3, LAYER_OUTPUT, ACTIVATION_TANH);
	err |= export_model(cnn, argv[1], "cnn", expected, 2);
	ann_free_network(cnn);

	// decoder: dense bottleneck, reshape, upsampling around a pool, strided tanh conv
	PNetwork ae = ann_make_network(OPT_ADAM, LOSS_MSE);
	ann_add_layer(ae, 12, LAYER_INPUT, ACTIVATION_NULL);
	ann_add_layer(ae, 8, LAYER_HIDDEN, ACTIVATION_RELU);
	ann_add_reshape(ae, 2, 2, 2);
	ann_add_upsample2d(ae, 2);
	ann_add_conv2d(ae, 3, 3, 1, 1, ACTIVATION_SIGMOID);
	ann_add_maxpool2d(ae, 2, 2);
	ann_add_upsample2d(ae, 2);
	ann_add_conv2d(ae, 2, 3, 2, 1, ACTIVATION_TANH);
	err |= export_model(ae, argv[1], "ae", expected, 3);
	ann_free_network(ae);

	// global average pooling in place of flatten: two convs, pooled, softmax dense output
	PNetwork gap = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
	ann_add_input(gap, 2, 7, 7);
	ann_add_conv2d(gap, 4, 3, 1, 1, ACTIVATION_RELU);
	ann_add_conv2d(gap, 6, 3, 2, 0, ACTIVATION_TANH);
	ann_add_global_avgpool2d(gap);
	ann_add_layer(gap, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
	err |= export_model(gap, argv[1], "gap", expected, 4);
	ann_free_network(gap);

	if (fclose(expected) != 0 || err != ERR_OK)
	{
		fprintf(stderr, "gen_c_export: failed\n");
		return 1;
	}
	return 0;
}
