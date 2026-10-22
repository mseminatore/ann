//-----------------------------------------------------------
// int8 quantization (ann_quantize.c) and the ann_rt.h runtime.
//
// Fixed-point rescaling and quantize/dequantize against float math, every
// activation and layer kind against the float network (within a few int8
// steps of the output scale), trained models keeping their accuracy, the
// int8 entry points agreeing with each other, header export, and argument
// errors. tests/test_c_export.c compiles the exported headers on their own.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "ann_rt.h"
#include "testy/test.h"

#define N_CAL 64
#define MAX_IN 72
#define MAX_OUT 8

static void quiet(const char *s) { (void)s; }

// swallow the error messages the argument checks provoke
static int g_errors;
static void count_errors(int code, const char *msg, const char *fn) { (void)code; (void)msg; (void)fn; g_errors++; }

// 1 if the file contains the text
static int file_contains(const char *path, const char *needle)
{
    static char buf[1 << 18];
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

// Fill every parameter with fixed, varied values
static void set_params(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++)
                t->values[i] = 0.5f * (real)sin(1.7 * i + 0.9 * l + 2.3 * p + 0.4);
        }
}

// N_CAL input rows in [-1, 1]
static PTensor make_inputs(int cols)
{
    PTensor t = tensor_zeros(N_CAL, cols);
    for (int i = 0; i < N_CAL * cols; i++)
        t->values[i] = (real)sin(0.77 * i + 0.3);
    return t;
}

// Largest difference between the float and int8 predictions over the rows,
// in units of the int8 output scale (or absolute, when absolute is set)
static double max_error(PNetwork net, PQuantizedNetwork q, PTensor x, int absolute)
{
    int out = net->layers[net->layer_count - 1].node_count;
    real yf[MAX_OUT], yq[MAX_OUT], scale;
    double worst = 0.0;

    ann_quantized_get_io_params(q, NULL, NULL, &scale, NULL);
    for (int r = 0; r < x->rows; r++) {
        ann_predict(net, &x->values[r * x->cols], yf);
        ann_quantized_predict(q, &x->values[r * x->cols], yq);
        for (int k = 0; k < out; k++) {
            double e = fabs(yf[k] - yq[k]) / (absolute ? 1.0 : scale);
            if (e > worst) worst = e;
        }
    }
    return worst;
}

// in -> dense(6, act) -> dense(out, out_act)
static PNetwork dense_net(int in, Activation_type act, int out, Activation_type out_act)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    net->print_func = quiet;
    ann_add_layer(net, in, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 6, LAYER_HIDDEN, act);
    ann_add_layer(net, out, LAYER_OUTPUT, out_act);
    set_params(net);
    return net;
}

// 1x6x6 images of a horizontal (class 0) or vertical (class 1) bar
static void make_bars(real *x, real *y, int count)
{
    for (int n = 0; n < count; n++) {
        int vertical = n % 2, pos = (n / 2) % 6;
        for (int i = 0; i < 36; i++)
            x[n * 36 + i] = 0.05f * (real)sin(3.1 * i + n);
        for (int k = 0; k < 6; k++)
            x[n * 36 + (vertical ? k * 6 + pos : pos * 6 + k)] = 1.0f;
        y[n * 2 + 0] = (real)!vertical;
        y[n * 2 + 1] = (real)vertical;
    }
}

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    MODULE("int8 Quantization and C Export");

    SUITE("Runtime Arithmetic");
    {
        // ann_rt_requantize(acc, mult, shift) == round(acc * mult * 2^-shift)
        static const double ms[] = { 0.9, 0.5, 0.123456, 3.7e-3, 1.0e-5, 1.5, 0.75 };
        static const int32_t accs[] = { 0, 1, -1, 7, -7, 1000, -1000, 123456, -123456, 2000000, -2000000 };
        int exact = 1;
        for (size_t a = 0; a < sizeof(ms) / sizeof(ms[0]); a++) {
            int e;
            double frac = frexp(ms[a], &e);
            int32_t mult = (int32_t)floor(frac * 2147483648.0 + 0.5);
            int32_t shift = 31 - e;
            double m = mult * pow(2.0, -shift);    // the multiplier as stored
            for (size_t b = 0; b < sizeof(accs) / sizeof(accs[0]); b++) {
                double want = floor(accs[b] * m + 0.5);
                if (want > 127.0) want = 127.0;
                if (want < -128.0) want = -128.0;
                if (ann_rt_requantize(accs[b], mult, shift, 0, -128) != (int8_t)want)
                    exact = 0;
            }
        }
        TESTEX("Fixed-point rescaling rounds like float math", exact);
        TESTEX("Rescaling adds the zero point and clamps",
               ann_rt_requantize(100, 1073741824, 31, 10, -128) == 60 &&
               ann_rt_requantize(1000, 1073741824, 31, 0, -128) == 127 &&
               ann_rt_requantize(-1000, 1073741824, 31, 0, -128) == -128 &&
               ann_rt_requantize(-100, 1073741824, 31, -20, -20) == -20);

        int round_trip = 1;
        const float scale = 2.0f / 255.0f;
        const int32_t zp = -1;
        for (int i = -1000; i <= 1000; i++) {
            float x = i / 1000.0f;
            if (fabsf(ann_rt_dequantize(ann_rt_quantize(x, scale, zp), scale, zp) - x) > scale * 0.5001f)
                round_trip = 0;
        }
        TESTEX("Quantize then dequantize is within half a step", round_trip);
        TESTEX("Quantize clamps to int8",
               ann_rt_quantize(10.0f, scale, zp) == 127 && ann_rt_quantize(-10.0f, scale, zp) == -128);
        int8_t v[5] = { -3, 9, 9, -128, 4 };
        TESTEX("argmax takes the first largest", ann_rt_argmax_q(v, 5) == 1);
    }

    SUITE("Activations and Layers");
    {
        static const struct { Activation_type act; const char *name; } acts[] = {
            { ACTIVATION_NULL, "none" }, { ACTIVATION_RELU, "ReLU" }, { ACTIVATION_LEAKY_RELU, "leaky ReLU" },
            { ACTIVATION_SIGMOID, "sigmoid" }, { ACTIVATION_TANH, "tanh" }, { ACTIVATION_SOFTSIGN, "softsign" },
            { ACTIVATION_GELU, "GELU" },
        };
        PTensor x = make_inputs(8);

        for (size_t a = 0; a < sizeof(acts) / sizeof(acts[0]); a++) {
            // the activation on the hidden layer and on the output layer
            PNetwork net = dense_net(8, acts[a].act, 4, acts[a].act);
            PQuantizedNetwork q = ann_quantize(net, x);
            double err = q ? max_error(net, q, x, 0) : 1e9;
            printf("\t  dense %s: %.2f steps\n", acts[a].name, err);
            TESTEX("Dense layers within 3 output steps of float", err <= 3.0);
            ann_free_quantized(q);
            ann_free_network(net);
        }

        PNetwork net = dense_net(8, ACTIVATION_RELU, 4, ACTIVATION_SOFTMAX);
        PQuantizedNetwork q = ann_quantize(net, x);
        double err = q ? max_error(net, q, x, 1) : 1e9;
        printf("\t  softmax: %.4f\n", err);
        TESTEX("Softmax probabilities within 0.02 of float", err <= 0.02);

        // the int8 entry point matches the float one, and the model outlives the network
        real in_scale, out_scale, yq[MAX_OUT];
        int in_zp, out_zp, same = 1;
        signed char xi[MAX_IN], yi[MAX_OUT];
        ann_quantized_get_io_params(q, &in_scale, &in_zp, &out_scale, &out_zp);
        TESTEX("Input scale covers [-1, 1]", fabs(in_scale - 2.0 / 255.0) < 1e-3 && in_zp >= -2 && in_zp <= 1);
        ann_free_network(net);
        for (int r = 0; r < x->rows; r++) {
            const real *row = &x->values[r * 8];
            double sum = 0.0, p[MAX_OUT];
            for (int i = 0; i < 8; i++)
                xi[i] = (signed char)ann_rt_quantize((float)row[i], (float)in_scale, in_zp);
            ann_quantized_predict_int8(q, xi, yi);
            ann_quantized_predict(q, row, yq);
            for (int k = 0; k < 4; k++)
                sum += p[k] = exp(ann_rt_dequantize(yi[k], (float)out_scale, out_zp));
            for (int k = 0; k < 4; k++)
                if (fabs(p[k] / sum - yq[k]) > 1e-5)
                    same = 0;
        }
        TESTEX("int8 and float entry points agree (after the network is freed)", same);
        ann_free_quantized(q);
        tensor_free(x);

        // conv with stride and padding, then GELU conv, then dense
        x = make_inputs(36);
        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 1, 6, 6);
        ann_add_conv2d(net, 2, 3, 2, 1, ACTIVATION_SIGMOID);
        ann_add_conv2d(net, 3, 2, 1, 0, ACTIVATION_GELU);
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
        set_params(net);
        q = ann_quantize(net, x);
        err = q ? max_error(net, q, x, 0) : 1e9;
        printf("\t  strided conv: %.2f steps\n", err);
        TESTEX("Conv (stride 2, padding 1) and GELU conv within 3 steps", err <= 3.0);
        ann_free_quantized(q);
        ann_free_network(net);

        // padded ReLU conv, max pooling with overlapping windows, flatten, tanh dense
        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 1, 6, 6);
        ann_add_conv2d(net, 3, 3, 1, 1, ACTIVATION_RELU);
        ann_add_maxpool2d(net, 3, 2);
        ann_add_flatten(net);
        ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_TANH);
        set_params(net);
        q = ann_quantize(net, x);
        err = q ? max_error(net, q, x, 0) : 1e9;
        // three rounding stages (conv, pooled values, then a sum over 12 of them)
        // into a narrow tanh output range: about 1.2 steps on average
        printf("\t  conv/pool: %.2f steps\n", err);
        TESTEX("ReLU conv, max pool and flatten within 5 steps", err <= 5.0);

        // the network is unchanged by quantizing
        real before[MAX_OUT], after[MAX_OUT];
        ann_predict(net, x->values, before);
        PQuantizedNetwork q2 = ann_quantize(net, x);
        ann_predict(net, x->values, after);
        TESTEX("Quantizing leaves the network unchanged", memcmp(before, after, 3 * sizeof(real)) == 0);
        ann_free_quantized(q2);
        ann_free_quantized(q);
        ann_free_network(net);

        // a decoder: reshape a dense layer to an image, upsample, convolve
        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 1, 6, 6);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_TANH);
        ann_add_reshape(net, 2, 2, 2);
        ann_add_upsample2d(net, 3);
        ann_add_conv2d(net, 2, 3, 3, 0, ACTIVATION_SIGMOID);
        set_params(net);
        q = ann_quantize(net, x);
        err = q ? max_error(net, q, x, 0) : 1e9;
        printf("\t  reshape/upsample: %.2f steps\n", err);
        TESTEX("Reshape, upsample and conv within 3 steps", err <= 3.0);
        TESTEX("Its header needs ann_rt.h version 2",
               ann_export_c_header(NULL, q, "test_quantize_up.h", "up_q") == ERR_OK &&
               file_contains("test_quantize_up.h", "ANN_RT_VERSION < 2") &&
               file_contains("test_quantize_up.h", "ANN_RT_UPSAMPLE2D"));
        remove("test_quantize_up.h");
        ann_free_quantized(q);
        ann_free_network(net);

        tensor_free(x);
    }

    SUITE("Trained Models");
    {
        // XOR, two softmax outputs
        real xor_x[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
        real xor_y[] = { 1, 0, 0, 1, 0, 1, 1, 0 };
        PTensor tx = tensor_create_from_array(4, 2, xor_x);
        PTensor ty = tensor_create_from_array(4, 2, xor_y);
        srand(1);
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        net->print_func = quiet;
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_TANH);
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
        ann_set_learning_rate(net, 0.05f);
        ann_set_convergence(net, 0.0f);
        ann_set_batch_size(net, 4);
        ann_set_epoch_limit(net, 500);
        ann_train_network(net, tx, ty, 4);
        PQuantizedNetwork q = ann_quantize(net, tx);
        TESTEX("XOR: float accuracy is 100%%", ann_evaluate_accuracy(net, tx, ty) == 1.0f);
        TESTEX("XOR: int8 accuracy is 100%%", q && ann_quantized_evaluate_accuracy(q, tx, ty) == 1.0f);
        ann_free_quantized(q);
        ann_free_network(net);
        tensor_free(tx);
        tensor_free(ty);

        // bars CNN
        enum { N = 24 };
        real x[N * 36], y[N * 2];
        make_bars(x, y, N);
        tx = tensor_create_from_array(N, 36, x);
        ty = tensor_create_from_array(N, 2, y);
        srand(42);
        net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        net->print_func = quiet;
        ann_add_input(net, 1, 6, 6);
        ann_add_conv2d(net, 4, 3, 1, 1, ACTIVATION_RELU);
        ann_add_maxpool2d(net, 2, 2);
        ann_add_flatten(net);
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
        ann_set_learning_rate(net, 0.01f);
        ann_set_convergence(net, 0.0f);
        ann_set_batch_size(net, 8);
        ann_set_epoch_limit(net, 60);
        ann_train_network(net, tx, ty, N);
        q = ann_quantize(net, tx);
        TESTEX("Bars CNN: float accuracy is 100%%", ann_evaluate_accuracy(net, tx, ty) == 1.0f);
        TESTEX("Bars CNN: int8 accuracy is 100%%", q && ann_quantized_evaluate_accuracy(q, tx, ty) == 1.0f);

        // header export
        TESTEX("int8 header exports", ann_export_c_header(NULL, q, "test_quantize_q.h", "bars_q") == ERR_OK);
        TESTEX("with sizes, quantization and int8 arrays",
               file_contains("test_quantize_q.h", "#define BARS_Q_INPUT_SIZE 36") &&
               file_contains("test_quantize_q.h", "#define BARS_Q_OUTPUT_SIZE 2") &&
               file_contains("test_quantize_q.h", "#define BARS_Q_INPUT_ZERO_POINT") &&
               file_contains("test_quantize_q.h", "static const int8_t bars_q_w0[36]") &&
               file_contains("test_quantize_q.h", "ANN_RT_FUNC int bars_q_predict(const int8_t *input") &&
               file_contains("test_quantize_q.h", "ANN_RT_VERSION < 1"));
        TESTEX("float header exports", ann_export_c_header(net, NULL, "test_quantize_f.h", "bars_f") == ERR_OK);
        TESTEX("with float arrays",
               file_contains("test_quantize_f.h", "static const float bars_f_w0[36]") &&
               file_contains("test_quantize_f.h", "ANN_RT_FUNC int bars_f_predict(const float *input") &&
               !file_contains("test_quantize_f.h", "int8_t bars_f"));
        remove("test_quantize_q.h");
        remove("test_quantize_f.h");

        ann_free_quantized(q);
        ann_free_network(net);
        tensor_free(tx);
        tensor_free(ty);
    }

    SUITE("Argument Errors");
    {
        ann_set_error_log_callback(count_errors);
        PTensor x = make_inputs(8);
        PTensor wrong = make_inputs(5);
        PNetwork net = dense_net(8, ACTIVATION_RELU, 2, ACTIVATION_NULL);
        PQuantizedNetwork q = ann_quantize(net, x);
        real out[MAX_OUT];

        TESTEX("NULL network or calibration data", !ann_quantize(NULL, x) && !ann_quantize(net, NULL));
        TESTEX("Calibration columns must match the inputs", !ann_quantize(net, wrong));

        PNetwork only_input = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(only_input, 8, LAYER_INPUT, ACTIVATION_NULL);
        TESTEX("Needs a layer after the input", !ann_quantize(only_input, x));
        ann_free_network(only_input);

        PNetwork hidden_softmax = dense_net(8, ACTIVATION_SOFTMAX, 2, ACTIVATION_NULL);
        TESTEX("Softmax only on the output layer", !ann_quantize(hidden_softmax, x) &&
               ann_export_c_header(hidden_softmax, NULL, "test_quantize_x.h", "m") == ERR_INVALID);
        ann_free_network(hidden_softmax);

        TESTEX("Predict needs all pointers", ann_quantized_predict(NULL, x->values, out) == ERR_NULL_PTR &&
               ann_quantized_predict(q, NULL, out) == ERR_NULL_PTR &&
               ann_quantized_predict_int8(q, NULL, NULL) == ERR_NULL_PTR);
        TESTEX("Accuracy checks its arguments", ann_quantized_evaluate_accuracy(q, wrong, wrong) < 0.0f &&
               ann_quantized_evaluate_accuracy(NULL, x, x) < 0.0f);
        TESTEX("Export needs a model, file and prefix",
               ann_export_c_header(NULL, NULL, "test_quantize_x.h", "m") == ERR_NULL_PTR &&
               ann_export_c_header(net, q, NULL, "m") == ERR_NULL_PTR &&
               ann_export_c_header(net, q, "test_quantize_x.h", NULL) == ERR_NULL_PTR);
        TESTEX("Prefix must be a C identifier",
               ann_export_c_header(net, q, "test_quantize_x.h", "") == ERR_INVALID &&
               ann_export_c_header(net, q, "test_quantize_x.h", "9lives") == ERR_INVALID &&
               ann_export_c_header(net, q, "test_quantize_x.h", "my-model") == ERR_INVALID);
        TESTEX("Unwritable file is ERR_IO",
               ann_export_c_header(net, q, "no_such_dir/model.h", "model") == ERR_IO);
        TESTEX("Errors are reported to the callback", g_errors >= 10);
        ann_free_quantized(NULL);
        remove("test_quantize_x.h");

        ann_free_quantized(q);
        ann_free_network(net);
        tensor_free(x);
        tensor_free(wrong);
        ann_clear_error_log_callback();
    }
}
