//-----------------------------------------------------------
// Low-bit weights (ann_set_layer_weight_format): int4, ternary and binary.
//
// The rounding rules against a hand-worked row and an independent
// implementation (a twin network given the rounded weights predicts the same,
// for dense, conv and grouped conv layers), the API's checks, save/load and
// ONNX round trips, int8 quantization (packed weights, close to ann_predict,
// batch norm folding with a negative gamma), the packed runtime format, and
// quantization-aware training: from the start, fine-tuning a float model with
// the online API, and the same results for any thread count.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "ann_rt.h"
#include "testy/test.h"

#define MAX_N 1024

// TESTEX with a message built at run time
#define TESTV(msg, s) do { printf("\t%d test case: %s%s%s ", ++test_number, TERM_CYAN, (msg), TERM_RESET); \
                           TEST_ASSERT(s, __FILE__, __LINE__) } while (0)

static void quiet(const char *s) { (void)s; }

static int g_errors;
static void count_errors(int code, const char *msg, const char *fn) { (void)code; (void)msg; (void)fn; g_errors++; }

// fixed, varied parameters (batch norm: gamma near 1, variance positive)
static void set_params(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            int bn = net->layers[l].kind == LAYER_KIND_BATCHNORM;
            for (int i = 0; i < t->rows * t->cols; i++) {
                real v = 0.5f * (real)sin(1.7 * i + 0.9 * l + 2.3 * p + 0.4);
                t->values[i] = bn && p == 0 ? 1.0f + v : bn && p == 3 ? 0.3f + (real)fabs(v) : v;
            }
        }
    net->weights_set = 1;
}

static PTensor make_inputs(int rows, int cols)
{
    PTensor t = tensor_zeros(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        t->values[i] = (real)sin(0.77 * i + 0.3);
    return t;
}

// the rules of Weight_format, written out independently of ann_lowbit.h
static void round_row(Weight_format f, const real *w, real *q, int n)
{
    double sum = 0.0, max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(w[i]);
        if (fabs(w[i]) > max_abs) max_abs = fabs(w[i]);
    }
    if (f == WEIGHT_FORMAT_INT4) {
        double s = max_abs / 7.0;
        for (int i = 0; i < n; i++) {
            double k = s > 0 ? floor(w[i] / s + 0.5) : 0;
            q[i] = (real)(s * (k > 7 ? 7 : k < -7 ? -7 : k));
        }
    } else if (f == WEIGHT_FORMAT_TERNARY) {
        double t = 0.7 * sum / n, kept = 0.0;
        int count = 0;
        for (int i = 0; i < n; i++)
            if (fabs(w[i]) > t) { kept += fabs(w[i]); count++; }
        double s = count ? kept / count : 0.0;
        for (int i = 0; i < n; i++)
            q[i] = (real)(fabs(w[i]) > t ? (w[i] > 0 ? s : -s) : 0.0);
    } else if (f == WEIGHT_FORMAT_BINARY) {
        double s = sum / n;
        for (int i = 0; i < n; i++)
            q[i] = (real)(w[i] >= 0 ? s : -s);
    } else {
        memcpy(q, w, n * sizeof(real));
    }
}

// give twin layer `layer`'s weights rounded as `net`'s layer is: twin then
// computes with exactly the weights net does
static void copy_rounded(PNetwork net, PNetwork twin, int layer, Weight_format f)
{
    int rows, cols;
    ann_get_layer_param_shape(net, layer, 0, &rows, &cols);
    real *w = malloc(rows * cols * sizeof(real)), *q = malloc(rows * cols * sizeof(real));
    ann_get_layer_param(net, layer, 0, w, rows * cols);
    for (int r = 0; r < rows; r++)
        round_row(f, w + r * cols, q + r * cols, cols);
    ann_set_layer_param(twin, layer, 0, q, rows * cols);
    free(w);
    free(q);
}

// 1 if both networks predict exactly the same over the rows
static int same_predictions(PNetwork a, PNetwork b, PTensor x)
{
    int out = a->layers[a->layer_count - 1].node_count;
    real ya[MAX_N], yb[MAX_N];
    for (int r = 0; r < x->rows; r++) {
        ann_predict(a, &x->values[r * x->cols], ya);
        ann_predict(b, &x->values[r * x->cols], yb);
        if (memcmp(ya, yb, out * sizeof(real)) != 0)
            return 0;
    }
    return 1;
}

// largest difference between ann_predict and the int8 model, in output steps
static double int8_error(PNetwork net, PQuantizedNetwork q, PTensor x)
{
    int out = net->layers[net->layer_count - 1].node_count;
    real yf[MAX_N], yq[MAX_N], scale;
    double worst = 0.0;
    ann_quantized_get_io_params(q, NULL, NULL, &scale, NULL);
    for (int r = 0; r < x->rows; r++) {
        ann_predict(net, &x->values[r * x->cols], yf);
        ann_quantized_predict(q, &x->values[r * x->cols], yq);
        for (int k = 0; k < out; k++)
            if (fabs(yf[k] - yq[k]) / scale > worst) worst = fabs(yf[k] - yq[k]) / scale;
    }
    return worst;
}

// largest difference between two int8 models' outputs, in output steps
static double int8_diff(PQuantizedNetwork a, PQuantizedNetwork b, PTensor x, int out)
{
    real ya[MAX_N], yb[MAX_N], scale;
    double worst = 0.0;
    ann_quantized_get_io_params(a, NULL, NULL, &scale, NULL);
    for (int r = 0; r < x->rows; r++) {
        ann_quantized_predict(a, &x->values[r * x->cols], ya);
        ann_quantized_predict(b, &x->values[r * x->cols], yb);
        for (int k = 0; k < out; k++)
            if (fabs(ya[k] - yb[k]) / scale > worst) worst = fabs(ya[k] - yb[k]) / scale;
    }
    return worst;
}

static PNetwork dense_net(int in, int hidden, int out, Activation_type act);

// 1 if q (net quantized with its layer 1 in format f) gives what plain int8
// quantization of a float twin holding the same rounded weights gives, to an
// output step: the twin's integers (+-127 for binary and ternary) and so its
// rescaling multipliers differ, which can round a result the other way. A
// packing mistake (bit order, sign) is off by far more.
static int same_as_twin(PNetwork net, PQuantizedNetwork q, Weight_format f, PTensor x)
{
    int in = net->layers[0].node_count, hidden = net->layers[1].node_count, out = net->layers[2].node_count;
    PNetwork twin = dense_net(in, hidden, out, net->layers[1].activation);
    int rows, cols;
    real *w;
    ann_get_layer_param_shape(net, 2, 0, &rows, &cols);
    w = malloc(rows * cols * sizeof(real));
    ann_get_layer_param(net, 2, 0, w, rows * cols);
    ann_set_layer_param(twin, 2, 0, w, rows * cols);
    free(w);
    copy_rounded(net, twin, 1, f);
    PQuantizedNetwork qt = ann_quantize(twin, x);
    double d = qt ? int8_diff(q, qt, x, out) : 1e9;
    ann_free_quantized(qt);
    ann_free_network(twin);
    (void)f;
    return d <= 1.01;      // one step, give or take float rounding
}

static PNetwork dense_net(int in, int hidden, int out, Activation_type act)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    net->print_func = quiet;
    ann_add_layer(net, in, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, hidden, LAYER_HIDDEN, act);
    ann_add_layer(net, out, LAYER_OUTPUT, ACTIVATION_NULL);
    set_params(net);
    return net;
}

static PNetwork cnn_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    net->print_func = quiet;
    ann_add_input(net, 2, 6, 6);
    ann_add_conv2d(net, 4, 3, 1, 1, ACTIVATION_RELU);
    ann_add_conv2d_grouped(net, 6, 3, 2, 1, 2, ACTIVATION_TANH);
    ann_add_flatten(net);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_NULL);
    set_params(net);
    return net;
}

// classes from fixed prototypes plus noise: `classes` prototypes of `in` values
static void make_classes(real *x, real *y, int rows, int in, int classes, unsigned seed)
{
    unsigned r = seed * 2654435761u + 1u;
    for (int n = 0; n < rows; n++) {
        int c = n % classes;
        for (int i = 0; i < in; i++) {
            r = r * 1103515245u + 12345u;
            double noise = ((r >> 8) & 0xffff) / 65536.0 - 0.5;
            x[n * in + i] = (real)(sin(1.3 * c * i + c) + 0.8 * noise);
        }
        for (int k = 0; k < classes; k++)
            y[n * classes + k] = (real)(k == c);
    }
}

static PNetwork class_net(int in, int hidden, int classes)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_layer(net, in, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, hidden, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, classes, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.005f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 16);
    return net;
}

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    static const Weight_format formats[] = { WEIGHT_FORMAT_INT4, WEIGHT_FORMAT_TERNARY, WEIGHT_FORMAT_BINARY };
    static const char *names[] = { "int4", "ternary", "binary" };

    MODULE("Low-bit Weights");

    SUITE("Rounding Rules");
    {
        // one row, worked by hand: w = { 0.1, -0.4, 0.05, 0.8 }
        //   int4:    s = 0.8 / 7; q = round(w / s) = { 1, -3 (-3.5 rounds up), 0, 7 }
        //   ternary: threshold 0.7 * 0.3375 = 0.23625 keeps -0.4 and 0.8, s = 0.6
        //   binary:  s = mean |w| = 0.3375
        static const real w[4] = { 0.1f, -0.4f, 0.05f, 0.8f };
        static const double want[3][4] = {
            { 0.8 / 7, -3 * 0.8 / 7, 0, 0.8 },
            { 0, -0.6, 0, 0.6 },
            { 0.3375, -0.3375, 0.3375, 0.3375 },
        };
        for (int f = 0; f < 3; f++) {
            PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
            ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
            ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
            real zero = 0.0f, got[4], unit[4], y, back[4];
            ann_set_layer_param(net, 1, 0, w, 4);
            ann_set_layer_param(net, 1, 1, &zero, 1);
            ann_set_layer_weight_format(net, 1, formats[f]);
            int ok = 1;
            for (int i = 0; i < 4; i++) {
                memset(unit, 0, sizeof(unit));
                unit[i] = 1.0f;
                ann_predict(net, unit, &y);
                got[i] = y;
                ok &= fabs(got[i] - want[f][i]) < 1e-6;
            }
            ann_get_layer_param(net, 1, 0, back, 4);
            char what[96];
            snprintf(what, sizeof(what), "%s: a hand-worked row predicts with its rounded weights", names[f]);
            TESTV(what, ok);
            snprintf(what, sizeof(what), "%s: ann_get_layer_param still reads the float weights", names[f]);
            TESTV(what, memcmp(back, w, sizeof(w)) == 0);
            ann_free_network(net);
        }

        // dense layers (both, including the output layer) and conv layers (plain
        // and grouped): a twin given the rounded weights predicts identically
        PTensor x = make_inputs(12, 9);
        PTensor xc = make_inputs(6, 72);
        for (int f = 0; f < 3; f++) {
            PNetwork net = dense_net(9, 7, 3, ACTIVATION_RELU), twin = dense_net(9, 7, 3, ACTIVATION_RELU);
            ann_set_layer_weight_format(net, 1, formats[f]);
            ann_set_layer_weight_format(net, 2, formats[f]);
            copy_rounded(net, twin, 1, formats[f]);
            copy_rounded(net, twin, 2, formats[f]);
            char what[96];
            snprintf(what, sizeof(what), "%s dense layers predict as with the rounded weights", names[f]);
            TESTV(what, same_predictions(net, twin, x));

            ann_set_layer_weight_format(net, 1, WEIGHT_FORMAT_FLOAT);
            ann_set_layer_weight_format(net, 2, WEIGHT_FORMAT_FLOAT);
            PNetwork plain = dense_net(9, 7, 3, ACTIVATION_RELU);
            snprintf(what, sizeof(what), "%s set back to float predicts with the float weights", names[f]);
            TESTV(what, same_predictions(net, plain, x));
            ann_free_network(plain);
            ann_free_network(net);
            ann_free_network(twin);

            net = cnn_net();
            twin = cnn_net();
            ann_set_layer_weight_format(net, 1, formats[f]);
            ann_set_layer_weight_format(net, 2, formats[f]);
            copy_rounded(net, twin, 1, formats[f]);
            copy_rounded(net, twin, 2, formats[f]);
            snprintf(what, sizeof(what), "%s conv and grouped conv layers predict as with the rounded weights", names[f]);
            TESTV(what, same_predictions(net, twin, xc));
            ann_free_network(net);
            ann_free_network(twin);
        }
        tensor_free(x);
        tensor_free(xc);
    }

    SUITE("API");
    {
        ann_set_error_log_callback(count_errors);
        g_errors = 0;
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 1, 6, 6);
        ann_add_conv2d(net, 2, 3, 1, 1, ACTIVATION_RELU);      // 1
        ann_add_batchnorm(net, ACTIVATION_RELU);                // 2
        ann_add_maxpool2d(net, 2, 2);                           // 3
        ann_add_flatten(net);                                   // 4
        ann_add_layer(net, 5, LAYER_HIDDEN, ACTIVATION_RELU);   // 5
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);   // 6

        TESTEX("NULL network", ann_set_layer_weight_format(NULL, 1, WEIGHT_FORMAT_BINARY) == ERR_NULL_PTR &&
               ann_set_weight_format(NULL, WEIGHT_FORMAT_BINARY) == ERR_NULL_PTR);
        TESTEX("Only dense and conv layers take a format",
               ann_set_layer_weight_format(net, 0, WEIGHT_FORMAT_BINARY) == ERR_INVALID &&
               ann_set_layer_weight_format(net, 2, WEIGHT_FORMAT_BINARY) == ERR_INVALID &&
               ann_set_layer_weight_format(net, 3, WEIGHT_FORMAT_BINARY) == ERR_INVALID &&
               ann_set_layer_weight_format(net, 4, WEIGHT_FORMAT_BINARY) == ERR_INVALID &&
               ann_set_layer_weight_format(net, 7, WEIGHT_FORMAT_BINARY) == ERR_INVALID &&
               ann_set_layer_weight_format(net, -1, WEIGHT_FORMAT_BINARY) == ERR_INVALID);
        TESTEX("Bad formats are rejected",
               ann_set_layer_weight_format(net, 1, (Weight_format)4) == ERR_INVALID &&
               ann_set_layer_weight_format(net, 1, (Weight_format)-1) == ERR_INVALID &&
               ann_set_weight_format(net, (Weight_format)9) == ERR_INVALID);
        TESTEX("Rejections are reported", g_errors == 11);
        TESTEX("Layers start float", ann_get_layer_weight_format(net, 1) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 5) == WEIGHT_FORMAT_FLOAT);
        TESTEX("ann_set_weight_format sets the dense and conv layers but the output",
               ann_set_weight_format(net, WEIGHT_FORMAT_TERNARY) == ERR_OK &&
               ann_get_layer_weight_format(net, 1) == WEIGHT_FORMAT_TERNARY &&
               ann_get_layer_weight_format(net, 2) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 5) == WEIGHT_FORMAT_TERNARY &&
               ann_get_layer_weight_format(net, 6) == WEIGHT_FORMAT_FLOAT);
        TESTEX("ann_set_layer_weight_format sets one layer, the output layer too",
               ann_set_layer_weight_format(net, 6, WEIGHT_FORMAT_INT4) == ERR_OK &&
               ann_get_layer_weight_format(net, 6) == WEIGHT_FORMAT_INT4 &&
               ann_get_layer_weight_format(net, 5) == WEIGHT_FORMAT_TERNARY);
        TESTEX("Bad queries give float", ann_get_layer_weight_format(NULL, 1) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 99) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 0) == WEIGHT_FORMAT_FLOAT);
        TESTEX("ann_set_weight_format(FLOAT) clears them",
               ann_set_weight_format(net, WEIGHT_FORMAT_FLOAT) == ERR_OK &&
               ann_get_layer_weight_format(net, 1) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 5) == WEIGHT_FORMAT_FLOAT &&
               ann_get_layer_weight_format(net, 6) == WEIGHT_FORMAT_INT4);
        ann_free_network(net);
        ann_clear_error_log_callback();

        if (ann_gpu_init()) {
            net = dense_net(9, 7, 3, ACTIVATION_RELU);
            int uploaded = ann_gpu_upload_network(net) == ERR_OK;
            ann_set_error_log_callback(count_errors);
            ann_set_layer_weight_format(net, 1, WEIGHT_FORMAT_BINARY);
            int refused = ann_gpu_upload_network(net) == ERR_INVALID;
            ann_clear_error_log_callback();
            PNetwork twin = dense_net(9, 7, 3, ACTIVATION_RELU);
            copy_rounded(net, twin, 1, WEIGHT_FORMAT_BINARY);
            PTensor x = make_inputs(8, 9);
            TESTEX("GPU: a low-bit network is not uploaded and predicts on the CPU",
                   uploaded && refused && same_predictions(net, twin, x));
            tensor_free(x);
            ann_free_network(twin);
            ann_free_network(net);
        }
    }

    SUITE("Save, Load and ONNX");
    {
        PTensor xc = make_inputs(6, 72);
        PNetwork net = cnn_net();
        ann_set_layer_weight_format(net, 1, WEIGHT_FORMAT_TERNARY);
        ann_set_layer_weight_format(net, 2, WEIGHT_FORMAT_BINARY);
        ann_set_layer_weight_format(net, 4, WEIGHT_FORMAT_INT4);
        ann_set_layer_trainable(net, 2, 0);

        for (int binary = 0; binary < 2; binary++) {
            const char *file = binary ? "test_lowbit.nnb" : "test_lowbit.nna";
            int saved = (binary ? ann_save_network_binary(net, file) : ann_save_network(net, file)) == ERR_OK;
            PNetwork back = binary ? ann_load_network_binary(file) : ann_load_network(file);
            int ok = saved && back &&
                     ann_get_layer_weight_format(back, 1) == WEIGHT_FORMAT_TERNARY &&
                     ann_get_layer_weight_format(back, 2) == WEIGHT_FORMAT_BINARY &&
                     ann_get_layer_weight_format(back, 4) == WEIGHT_FORMAT_INT4 &&
                     !ann_get_layer_trainable(back, 2) && ann_get_layer_trainable(back, 1);
            if (back) back->print_func = quiet;
            TESTV(binary ? ".nnb keeps the formats and the frozen layer" : ".nna keeps the formats and the frozen layer", ok);
            TESTV(binary ? ".nnb loads predicting the same" : ".nna loads predicting the same",
                   back && same_predictions(net, back, xc));
            if (back) {
                real w[MAX_N], wb[MAX_N];
                int rows, cols;
                ann_get_layer_param_shape(net, 1, 0, &rows, &cols);
                ann_get_layer_param(net, 1, 0, w, rows * cols);
                ann_get_layer_param(back, 1, 0, wb, rows * cols);
                TESTV(binary ? ".nnb keeps the float weights under them" : ".nna keeps the float weights under them",
                       memcmp(w, wb, rows * cols * sizeof(real)) == 0);
            }
            ann_free_network(back);
            remove(file);
        }

        // ONNX holds the weights the network computes with
        TESTEX("ONNX export", ann_export_onnx(net, "test_lowbit.onnx.json") == ERR_OK);
        PNetwork imported = ann_import_onnx("test_lowbit.onnx.json");
        int close = imported != NULL;
        for (int r = 0; close && r < xc->rows; r++) {
            real ya[8], yb[8];
            ann_predict(net, &xc->values[r * xc->cols], ya);
            ann_predict(imported, &xc->values[r * xc->cols], yb);
            for (int k = 0; k < 3; k++)
                close &= fabs(ya[k] - yb[k]) <= 1e-5 * (1 + fabs(ya[k]));
        }
        TESTEX("and imports predicting as the low-bit network", close);
        ann_free_network(imported);
        remove("test_lowbit.onnx.json");
        ann_free_network(net);
        tensor_free(xc);
    }

    SUITE("int8 Quantization");
    {
        // the runtime's packed format
        static const uint8_t b1[] = { 0x2d };          // 1,0,1,1,0,1,0,0 -> +1,-1,+1,+1,-1,+1,-1,-1
        static const uint8_t b2[] = { 0x1d, 0x03 };    // 01,11,01,00, 11 -> +1,-1,+1,0,-1
        static const uint8_t b4[] = { 0x97, 0x00 };    // 0111,1001, 0000 -> 7,-7,0
        static const int w1[] = { 1, -1, 1, 1, -1, 1, -1, -1 }, w2[] = { 1, -1, 1, 0, -1 }, w4[] = { 7, -7, 0 };
        int ok = 1;
        for (int i = 0; i < 8; i++) ok &= ann_rt_weight_q(b1, 1, i) == w1[i];
        for (int i = 0; i < 5; i++) ok &= ann_rt_weight_q(b2, 2, i) == w2[i];
        for (int i = 0; i < 3; i++) ok &= ann_rt_weight_q(b4, 4, i) == w4[i];
        TESTEX("ann_rt_weight_q unpacks binary, ternary and int4 weights", ok);

        PTensor x = make_inputs(64, 64);
        size_t bytes[4];
        for (int f = 0; f < 4; f++) {
            PNetwork net = dense_net(64, 32, 4, ACTIVATION_RELU);
            if (f > 0)
                ann_set_layer_weight_format(net, 1, formats[f - 1]);
            PQuantizedNetwork q = ann_quantize(net, x);
            bytes[f] = ann_quantized_param_bytes(q);
            if (f > 0) {
                char what[96];
                snprintf(what, sizeof(what), "%s: the int8 model follows ann_predict (%.2f output steps)",
                         names[f - 1], q ? int8_error(net, q, x) : -1.0);
                TESTV(what, q && int8_error(net, q, x) <= 3.0);
                snprintf(what, sizeof(what), "%s: packed, it matches int8 of the same rounded weights", names[f - 1]);
                TESTV(what, q && same_as_twin(net, q, formats[f - 1], x));
            }
            ann_free_quantized(q);
            ann_free_network(net);
        }
        // hidden layer 32 rows of 64 weights plus 32 * 9 bytes of bias, multiplier and shift;
        // the output layer 4 * 32 + 4 * 9
        size_t rest = 32 * 9 + 4 * 32 + 4 * 9;
        TESTEX("Packed sizes: 8, 4, 2 and 1 bits per weight",
               bytes[0] == 32 * 64 + rest && bytes[1] == 32 * 32 + rest &&
               bytes[2] == 32 * 16 + rest && bytes[3] == 32 * 8 + rest);
        TESTEX("ann_quantized_param_bytes(NULL) is 0", ann_quantized_param_bytes(NULL) == 0);

        // batch norm folded into a low-bit layer, a negative gamma flipping the integers
        PNetwork bn = ann_make_network(OPT_ADAM, LOSS_MSE);
        bn->print_func = quiet;
        ann_add_layer(bn, 64, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(bn, 8, LAYER_HIDDEN, ACTIVATION_NULL);
        ann_add_batchnorm(bn, ACTIVATION_RELU);
        ann_add_layer(bn, 3, LAYER_OUTPUT, ACTIVATION_NULL);
        set_params(bn);
        bn->layers[2].params[0].value->values[1] = -0.8f;
        bn->layers[2].params[0].value->values[5] = -1.3f;
        for (int f = 0; f < 3; f++) {
            ann_set_layer_weight_format(bn, 1, formats[f]);
            PQuantizedNetwork q = ann_quantize(bn, x);
            char what[96];
            snprintf(what, sizeof(what), "%s with batch norm folded in (negative gammas) follows ann_predict", names[f]);
            TESTV(what, q && int8_error(bn, q, x) <= 3.0);
            ann_free_quantized(q);
        }
        ann_free_network(bn);

        // a binary row of zeros has no int8 form
        PNetwork zero = dense_net(64, 4, 2, ACTIVATION_RELU);
        real w0[4 * 64];
        ann_get_layer_param(zero, 1, 0, w0, 4 * 64);
        memset(w0, 0, 64 * sizeof(real));               // the first row
        ann_set_layer_param(zero, 1, 0, w0, 4 * 64);
        ann_set_layer_weight_format(zero, 1, WEIGHT_FORMAT_BINARY);
        ann_set_error_log_callback(count_errors);
        PQuantizedNetwork qz = ann_quantize(zero, x);
        ann_clear_error_log_callback();
        TESTEX("A binary row of zeros cannot be quantized", qz == NULL);
        ann_set_layer_weight_format(zero, 1, WEIGHT_FORMAT_TERNARY);
        qz = ann_quantize(zero, x);
        TESTEX("A ternary one can, its zero row packed as zeros", qz != NULL && same_as_twin(zero, qz, WEIGHT_FORMAT_TERNARY, x));
        ann_free_quantized(qz);
        ann_free_network(zero);
        tensor_free(x);
    }

    SUITE("Quantization-aware Training");
    {
        enum { IN = 24, HIDDEN = 24, CLASSES = 6, ROWS = 384 };
        real *xs = malloc(ROWS * IN * sizeof(real)), *ys = malloc(ROWS * CLASSES * sizeof(real));
        make_classes(xs, ys, ROWS, IN, CLASSES, 3);
        PTensor x = tensor_create_from_array(ROWS, IN, xs);
        PTensor y = tensor_create_from_array(ROWS, CLASSES, ys);

        // from the start, in binary
        ann_set_seed(11);
        PNetwork net = class_net(IN, HIDDEN, CLASSES);
        ann_set_weight_format(net, WEIGHT_FORMAT_BINARY);
        ann_set_epoch_limit(net, 40);
        ann_train_network(net, x, y, ROWS);
        real acc = ann_evaluate_accuracy(net, x, y);
        PQuantizedNetwork q = ann_quantize(net, x);
        real acc_q = q ? ann_quantized_evaluate_accuracy(q, x, y) : 0;
        char what[128];
        snprintf(what, sizeof(what), "Binary hidden layer trained from the start: %.1f%%", 100.0 * acc);
        TESTV(what, acc >= 0.95f);
        snprintf(what, sizeof(what), "and its int8 model keeps it: %.1f%%", 100.0 * acc_q);
        TESTV(what, acc_q >= acc - 0.03f);
        ann_free_quantized(q);
        ann_free_network(net);

        // float, then ternary: rounding costs accuracy, fine-tuning with the online API wins it back
        ann_set_seed(12);
        net = class_net(IN, HIDDEN, CLASSES);
        ann_set_epoch_limit(net, 30);
        ann_train_network(net, x, y, ROWS);
        real acc_float = ann_evaluate_accuracy(net, x, y);
        ann_set_layer_weight_format(net, 1, WEIGHT_FORMAT_TERNARY);
        ann_set_layer_weight_format(net, 2, WEIGHT_FORMAT_TERNARY);
        real acc_rounded = ann_evaluate_accuracy(net, x, y);
        ann_train_begin(net);
        for (int epoch = 0; epoch < 15; epoch++)
            for (int r = 0; r + 16 <= ROWS; r += 16)
                ann_train_step(net, &xs[r * IN], &ys[r * CLASSES], 16);
        ann_train_end(net);
        real acc_tuned = ann_evaluate_accuracy(net, x, y);
        snprintf(what, sizeof(what), "Fine-tuning a ternary model: float %.1f%%, rounded %.1f%%, tuned %.1f%%",
                 100.0 * acc_float, 100.0 * acc_rounded, 100.0 * acc_tuned);
        TESTV(what, acc_tuned >= 0.95f && acc_tuned >= acc_rounded);
        ann_free_network(net);

        // the same results for any thread count
        real out[2][ROWS * CLASSES];
        for (int t = 0; t < 2; t++) {
            ann_set_num_threads(t ? 4 : 1);
            ann_set_seed(13);
            net = class_net(IN, HIDDEN, CLASSES);
            ann_set_layer_weight_format(net, 1, WEIGHT_FORMAT_INT4);
            ann_set_layer_weight_format(net, 2, WEIGHT_FORMAT_BINARY);
            ann_set_epoch_limit(net, 5);
            ann_train_network(net, x, y, ROWS);
            for (int r = 0; r < ROWS; r++)
                ann_predict(net, &xs[r * IN], &out[t][r * CLASSES]);
            ann_free_network(net);
        }
        ann_set_num_threads(0);
        TESTEX("Training gives the same results on 1 and 4 threads", memcmp(out[0], out[1], sizeof(out[0])) == 0);

        tensor_free(x);
        tensor_free(y);
        free(xs);
        free(ys);
    }
}
