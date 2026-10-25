//-----------------------------------------------------------
// Batch normalization layers.
//
// Shapes and argument checks, prediction against the formula with the running
// statistics, training-mode statistics and running averages, finite-difference
// gradient checks, training, save/load and ONNX export and import.
//
// Gradient checks follow test_conv.c, with a step of 3e-3 (the float loss is
// too noisy below that, and batch norm too curved above it), except that the loss must use the
// batch's statistics, as training does: it is read from ann_train_step with a
// learning rate of 0 (SGD then changes no parameter). The analytic gradient
// is read from one SGD step, g = (W_after - W_before) / lr = -dL/dW.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

#define GC_LR    0.01f
#define GC_EPS   3e-3f
#define GC_ABS   2e-3f   // absolute tolerance
#define GC_REL   2e-2f   // relative tolerance
#define GC_BATCH 4
#define MAX_IN   64
#define MAX_OUT  8

static void quiet(const char *s) { (void)s; }

static int file_contains(const char *path, const char *needle)
{
    static char buf[1 << 16];
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

static real g_x[GC_BATCH * MAX_IN];
static real g_t[GC_BATCH * MAX_OUT];

// Fill every trainable parameter with fixed, varied values (batch norm's
// gamma near 1) and inputs and targets with a pattern
static void set_params(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            if (net->layers[l].params[p].state)
                continue;
            for (int i = 0; i < t->rows * t->cols; i++) {
                real v = 0.5f * (real)sin(1.7 * i + 0.9 * l + 2.3 * p + 0.4);
                t->values[i] = (net->layers[l].kind == LAYER_KIND_BATCHNORM && p == 0) ? 1.0f + 0.5f * v : v;
            }
        }
}

static void set_data(PNetwork net, int softmax_targets)
{
    int in = net->layers[0].node_count, out = net->layers[net->layer_count - 1].node_count;
    for (int i = 0; i < GC_BATCH * in; i++)
        g_x[i] = (real)sin(0.77 * i + 0.3) * 0.9f + 0.2f * (real)cos(0.31 * i);
    for (int b = 0; b < GC_BATCH; b++)
        for (int k = 0; k < out; k++)
            g_t[b * out + k] = softmax_targets ? (k == b % out ? 0.7f : 0.3f / (out - 1))
                                               : 0.4f * (real)cos(1.3 * k + b);
}

// The batch's loss in training mode, as the sum over samples of
// 1/2 * sum((t - y)^2) for MSE or -sum(t * log(y)) for cross-entropy
static double train_loss(PNetwork net)
{
    int out = net->layers[net->layer_count - 1].node_count;
    ann_set_learning_rate(net, 0.0f);
    double loss = ann_train_step(net, g_x, g_t, GC_BATCH);
    ann_set_learning_rate(net, GC_LR);
    return net->loss_type == LOSS_MSE ? loss * GC_BATCH * out / 2.0 : loss * GC_BATCH;
}

// Largest scaled error between analytic and numeric gradients over every
// trainable parameter; <= 1 when all are within tolerance. Frees net.
static real grad_check(PNetwork net, int softmax_targets)
{
    if (!net) return 1e9f;
    net->print_func = quiet;
    ann_train_begin(net);
    set_params(net);
    set_data(net, softmax_targets);

    static real numeric[4096], before[4096];
    int n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            if (net->layers[l].params[p].state)
                continue;
            for (int i = 0; i < t->rows * t->cols; i++, n++) {
                real saved = t->values[i];
                t->values[i] = saved + GC_EPS;
                double lp = train_loss(net);
                t->values[i] = saved - GC_EPS;
                double lm = train_loss(net);
                t->values[i] = saved;
                numeric[n] = (real)(-(lp - lm) / (2.0 * GC_EPS));
                before[n] = saved;
            }
        }

    ann_set_learning_rate(net, GC_LR);
    ann_train_step(net, g_x, g_t, GC_BATCH);

    real worst = 0.0f;
    n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            if (net->layers[l].params[p].state)
                continue;
            for (int i = 0; i < t->rows * t->cols; i++, n++) {
                real analytic = (t->values[i] - before[n]) / GC_LR;
                real err = (real)fabs(analytic - numeric[n]) / (GC_ABS + GC_REL * (real)fabs(numeric[n]));
                if (err > worst) worst = err;
            }
        }

    ann_train_end(net);
    ann_free_network(net);
    return worst;
}

// 6 -> dense(5, linear) -> batch norm (tanh) -> dense(3, linear), MSE
static PNetwork net_dense_bn(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 6, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 5, LAYER_HIDDEN, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_TANH);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_NULL);
    return net;
}

// 2x4x4 -> conv(3, k3, padding 1, linear) -> batch norm (sigmoid) -> maxpool 2 -> dense(3, softmax), cross-entropy
static PNetwork net_conv_bn(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_input(net, 2, 4, 4);
    ann_add_conv2d(net, 3, 3, 1, 1, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_SIGMOID);
    ann_add_maxpool2d(net, 2, 2);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    return net;
}

// the same conv -> batch norm (linear) -> dense(3, linear), MSE
static PNetwork net_conv_bn_mse(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_input(net, 2, 4, 4);
    ann_add_conv2d(net, 3, 3, 1, 1, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_NULL);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_NULL);
    return net;
}

// batch norm straight off a 1x3x5 input (no gradient to pass down) -> conv(2, k2, gelu) -> dense(2), MSE
static PNetwork net_input_bn(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_input(net, 1, 3, 5);
    ann_add_batchnorm(net, ACTIVATION_NULL);
    ann_add_conv2d(net, 2, 2, 1, 0, ACTIVATION_GELU);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
    return net;
}

// batch norm as the output layer: 4 -> dense(3, relu) -> batch norm (sigmoid), MSE
static PNetwork net_bn_output(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 3, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_batchnorm(net, ACTIVATION_SIGMOID);
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

// conv -> batch norm (relu) -> pool -> dense -> batch norm (relu) -> softmax
static PNetwork make_bar_bn_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_input(net, 1, 6, 6);
    ann_add_conv2d(net, 4, 3, 1, 1, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_RELU);
    ann_add_maxpool2d(net, 2, 2);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_RELU);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 8);
    return net;
}

static int same_structure(PNetwork a, PNetwork b)
{
    if (!a || !b || a->layer_count != b->layer_count)
        return 0;
    for (int l = 0; l < a->layer_count; l++)
        if (a->layers[l].kind != b->layers[l].kind || a->layers[l].activation != b->layers[l].activation ||
            a->layers[l].shape.c != b->layers[l].shape.c || a->layers[l].shape.h != b->layers[l].shape.h ||
            a->layers[l].shape.w != b->layers[l].shape.w)
            return 0;
    return 1;
}

static int file_version(const char *path, int binary)
{
    int version = 0;
    FILE *f = fopen(path, binary ? "rb" : "rt");
    if (f) {
        if (binary) { if (fread(&version, sizeof(int), 1, f) != 1) version = 0; }
        else if (fscanf(f, "%d", &version) != 1) version = 0;
        fclose(f);
    }
    return version;
}

// Largest difference between two networks' predictions on count inputs
static real max_pred_diff(PNetwork a, PNetwork b, const real *x, int count)
{
    int in = a->layers[0].node_count, out = a->layers[a->layer_count - 1].node_count;
    real worst = 0.0f, ya[MAX_OUT], yb[MAX_OUT];
    if (out > MAX_OUT || b->layers[b->layer_count - 1].node_count != out)
        return 1e9f;
    for (int n = 0; n < count; n++) {
        ann_predict(a, &x[n * in], ya);
        ann_predict(b, &x[n * in], yb);
        for (int k = 0; k < out; k++)
            if (fabs(ya[k] - yb[k]) > worst) worst = (real)fabs(ya[k] - yb[k]);
    }
    return worst;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Batch Normalization Layers");

    SUITE("Shapes");
    {
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 2, 4, 5);
        TESTEX("Batch norm on an image keeps its shape", ann_add_batchnorm(net, ACTIVATION_RELU) == ERR_OK &&
               net->layers[1].shape.c == 2 && net->layers[1].shape.h == 4 && net->layers[1].shape.w == 5 &&
               ann_get_layer_kind(net, 1) == LAYER_KIND_BATCHNORM && net->layers[1].activation == ACTIVATION_RELU);
        TESTEX("with gamma, beta, running mean and running variance, one per channel",
               net->layers[1].param_count == 4 && net->layers[1].params[0].value->cols == 2 &&
               net->layers[1].params[3].value->cols == 2);
        TESTEX("of which the running statistics are state, not trained or regularized",
               !net->layers[1].params[0].state && !net->layers[1].params[1].state &&
               net->layers[1].params[2].state && net->layers[1].params[3].state &&
               !net->layers[1].params[0].regularize);
        TESTEX("starting as the identity: gamma 1, beta 0, mean 0, variance 1",
               net->layers[1].params[0].value->values[1] == 1.0f && net->layers[1].params[1].value->values[1] == 0.0f &&
               net->layers[1].params[2].value->values[1] == 0.0f && net->layers[1].params[3].value->values[1] == 1.0f);
        int k, st, p;
        TESTEX("ann_get_layer_settings reports zeros",
               ann_get_layer_settings(net, 1, &k, &st, &p) == ERR_OK && k == 0 && st == 0 && p == 0);
        ann_add_flatten(net);
        TESTEX("Batch norm on a flat layer has one channel per node", ann_add_batchnorm(net, ACTIVATION_NULL) == ERR_OK &&
               net->layers[3].shape.c == 40 && net->layers[3].params[0].value->cols == 40);
        TESTEX("A bad activation is rejected", ann_add_batchnorm(net, (Activation_type)99) == ERR_INVALID);
        TESTEX("Rejected layers are not counted", net->layer_count == 4);
        ann_free_network(net);

        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        TESTEX("Batch norm as the first layer is rejected", ann_add_batchnorm(net, ACTIVATION_NULL) == ERR_INVALID);
        ann_free_network(net);
    }

    SUITE("Prediction");
    {
        // y = gamma * (x - mean) / sqrt(var + 1e-5) + beta, per channel
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 3, 2, 2);
        ann_add_batchnorm(net, ACTIVATION_NULL);
        ann_train_begin(net);
        ann_train_end(net);
        const real gamma[3] = { 1.5f, -0.5f, 2.0f }, beta[3] = { 0.25f, 0.0f, -1.0f };
        const real mean[3] = { 0.1f, -0.3f, 0.7f }, var[3] = { 0.5f, 2.0f, 0.01f };
        for (int c = 0; c < 3; c++) {
            net->layers[1].params[0].value->values[c] = gamma[c];
            net->layers[1].params[1].value->values[c] = beta[c];
            net->layers[1].params[2].value->values[c] = mean[c];
            net->layers[1].params[3].value->values[c] = var[c];
        }
        real x[2 * 12], y[2 * 12], y1[12];
        for (int i = 0; i < 24; i++) x[i] = (real)sin(0.9 * i) * 2.0f;
        ann_predict(net, x, y1);
        int close = 1;
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 4; i++) {
                double want = gamma[c] * (x[c * 4 + i] - mean[c]) / sqrt(var[c] + 1e-5) + beta[c];
                close &= fabs(y1[c * 4 + i] - want) < 1e-5;
            }
        TESTEX("Prediction uses the running statistics", close);
        TESTEX("ann_predict_batch matches ann_predict", ann_predict_batch(net, x, y, 2) == ERR_OK &&
               memcmp(y, y1, sizeof(y1)) == 0);
        ann_free_network(net);
    }

    SUITE("Training Statistics");
    {
        // one step: the output is normalized with the batch's statistics, and
        // the running averages move 0.1 of the way toward them
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        net->print_func = quiet;
        ann_add_layer(net, 3, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_batchnorm(net, ACTIVATION_NULL);
        ann_train_begin(net);
        ann_set_learning_rate(net, 0.0f);
        enum { B = 5 };
        real x[B * 3], t[B * 3];
        for (int i = 0; i < B * 3; i++) {
            x[i] = (real)(i % 3) * 2.0f + (real)sin(1.1 * i);
            t[i] = 0.0f;
        }
        ann_train_step(net, x, t, B);
        int normalized = 1, running = 1;
        for (int c = 0; c < 3; c++) {
            double m = 0.0, v = 0.0, ym = 0.0, yv = 0.0;
            for (int b = 0; b < B; b++) m += x[b * 3 + c] / B;
            for (int b = 0; b < B; b++) v += (x[b * 3 + c] - m) * (x[b * 3 + c] - m);
            for (int b = 0; b < B; b++) ym += net->layers[1].t_batch_values->values[b * 3 + c] / B;
            for (int b = 0; b < B; b++) yv += pow(net->layers[1].t_batch_values->values[b * 3 + c] - ym, 2) / B;
            normalized &= fabs(ym) < 1e-5 && fabs(yv - v / B / (v / B + 1e-5)) < 1e-4;
            running &= fabs(net->layers[1].params[2].value->values[c] - 0.1 * m) < 1e-5 &&
                       fabs(net->layers[1].params[3].value->values[c] - (0.9 + 0.1 * v / (B - 1))) < 1e-5;
        }
        TESTEX("Training normalizes each channel with the batch's mean and variance", normalized);
        TESTEX("and updates the running mean and unbiased variance with momentum 0.1", running);
        TESTEX("Gamma and beta did not move with a learning rate of 0",
               net->layers[1].params[0].value->values[0] == 1.0f && net->layers[1].params[1].value->values[0] == 0.0f);

        real y[3];
        ann_predict(net, x, y);
        double want = (x[0] - net->layers[1].params[2].value->values[0]) /
                      sqrt(net->layers[1].params[3].value->values[0] + 1e-5);
        TESTEX("ann_predict mid-training uses the running statistics", fabs(y[0] - want) < 1e-5);
        ann_train_end(net);
        ann_free_network(net);
    }

    SUITE("Gradients");
    TESTEX("Gradients: dense -> batch norm (tanh) -> dense", grad_check(net_dense_bn(), 0) <= 1.0f);
    TESTEX("Gradients: conv -> batch norm (sigmoid) -> pool -> softmax", grad_check(net_conv_bn(), 1) <= 1.0f);
    TESTEX("Gradients: batch norm on the input -> conv (gelu) -> dense", grad_check(net_input_bn(), 0) <= 1.0f);
    TESTEX("Gradients: batch norm (sigmoid) as the output layer", grad_check(net_bn_output(), 0) <= 1.0f);
    TESTEX("Gradients: conv -> batch norm -> dense, MSE", grad_check(net_conv_bn_mse(), 0) <= 1.0f);

    SUITE("Training, Saving and ONNX");
    {
        enum { N = 48 };
        real x[N * 36], y[N * 2];
        make_bars(x, y, N);
        PTensor tx = tensor_create_from_array(N, 36, x);
        PTensor ty = tensor_create_from_array(N, 2, y);
        PNetwork net = make_bar_bn_net();
        ann_set_epoch_limit(net, 100);
        ann_train_network(net, tx, ty, N);
        real acc = ann_evaluate_accuracy(net, tx, ty);
        printf("\t  bars with batch norm: accuracy %.2f\n", acc);
        TESTEX("A CNN with batch norm learns the bars (predicting with running statistics)", acc >= 0.95f);
        TESTEX("and its running statistics moved", net->layers[2].params[2].value->values[0] != 0.0f &&
               net->layers[2].params[3].value->values[0] != 1.0f);

        TESTEX("Text and binary saves succeed", ann_save_network(net, "test_bn.nna") == ERR_OK &&
               ann_save_network_binary(net, "test_bn.nnb") == ERR_OK);
        TESTEX("and write format v2", file_version("test_bn.nna", 0) == 2 && file_version("test_bn.nnb", 1) == 2);
        PNetwork text = ann_load_network("test_bn.nna");
        PNetwork bin = ann_load_network_binary("test_bn.nnb");
        TESTEX("Loaded networks keep the layer kinds and activations", same_structure(net, text) && same_structure(net, bin));
        TESTEX("and the running statistics", bin && bin->layers[5].params[3].value->values[2] == net->layers[5].params[3].value->values[2]);
        TESTEX("Binary load predicts identically", bin && max_pred_diff(net, bin, x, N) == 0.0f);
        TESTEX("Text load predicts the same (6 decimals)", text && max_pred_diff(net, text, x, N) < 1e-3f);
        ann_free_network(text);
        ann_free_network(bin);
        remove("test_bn.nna");
        remove("test_bn.nnb");

        const char *onnx = "test_bn.onnx.json";
        TESTEX("A CNN with batch norm exports to ONNX", ann_export_onnx(net, onnx) == ERR_OK);
        TESTEX("with BatchNormalization nodes and their statistics",
               file_contains(onnx, "\"op_type\": \"BatchNormalization\"") &&
               file_contains(onnx, "\"input\": [\"conv_0_out\", \"bn_1_scale\", \"bn_1_bias\", \"bn_1_mean\", \"bn_1_var\"]") &&
               file_contains(onnx, "\"name\": \"bn_4_var\"") &&
               file_contains(onnx, "{\"name\": \"epsilon\", \"type\": 1, \"f\": 1e-05}, {\"name\": \"momentum\", \"type\": 1, \"f\": 0.9}"));
        PNetwork back = ann_import_onnx(onnx);
        TESTEX("and imports with the same structure", same_structure(net, back));
        TESTEX("and predicts identically", back && max_pred_diff(net, back, x, N) == 0.0f);
        ann_free_network(back);
        remove(onnx);
        tensor_free(tx);
        tensor_free(ty);
        ann_free_network(net);

        // online training keeps the running statistics going
        net = make_bar_bn_net();
        ann_train_begin(net);
        for (int step = 0; step < 6; step++)
            ann_train_step(net, &x[step * 8 * 36], &y[step * 8 * 2], 8);
        real mean_before = net->layers[2].params[2].value->values[1];
        ann_train_end(net);
        ann_train_begin(net);
        ann_train_step(net, x, y, 8);
        ann_train_end(net);
        TESTEX("Online training continues the running statistics",
               mean_before != 0.0f && net->layers[2].params[2].value->values[1] != mean_before &&
               fabs(net->layers[2].params[2].value->values[1] - mean_before) < fabs(mean_before));
        ann_free_network(net);

        // a PyTorch-style model (protobuf JSON): Gemm, BatchNormalization
        // with epsilon 1e-3 (Keras's), Relu, Gemm
        static const char *model =
            "{\"irVersion\":\"8\",\"producerName\":\"pytorch\",\"graph\":{\"node\":["
            "{\"input\":[\"x\",\"fc1.w\",\"fc1.b\"],\"output\":[\"h\"],\"opType\":\"Gemm\",\"attribute\":[{\"name\":\"transB\",\"i\":\"1\",\"type\":\"INT\"}]},"
            "{\"input\":[\"h\",\"bn.s\",\"bn.b\",\"bn.m\",\"bn.v\"],\"output\":[\"n\"],\"opType\":\"BatchNormalization\",\"attribute\":[{\"name\":\"epsilon\",\"f\":0.001,\"type\":\"FLOAT\"},{\"name\":\"momentum\",\"f\":0.9,\"type\":\"FLOAT\"}]},"
            "{\"input\":[\"n\"],\"output\":[\"r\"],\"opType\":\"Relu\"},"
            "{\"input\":[\"r\",\"fc2.w\",\"fc2.b\"],\"output\":[\"y\"],\"opType\":\"Gemm\",\"attribute\":[{\"name\":\"transB\",\"i\":\"1\",\"type\":\"INT\"}]}],"
            "\"name\":\"g\",\"initializer\":["
            "{\"dims\":[\"2\",\"3\"],\"dataType\":1,\"name\":\"fc1.w\",\"floatData\":[0.5,-1,0.25,1.5,0.75,-0.5]},"
            "{\"dims\":[\"2\"],\"dataType\":1,\"name\":\"fc1.b\",\"floatData\":[0.1,-0.2]},"
            "{\"dims\":[\"2\"],\"dataType\":1,\"name\":\"bn.s\",\"floatData\":[1.25,0.5]},"
            "{\"dims\":[\"2\"],\"dataType\":1,\"name\":\"bn.b\",\"floatData\":[0.2,-0.1]},"
            "{\"dims\":[\"2\"],\"dataType\":1,\"name\":\"bn.m\",\"floatData\":[0.3,-0.4]},"
            "{\"dims\":[\"2\"],\"dataType\":1,\"name\":\"bn.v\",\"floatData\":[0.04,2.5]},"
            "{\"dims\":[\"1\",\"2\"],\"dataType\":1,\"name\":\"fc2.w\",\"floatData\":[1,-2]},"
            "{\"dims\":[\"1\"],\"dataType\":1,\"name\":\"fc2.b\",\"floatData\":[0.5]}],"
            "\"input\":[{\"name\":\"x\",\"type\":{\"tensorType\":{\"elemType\":1,\"shape\":{\"dim\":[{\"dimParam\":\"N\"},{\"dimValue\":\"3\"}]}}}}],"
            "\"output\":[{\"name\":\"y\",\"type\":{\"tensorType\":{\"elemType\":1,\"shape\":{\"dim\":[{\"dimParam\":\"N\"},{\"dimValue\":\"1\"}]}}}}]},"
            "\"opsetImport\":[{\"domain\":\"\",\"version\":\"17\"}]}";
        static const double w1[6] = { 0.5, -1, 0.25, 1.5, 0.75, -0.5 }, b1[2] = { 0.1, -0.2 };
        static const double s[2] = { 1.25, 0.5 }, bb[2] = { 0.2, -0.1 }, m[2] = { 0.3, -0.4 }, v[2] = { 0.04, 2.5 };
        FILE *f = fopen(onnx, "w");
        if (f) {
            fputs(model, f);
            fclose(f);
        }
        back = ann_import_onnx(onnx);
        int close = back != NULL && back->layer_count == 4 && ann_get_layer_kind(back, 2) == LAYER_KIND_BATCHNORM &&
                    back->layers[2].activation == ACTIVATION_RELU;
        for (int r = 0; close && r < 4; r++) {
            real in[3], out[1];
            double want = 0.5;
            for (int i = 0; i < 3; i++) in[i] = (real)sin(1.9 * (r * 3 + i) + 0.4);
            for (int c = 0; c < 2; c++) {
                double h = b1[c] + w1[c * 3] * in[0] + w1[c * 3 + 1] * in[1] + w1[c * 3 + 2] * in[2];
                double n = s[c] * (h - m[c]) / sqrt(v[c] + 1e-3) + bb[c];
                want += (c ? -2.0 : 1.0) * (n > 0.0 ? n : 0.0);
            }
            ann_predict(back, in, out);
            close &= fabs(out[0] - want) < 1e-5;
        }
        TESTEX("External model with BatchNormalization (epsilon 1e-3) imports and matches a reference", close);
        ann_free_network(back);

        // the training form (three outputs) is rejected
        char *training = NULL;
        size_t len = strlen(model);
        training = (char *)malloc(len + 64);
        if (training) {
            const char *at = strstr(model, "\"output\":[\"n\"]");
            size_t head = (size_t)(at - model);
            memcpy(training, model, head);
            strcpy(training + head, "\"output\":[\"n\",\"rm\",\"rv\"]");
            strcat(training, at + strlen("\"output\":[\"n\"]"));
            f = fopen(onnx, "w");
            if (f) {
                fputs(training, f);
                fclose(f);
            }
            free(training);
        }
        back = ann_import_onnx(onnx);
        TESTEX("BatchNormalization in training form is rejected", back == NULL);
        ann_free_network(back);
        remove(onnx);
    }
}
