//-----------------------------------------------------------
// Convolution, max pooling and flatten layers.
//
// Shapes and argument checks, forward passes against naive reference loops,
// finite-difference gradient checks of every parameter (which also checks the
// gradient each layer passes down), training, and save/load.
//
// Gradient checks follow test_gradients.c: the analytic gradient is read from
// one SGD step (g = (W_after - W_before) / lr, which should equal -dL/dW). The
// step uses a batch of two samples, so the loss is the sum of both samples'
// losses, MSE as 1/2 * sum((t - y)^2) or cross-entropy -sum(t * log(y)).
//-----------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

#define GC_LR    0.01f
#define GC_EPS   1e-3f
#define GC_ABS   1e-3f   // absolute tolerance
#define GC_REL   2e-2f   // relative tolerance
#define GC_BATCH 2
#define MAX_IN   72
#define MAX_OUT  4

static void quiet(const char *s) { (void)s; }

static real g_x[GC_BATCH * MAX_IN];
static real g_t[GC_BATCH * MAX_OUT];

// Fill every parameter with fixed, varied values and inputs with a pattern
static void set_params(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++)
                t->values[i] = 0.5f * (real)sin(1.7 * i + 0.9 * l + 2.3 * p + 0.4);
        }
}

static void set_data(PNetwork net, int softmax_targets)
{
    int in = net->layers[0].node_count, out = net->layers[net->layer_count - 1].node_count;
    for (int i = 0; i < GC_BATCH * in; i++)
        g_x[i] = (real)sin(0.77 * i + 0.3) * 0.9f;
    for (int b = 0; b < GC_BATCH; b++)
        for (int k = 0; k < out; k++)
            g_t[b * out + k] = softmax_targets ? (k == b % out ? 0.7f : 0.3f / (out - 1))
                                               : 0.4f * (real)cos(1.3 * k + b);
}

// Sum over the batch of each sample's loss
static double batch_loss(PNetwork net)
{
    int in = net->layers[0].node_count, out = net->layers[net->layer_count - 1].node_count;
    double loss = 0.0;
    real y[MAX_OUT];
    for (int b = 0; b < GC_BATCH; b++) {
        ann_predict(net, &g_x[b * in], y);
        for (int k = 0; k < out; k++) {
            real t = g_t[b * out + k];
            if (net->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY)
                loss -= t * log(y[k]);
            else
                loss += 0.5 * (t - y[k]) * (t - y[k]);
        }
    }
    return loss;
}

// Largest scaled error between analytic and numeric gradients over every
// parameter of every layer; <= 1 when all are within tolerance. Frees net.
static real grad_check(PNetwork net, int softmax_targets)
{
    if (!net) return 1e9f;
    net->print_func = quiet;
    ann_set_learning_rate(net, GC_LR);
    ann_train_begin(net);
    set_params(net);
    set_data(net, softmax_targets);

    static real numeric[4096], before[4096];
    int n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++, n++) {
                real saved = t->values[i];
                t->values[i] = saved + GC_EPS;
                double lp = batch_loss(net);
                t->values[i] = saved - GC_EPS;
                double lm = batch_loss(net);
                t->values[i] = saved;
                numeric[n] = (real)(-(lp - lm) / (2.0 * GC_EPS));
                before[n] = saved;
            }
        }

    ann_train_step(net, g_x, g_t, GC_BATCH);

    real worst = 0.0f;
    n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
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

// 2x5x5 -> conv(3 filters, k3, padding 1, act) -> flatten -> dense(2, linear), MSE
static PNetwork net_conv_flatten(Activation_type act)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_input(net, 2, 5, 5);
    ann_add_conv2d(net, 3, 3, 1, 1, act);
    ann_add_flatten(net);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
    return net;
}

// 1x6x6 -> conv(2, k3, stride 2, padding 1, sigmoid) -> conv(3, k2, gelu) -> dense(2), MSE
static PNetwork net_conv_conv(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_input(net, 1, 6, 6);
    ann_add_conv2d(net, 2, 3, 2, 1, ACTIVATION_SIGMOID);
    ann_add_conv2d(net, 3, 2, 1, 0, ACTIVATION_GELU);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
    return net;
}

// 1x6x6 -> conv(2, k3, tanh) -> maxpool(pool, stride) -> flatten -> dense(3, softmax), cross-entropy
static PNetwork net_conv_pool(int pool, int stride)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_input(net, 1, 6, 6);
    ann_add_conv2d(net, 2, 3, 1, 0, ACTIVATION_TANH);
    ann_add_maxpool2d(net, pool, stride);
    ann_add_flatten(net);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    return net;
}

// Reference convolution of one sample: out[f][oh][ow] for the conv layer's weights
static double ref_conv(const Layer *l, const real *x, int f, int oh, int ow)
{
    int k = l->config[0], s = l->config[1], p = l->config[2];
    const real *w = l->params[0].value->values;
    double sum = l->params[1].value->values[f];
    for (int c = 0; c < l->in_shape.c; c++)
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++) {
                int ih = oh * s - p + i, iw = ow * s - p + j;
                if (ih < 0 || ih >= l->in_shape.h || iw < 0 || iw >= l->in_shape.w) continue;
                sum += w[((f * l->in_shape.c + c) * k + i) * k + j] * x[(c * l->in_shape.h + ih) * l->in_shape.w + iw];
            }
    return sum;
}

// 1x6x6 images of a horizontal (class 0) or vertical (class 1) bar
static void make_bars(real *x, real *y, int count)
{
    for (int n = 0; n < count; n++) {
        int vertical = n % 2, pos = (n / 2) % 6;
        for (int i = 0; i < 36; i++)
            x[n * 36 + i] = 0.05f * (real)sin(3.1 * i + n);   // a little noise
        for (int k = 0; k < 6; k++)
            x[n * 36 + (vertical ? k * 6 + pos : pos * 6 + k)] = 1.0f;
        y[n * 2 + 0] = (real)!vertical;
        y[n * 2 + 1] = (real)vertical;
    }
}

static PNetwork make_bar_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_input(net, 1, 6, 6);
    ann_add_conv2d(net, 4, 3, 1, 1, ACTIVATION_RELU);
    ann_add_maxpool2d(net, 2, 2);
    ann_add_flatten(net);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 8);
    return net;
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

// Largest difference between two networks' predictions on the bar images
static real max_pred_diff(PNetwork a, PNetwork b, const real *x, int count)
{
    real worst = 0.0f, ya[2], yb[2];
    for (int n = 0; n < count; n++) {
        ann_predict(a, &x[n * 36], ya);
        ann_predict(b, &x[n * 36], yb);
        for (int k = 0; k < 2; k++)
            if (fabs(ya[k] - yb[k]) > worst) worst = (real)fabs(ya[k] - yb[k]);
    }
    return worst;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Convolution, Pooling and Flatten Layers");

    SUITE("Shapes");
    {
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        TESTEX("Shaped input layer", ann_add_input(net, 1, 6, 6) == ERR_OK && net->layers[0].node_count == 36);
        TESTEX("Conv k3 s1 p0: 3x4x4", ann_add_conv2d(net, 3, 3, 1, 0, ACTIVATION_RELU) == ERR_OK &&
               net->layers[1].shape.c == 3 && net->layers[1].shape.h == 4 && net->layers[1].shape.w == 4 &&
               net->layers[1].node_count == 48 && ann_get_layer_kind(net, 1) == LAYER_KIND_CONV2D);
        TESTEX("Conv weights are [filters x C*k*k] and bias [1 x filters]",
               net->layers[1].param_count == 2 && net->layers[1].params[0].value->rows == 3 &&
               net->layers[1].params[0].value->cols == 9 && net->layers[1].params[1].value->cols == 3);
        TESTEX("MaxPool 2/2: 3x2x2", ann_add_maxpool2d(net, 2, 2) == ERR_OK &&
               net->layers[2].shape.c == 3 && net->layers[2].shape.h == 2 && net->layers[2].shape.w == 2 &&
               net->layers[2].param_count == 0);
        TESTEX("Flatten: 12x1x1", ann_add_flatten(net) == ERR_OK && net->layers[3].shape.c == 12 &&
               net->layers[3].shape.h == 1 && ann_get_layer_kind(net, 3) == LAYER_KIND_FLATTEN);
        TESTEX("Dense after flatten reads 12 inputs", ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX) == ERR_OK &&
               net->layers[4].t_weights->cols == 12);
        ann_free_network(net);

        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_input(net, 2, 6, 6);
        TESTEX("Conv k3 s2 p1: (6 + 2 - 3) / 2 + 1 = 3", ann_add_conv2d(net, 4, 3, 2, 1, ACTIVATION_TANH) == ERR_OK &&
               net->layers[1].shape.h == 3 && net->layers[1].shape.w == 3 && net->layers[1].params[0].value->cols == 18);
        TESTEX("MaxPool 3/1 overlapping windows: 1x1", ann_add_maxpool2d(net, 3, 1) == ERR_OK &&
               net->layers[2].shape.h == 1 && net->layers[2].shape.c == 4);
        TESTEX("Dense directly after pooling (no flatten)", ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL) == ERR_OK &&
               net->layers[3].t_weights->cols == 4);
        ann_free_network(net);
    }

    SUITE("Invalid Layers");
    {
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        TESTEX("Conv as the first layer is rejected", ann_add_conv2d(net, 2, 3, 1, 0, ACTIVATION_RELU) == ERR_INVALID);
        TESTEX("Pool as the first layer is rejected", ann_add_maxpool2d(net, 2, 2) == ERR_INVALID);
        TESTEX("Flatten as the first layer is rejected", ann_add_flatten(net) == ERR_INVALID);
        TESTEX("Zero-size input is rejected", ann_add_input(net, 1, 0, 4) == ERR_INVALID);
        TESTEX("Nothing was added", net->layer_count == 0);

        ann_add_input(net, 1, 4, 4);
        TESTEX("A second input layer is rejected", ann_add_input(net, 1, 4, 4) == ERR_INVALID);
        TESTEX("Kernel larger than the padded input", ann_add_conv2d(net, 2, 7, 1, 1, ACTIVATION_RELU) == ERR_INVALID);
        TESTEX("Zero filters", ann_add_conv2d(net, 0, 3, 1, 0, ACTIVATION_RELU) == ERR_INVALID);
        TESTEX("Zero stride", ann_add_conv2d(net, 2, 3, 0, 0, ACTIVATION_RELU) == ERR_INVALID);
        TESTEX("Negative padding", ann_add_conv2d(net, 2, 3, 1, -1, ACTIVATION_RELU) == ERR_INVALID);
        TESTEX("Softmax on a conv layer", ann_add_conv2d(net, 2, 3, 1, 0, ACTIVATION_SOFTMAX) == ERR_INVALID);
        TESTEX("Pool window larger than the input", ann_add_maxpool2d(net, 5, 1) == ERR_INVALID);
        TESTEX("Rejected layers are not counted", net->layer_count == 1);
        TESTEX("A valid layer still adds after failures", ann_add_conv2d(net, 2, 3, 1, 0, ACTIVATION_RELU) == ERR_OK &&
               net->layer_count == 2 && net->layers[1].shape.h == 2);
        ann_free_network(net);
    }

    SUITE("Forward Pass");
    {
        // 2x5x5 input, conv(3, k3, stride 2, padding 1, linear) -> 3x3x3, then maxpool(2, 1) -> 3x2x2
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 2, 5, 5);
        ann_add_conv2d(net, 3, 3, 2, 1, ACTIVATION_NULL);
        ann_add_maxpool2d(net, 2, 1);
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
        ann_train_begin(net);
        set_params(net);
        ann_train_end(net);

        real x[50], y[2];
        for (int i = 0; i < 50; i++) x[i] = (real)cos(0.37 * i) * 1.5f;
        ann_predict(net, x, y);

        const Layer *conv = &net->layers[1];
        double worst = 0.0;
        for (int f = 0; f < 3; f++)
            for (int oh = 0; oh < 3; oh++)
                for (int ow = 0; ow < 3; ow++) {
                    double d = fabs(conv->t_values->values[(f * 3 + oh) * 3 + ow] - ref_conv(conv, x, f, oh, ow));
                    if (d > worst) worst = d;
                }
        TESTEX("Conv output matches a direct convolution (stride 2, padding 1, 2 channels)", worst < 1e-5);

        worst = 0.0;
        const real *a = conv->t_values->values, *pooled = net->layers[2].t_values->values;
        for (int c = 0; c < 3; c++)
            for (int oh = 0; oh < 2; oh++)
                for (int ow = 0; ow < 2; ow++) {
                    real m = a[(c * 3 + oh) * 3 + ow];
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            if (a[(c * 3 + oh + i) * 3 + ow + j] > m) m = a[(c * 3 + oh + i) * 3 + ow + j];
                    if (fabs(pooled[(c * 2 + oh) * 2 + ow] - m) > worst) worst = fabs(pooled[(c * 2 + oh) * 2 + ow] - m);
                }
        TESTEX("MaxPool output is each window's maximum", worst == 0.0);

        // Batched prediction agrees with single-sample prediction
        real xb[100], yb[4], y1[2];
        for (int i = 0; i < 100; i++) xb[i] = (real)sin(0.53 * i);
        ann_predict_batch(net, xb, yb, 2);
        int same = 1;
        for (int b = 0; b < 2; b++) {
            ann_predict(net, &xb[b * 50], y1);
            for (int k = 0; k < 2; k++)
                if (fabs(y1[k] - yb[b * 2 + k]) > 1e-5) same = 0;
        }
        TESTEX("ann_predict_batch matches ann_predict", same);
        ann_free_network(net);
    }

    SUITE("Gradients");
    TESTEX("Conv (tanh) -> flatten -> dense",             grad_check(net_conv_flatten(ACTIVATION_TANH), 0) <= 1.0f);
    TESTEX("Conv (linear) -> flatten -> dense",           grad_check(net_conv_flatten(ACTIVATION_NULL), 0) <= 1.0f);
    TESTEX("Conv (softsign) -> flatten -> dense",         grad_check(net_conv_flatten(ACTIVATION_SOFTSIGN), 0) <= 1.0f);
    TESTEX("Conv stride 2 -> conv (propagates through conv)", grad_check(net_conv_conv(), 0) <= 1.0f);
    TESTEX("Conv -> maxpool 2/2 -> dense softmax (cross-entropy)", grad_check(net_conv_pool(2, 2), 1) <= 1.0f);
    TESTEX("Conv -> overlapping maxpool 3/1 -> dense softmax", grad_check(net_conv_pool(3, 1), 1) <= 1.0f);

    SUITE("Training");
    {
        enum { N = 24 };
        real x[N * 36], y[N * 2];
        make_bars(x, y, N);
        PTensor tx = tensor_create_from_array(N, 36, x);
        PTensor ty = tensor_create_from_array(N, 2, y);

        srand(42);
        PNetwork net = make_bar_net();
        ann_set_epoch_limit(net, 60);
        ann_train_network(net, tx, ty, N);
        TESTEX("Learns horizontal vs vertical bars", ann_evaluate_accuracy(net, tx, ty) == 1.0f);
        TESTEX("Trained loss is small", ann_evaluate_loss(net, tx, ty) < 0.1f);

        // Save and load (format v2 keeps the layer kinds)
        TESTEX("Text save succeeds", ann_save_network(net, "test_conv.nna") == ERR_OK);
        TESTEX("Binary save succeeds", ann_save_network_binary(net, "test_conv.nnb") == ERR_OK);
        TESTEX("Saved as format v2", file_version("test_conv.nna", 0) == 2 && file_version("test_conv.nnb", 1) == 2);
        PNetwork text = ann_load_network("test_conv.nna");
        PNetwork bin = ann_load_network_binary("test_conv.nnb");
        TESTEX("Loaded networks keep the layer kinds", text && bin &&
               ann_get_layer_kind(bin, 1) == LAYER_KIND_CONV2D && ann_get_layer_kind(bin, 2) == LAYER_KIND_MAXPOOL2D &&
               ann_get_layer_kind(bin, 3) == LAYER_KIND_FLATTEN && bin->layers[0].shape.h == 6);
        TESTEX("Binary load predicts identically", bin && max_pred_diff(net, bin, x, N) == 0.0f);
        TESTEX("Text load predicts the same (6 decimals)", text && max_pred_diff(net, text, x, N) < 1e-4f);
        ann_free_network(text);
        ann_free_network(bin);
        remove("test_conv.nna");
        remove("test_conv.nnb");

        // A shaped input alone forces v2, so the shape survives
        PNetwork shaped = ann_make_network(OPT_ADAM, LOSS_MSE);
        shaped->print_func = quiet;
        ann_add_input(shaped, 1, 6, 6);
        ann_add_layer(shaped, 2, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_save_network(shaped, "test_conv.nna");
        PNetwork back = ann_load_network("test_conv.nna");
        TESTEX("Dense network with a shaped input saves as v2 and keeps its shape",
               file_version("test_conv.nna", 0) == 2 && back && back->layers[0].shape.c == 1 &&
               back->layers[0].shape.h == 6 && back->layers[0].shape.w == 6);
        ann_free_network(back);
        ann_free_network(shaped);
        remove("test_conv.nna");

        // Online training continues from the trained weights
        real before = ann_evaluate_loss(net, tx, ty);
        ann_train_begin(net);
        for (int step = 0; step < 20; step++)
            ann_train_step(net, &x[(step % (N - 4)) * 36], &y[(step % (N - 4)) * 2], 4);
        ann_train_end(net);
        TESTEX("Online training keeps a low loss", ann_evaluate_loss(net, tx, ty) < before + 0.05f);
        ann_free_network(net);

        // Dropout on the conv layer, set by the default; pool and flatten are skipped
        srand(7);
        net = make_bar_net();
        ann_set_dropout(net, 0.2f);
        TESTEX("Default dropout applies to conv but not to pool/flatten",
               net->layers[1].dropout_rate > 0.0f && net->layers[2].dropout_rate == 0.0f &&
               net->layers[3].dropout_rate == 0.0f);
        ann_set_epoch_limit(net, 60);
        ann_train_network(net, tx, ty, N);
        TESTEX("Trains with dropout on the conv layer", ann_evaluate_accuracy(net, tx, ty) >= 0.9f);
        ann_free_network(net);

        // Early stopping restores the best epoch's weights: validate on flipped labels
        real flipped[N * 2];
        for (int i = 0; i < N * 2; i++) flipped[i] = 1.0f - y[i];
        PTensor tv = tensor_create_from_array(N, 2, flipped);
        srand(3);
        net = make_bar_net();
        ann_set_epoch_limit(net, 200);
        ann_set_validation_data(net, tx, tv);
        ann_set_early_stopping(net, 3, 0.0f, 1);
        ann_train_network(net, tx, ty, N);
        unsigned best = ann_get_best_epoch(net);
        TESTEX("Early stopping stops a conv network", net->history_count < 200 && best >= 1);
        TESTEX("and restores its best weights",
               fabs(ann_evaluate_loss(net, tx, tv) - net->val_loss_history[best - 1]) < 1e-5);
        ann_free_network(net);
        tensor_free(tv);

        tensor_free(tx);
        tensor_free(ty);
    }
}
