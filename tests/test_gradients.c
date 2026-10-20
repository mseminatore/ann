//-----------------------------------------------------------
// Gradient checks for backpropagation.
//
// The gradient computed by backprop is compared with a central finite
// difference of the loss, for every hidden-layer activation and every output
// activation. The network is 3 -> 4 (hidden) -> 3 (hidden) -> 2 (output), so
// the check covers the activation derivatives, propagation through two hidden
// layers, and all weights and biases.
//
// Losses: MSE is L = 1/2 * sum((t - y)^2). Cross-entropy is L = -sum(t * log(y))
// with a softmax output and targets that sum to 1, or binary cross-entropy
// L = -sum(t * log(y) + (1 - t) * log(1 - y)) with a sigmoid output.
//
// The analytic gradient is read from one SGD step: W += lr * g, so
// g = (W_after - W_before) / lr, and g should equal -dL/dW.
//-----------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

#define GC_LR    0.01f
#define GC_EPS   1e-2f
#define GC_ABS   1e-3f   // absolute tolerance
#define GC_REL   2e-2f   // relative tolerance

static const real gc_x[3] = {0.3f, -0.7f, 0.9f};
static const real gc_t[2] = {0.2f, 0.8f};   // sums to 1 for cross-entropy

static real gc_loss(PNetwork net)
{
    real y[2];
    ann_predict(net, gc_x, y);
    if (net->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY) {
        if (net->layers[net->layer_count - 1].activation == ACTIVATION_SIGMOID)
            return -(gc_t[0] * (real)log(y[0]) + (1 - gc_t[0]) * (real)log(1 - y[0]) +
                     gc_t[1] * (real)log(y[1]) + (1 - gc_t[1]) * (real)log(1 - y[1]));
        return -(gc_t[0] * (real)log(y[0]) + gc_t[1] * (real)log(y[1]));
    }
    real d0 = gc_t[0] - y[0], d1 = gc_t[1] - y[1];
    return 0.5f * (d0 * d0 + d1 * d1);
}

static PNetwork gc_make_net(Activation_type act, Activation_type out_act, Loss_type loss)
{
    PNetwork net = ann_make_network(OPT_SGD, loss);
    if (!net) return NULL;

    ann_add_layer(net, 3, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, act);
    ann_add_layer(net, 3, LAYER_HIDDEN, act);
    ann_add_layer(net, 2, LAYER_OUTPUT, out_act);
    ann_set_learning_rate(net, GC_LR);
    ann_set_batch_size(net, 1);

    // Initialize, then replace the random weights with fixed values chosen so
    // no pre-activation sits within GC_EPS of ReLU's kink at zero
    ann_train_begin(net);
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor w = net->layers[l + 1].t_weights, b = net->layers[l + 1].t_bias;
        for (int i = 0; i < w->rows * w->cols; i++)
            w->values[i] = 0.6f * (real)sin(1.7 * i + 0.9 * l + 0.4);
        for (int i = 0; i < b->cols; i++)
            b->values[i] = 0.3f * (real)cos(2.3 * i + 1.1 * l);
    }
    return net;
}

// Largest scaled error between analytic and numeric gradients over every
// weight and bias. Returns a value <= 1 when all are within tolerance.
static real gc_check_net(Activation_type act, Activation_type out_act, Loss_type loss)
{
    PNetwork net = gc_make_net(act, out_act, loss);
    if (!net) return 1e9f;

    // Numeric gradients: -dL/dp by central difference, for every parameter
    real numeric[64];
    int count = 0;
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor params[2] = { net->layers[l + 1].t_weights, net->layers[l + 1].t_bias };
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < params[k]->rows * params[k]->cols; i++) {
                real saved = params[k]->values[i];
                params[k]->values[i] = saved + GC_EPS;
                real lp = gc_loss(net);
                params[k]->values[i] = saved - GC_EPS;
                real lm = gc_loss(net);
                params[k]->values[i] = saved;
                numeric[count++] = -(lp - lm) / (2.0f * GC_EPS);
            }
        }
    }

    // Analytic gradients from one SGD step
    real before[64];
    int n = 0;
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor params[2] = { net->layers[l + 1].t_weights, net->layers[l + 1].t_bias };
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < params[k]->rows * params[k]->cols; i++)
                before[n++] = params[k]->values[i];
    }

    ann_train_step(net, gc_x, gc_t, 1);

    real worst = 0.0f;
    n = 0;
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor params[2] = { net->layers[l + 1].t_weights, net->layers[l + 1].t_bias };
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < params[k]->rows * params[k]->cols; i++, n++) {
                real analytic = (params[k]->values[i] - before[n]) / GC_LR;
                real err = (real)fabs(analytic - numeric[n]) /
                           (GC_ABS + GC_REL * (real)fabs(numeric[n]));
                if (err > worst) worst = err;
            }
        }
    }

    ann_train_end(net);
    ann_free_network(net);
    return worst;
}

// Hidden-layer check: MSE with a linear output
static real gc_check(Activation_type act)
{
    return gc_check_net(act, ACTIVATION_NULL, LOSS_MSE);
}

// Output-layer check: tanh hidden layers, the given output and loss
static real gc_check_output(Activation_type out_act, Loss_type loss)
{
    return gc_check_net(ACTIVATION_TANH, out_act, loss);
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Backpropagation Gradient Checks");

    SUITE("Analytic vs Finite-Difference Gradients");
    TESTEX("Sigmoid hidden layers",    (gc_check(ACTIVATION_SIGMOID)    <= 1.0f));
    TESTEX("ReLU hidden layers",       (gc_check(ACTIVATION_RELU)       <= 1.0f));
    TESTEX("Leaky ReLU hidden layers", (gc_check(ACTIVATION_LEAKY_RELU) <= 1.0f));
    TESTEX("Tanh hidden layers",       (gc_check(ACTIVATION_TANH)       <= 1.0f));
    TESTEX("Softsign hidden layers",   (gc_check(ACTIVATION_SOFTSIGN)   <= 1.0f));
    TESTEX("GELU hidden layers",       (gc_check(ACTIVATION_GELU)       <= 1.0f));

    SUITE("Output Layer Gradients");
    TESTEX("MSE with sigmoid output",    (gc_check_output(ACTIVATION_SIGMOID,    LOSS_MSE) <= 1.0f));
    TESTEX("MSE with tanh output",       (gc_check_output(ACTIVATION_TANH,       LOSS_MSE) <= 1.0f));
    TESTEX("MSE with softsign output",   (gc_check_output(ACTIVATION_SOFTSIGN,   LOSS_MSE) <= 1.0f));
    TESTEX("MSE with ReLU output",       (gc_check_output(ACTIVATION_RELU,       LOSS_MSE) <= 1.0f));
    TESTEX("MSE with leaky ReLU output", (gc_check_output(ACTIVATION_LEAKY_RELU, LOSS_MSE) <= 1.0f));
    TESTEX("MSE with GELU output",       (gc_check_output(ACTIVATION_GELU,       LOSS_MSE) <= 1.0f));
    TESTEX("MSE with linear output",     (gc_check_output(ACTIVATION_NULL,       LOSS_MSE) <= 1.0f));
    TESTEX("MSE with softmax output",    (gc_check_output(ACTIVATION_SOFTMAX,    LOSS_MSE) <= 1.0f));
    TESTEX("Cross-entropy with softmax output",
           (gc_check_output(ACTIVATION_SOFTMAX, LOSS_CATEGORICAL_CROSS_ENTROPY) <= 1.0f));
    TESTEX("Binary cross-entropy with sigmoid output",
           (gc_check_output(ACTIVATION_SIGMOID, LOSS_CATEGORICAL_CROSS_ENTROPY) <= 1.0f));
    TESTEX("LOSS_BINARY_CROSS_ENTROPY alias with sigmoid output",
           (gc_check_output(ACTIVATION_SIGMOID, LOSS_BINARY_CROSS_ENTROPY) <= 1.0f));
}
