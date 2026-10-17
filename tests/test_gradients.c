//-----------------------------------------------------------
// Gradient checks for backpropagation.
//
// For every hidden-layer activation, the gradient computed by backprop is
// compared with a central finite difference of the loss. The network is
// 3 -> 4 (act) -> 3 (act) -> 2 (linear), so the check covers the activation
// derivative, propagation through two hidden layers, and all weights and
// biases. The output layer is linear because backprop starts from
// delta = T - Y, which is the exact gradient of L = 1/2 * sum((t - y)^2)
// only for a linear output.
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
static const real gc_t[2] = {0.2f, -0.4f};

static real gc_loss(PNetwork net)
{
    real y[2];
    ann_predict(net, gc_x, y);
    real d0 = gc_t[0] - y[0], d1 = gc_t[1] - y[1];
    return 0.5f * (d0 * d0 + d1 * d1);
}

static PNetwork gc_make_net(Activation_type act)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 3, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, act);
    ann_add_layer(net, 3, LAYER_HIDDEN, act);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
    ann_set_learning_rate(net, GC_LR);
    ann_set_batch_size(net, 1);

    // Initialize, then replace the random weights with fixed values chosen so
    // no pre-activation sits within GC_EPS of ReLU's kink at zero
    ann_train_begin(net);
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor w = net->layers[l].t_weights, b = net->layers[l].t_bias;
        for (int i = 0; i < w->rows * w->cols; i++)
            w->values[i] = 0.6f * (real)sin(1.7 * i + 0.9 * l + 0.4);
        for (int i = 0; i < b->cols; i++)
            b->values[i] = 0.3f * (real)cos(2.3 * i + 1.1 * l);
    }
    return net;
}

// Largest scaled error between analytic and numeric gradients over every
// weight and bias. Returns a value <= 1 when all are within tolerance.
static real gc_check(Activation_type act)
{
    PNetwork net = gc_make_net(act);
    if (!net) return 1e9f;

    // Numeric gradients: -dL/dp by central difference, for every parameter
    real numeric[64];
    int count = 0;
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor params[2] = { net->layers[l].t_weights, net->layers[l].t_bias };
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
        PTensor params[2] = { net->layers[l].t_weights, net->layers[l].t_bias };
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < params[k]->rows * params[k]->cols; i++)
                before[n++] = params[k]->values[i];
    }

    ann_train_step(net, gc_x, gc_t, 1);

    real worst = 0.0f;
    n = 0;
    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor params[2] = { net->layers[l].t_weights, net->layers[l].t_bias };
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
}
