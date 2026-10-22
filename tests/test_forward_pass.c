//-----------------------------------------------------------
// Forward pass correctness.
//
// Networks get fixed weights and their outputs are compared with a forward
// pass computed independently here, in double precision, for every
// activation. Also checks ann_predict_batch() against ann_predict().
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#define FP_TOL 1e-5

static double fp_activate(Activation_type act, double z)
{
    switch (act) {
        case ACTIVATION_SIGMOID:    return 1.0 / (1.0 + exp(-z));
        case ACTIVATION_RELU:       return z > 0.0 ? z : 0.0;
        case ACTIVATION_LEAKY_RELU: return z > 0.0 ? z : 0.01 * z;
        case ACTIVATION_TANH:       return tanh(z);
        case ACTIVATION_SOFTSIGN:   return z / (1.0 + fabs(z));
        default:                    return z;
    }
}

// 3 -> 4 (hidden_act) -> 2 (output_act) with fixed weights. Untrained
// networks have no initialized weights, so they are set explicitly.
static PNetwork fp_make_net(Activation_type hidden_act, Activation_type output_act)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 3, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, hidden_act);
    ann_add_layer(net, 2, LAYER_OUTPUT, output_act);

    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor w = net->layers[l + 1].t_weights, b = net->layers[l + 1].t_bias;
        for (int i = 0; i < w->rows * w->cols; i++)
            w->values[i] = 0.8f * (real)sin(1.3 * i + l);
        for (int i = 0; i < b->cols; i++)
            b->values[i] = 0.2f * (real)cos(0.7 * i + l);
    }
    return net;
}

// Reference forward pass: y = act(W x + b) per layer, softmax on the output
static void fp_reference(PNetwork net, const real *x, double *out)
{
    double a[8], next[8];
    int n = net->layers[0].node_count;
    for (int i = 0; i < n; i++) a[i] = x[i];

    for (int l = 0; l < net->layer_count - 1; l++) {
        PTensor w = net->layers[l + 1].t_weights, b = net->layers[l + 1].t_bias;
        Activation_type act = net->layers[l + 1].activation;
        int m = net->layers[l + 1].node_count;

        for (int r = 0; r < m; r++) {
            double z = b->values[r];
            for (int c = 0; c < n; c++)
                z += (double)w->values[r * n + c] * a[c];
            next[r] = z;
        }
        if (act == ACTIVATION_SOFTMAX) {
            double mx = next[0], sum = 0.0;
            for (int r = 1; r < m; r++) if (next[r] > mx) mx = next[r];
            for (int r = 0; r < m; r++) sum += exp(next[r] - mx);
            for (int r = 0; r < m; r++) next[r] = exp(next[r] - mx) / sum;
        } else {
            for (int r = 0; r < m; r++) next[r] = fp_activate(act, next[r]);
        }
        for (int r = 0; r < m; r++) a[r] = next[r];
        n = m;
    }
    for (int i = 0; i < n; i++) out[i] = a[i];
}

static const real fp_inputs[4][3] = {
    { 0.5f, -0.3f,  0.8f},
    {-1.2f,  0.4f,  0.0f},
    { 2.0f,  1.5f, -0.7f},
    { 0.0f,  0.0f,  0.0f},
};

// Does ann_predict match the reference for every test input?
static int fp_matches(Activation_type hidden_act, Activation_type output_act)
{
    PNetwork net = fp_make_net(hidden_act, output_act);
    if (!net) return 0;

    int ok = 1;
    for (int s = 0; s < 4; s++) {
        real y[2];
        double expect[2];
        if (ann_predict(net, fp_inputs[s], y) != ERR_OK) ok = 0;
        fp_reference(net, fp_inputs[s], expect);
        for (int i = 0; i < 2; i++)
            if (fabs(y[i] - expect[i]) > FP_TOL) ok = 0;
    }
    ann_free_network(net);
    return ok;
}

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Forward Pass Correctness");

    SUITE("ann_predict() Matches Reference Forward Pass");
    TESTEX("Sigmoid hidden layer",    fp_matches(ACTIVATION_SIGMOID,    ACTIVATION_NULL));
    TESTEX("ReLU hidden layer",       fp_matches(ACTIVATION_RELU,       ACTIVATION_NULL));
    TESTEX("Leaky ReLU hidden layer", fp_matches(ACTIVATION_LEAKY_RELU, ACTIVATION_NULL));
    TESTEX("Tanh hidden layer",       fp_matches(ACTIVATION_TANH,       ACTIVATION_NULL));
    TESTEX("Softsign hidden layer",   fp_matches(ACTIVATION_SOFTSIGN,   ACTIVATION_NULL));
    TESTEX("Sigmoid output layer",    fp_matches(ACTIVATION_RELU,       ACTIVATION_SIGMOID));
    TESTEX("Softmax output layer",    fp_matches(ACTIVATION_TANH,       ACTIVATION_SOFTMAX));

    SUITE("Softmax Output Properties");
    {
        PNetwork net = fp_make_net(ACTIVATION_RELU, ACTIVATION_SOFTMAX);
        real y[2];
        ann_predict(net, fp_inputs[2], y);
        TESTEX("Softmax outputs sum to 1", fabs(y[0] + y[1] - 1.0) < FP_TOL);
        TESTEX("Softmax outputs are probabilities",
               (y[0] > 0.0f && y[0] < 1.0f && y[1] > 0.0f && y[1] < 1.0f));
        ann_free_network(net);
    }

    SUITE("ann_predict_batch() (CPU) Matches ann_predict()");
    {
        PNetwork net = fp_make_net(ACTIVATION_TANH, ACTIVATION_SOFTMAX);
        real batch_in[4 * 3], batch_out[4 * 2];
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 3; i++)
                batch_in[s * 3 + i] = fp_inputs[s][i];

        TESTEX("ann_predict_batch() returns ERR_OK",
               (ann_predict_batch(net, batch_in, batch_out, 4) == ERR_OK));

        int same = 1;
        for (int s = 0; s < 4; s++) {
            real y[2];
            ann_predict(net, fp_inputs[s], y);
            if (fabs(y[0] - batch_out[s * 2]) > FP_TOL ||
                fabs(y[1] - batch_out[s * 2 + 1]) > FP_TOL)
                same = 0;
        }
        TESTEX("Batch outputs match single-sample outputs", same);
        ann_free_network(net);
    }

    SUITE("ann_predict_batch() (CPU) in Chunks");
    {
        // 600 rows span several chunks and end with a partial one
        enum { ROWS = 600 };
        static real big_in[ROWS * 3], big_out[ROWS * 2];
        for (int i = 0; i < ROWS * 3; i++)
            big_in[i] = (real)sin(0.37 * i);

        PNetwork net = fp_make_net(ACTIVATION_RELU, ACTIVATION_SOFTMAX);
        ann_set_layer_dropout(net, 1, 0.5f);
        TESTEX("A 600-row batch returns ERR_OK",
               (ann_predict_batch(net, big_in, big_out, ROWS) == ERR_OK));

        double worst = 0.0;
        for (int s = 0; s < ROWS; s++) {
            real y[2];
            ann_predict(net, &big_in[s * 3], y);
            for (int k = 0; k < 2; k++)
                if (fabs(y[k] - big_out[s * 2 + k]) > worst) worst = fabs(y[k] - big_out[s * 2 + k]);
        }
        TESTEX("Every row matches ann_predict() (dropout is not applied)", worst < FP_TOL);

        // During online training the batch tensors keep the training batch size
        ann_set_batch_size(net, 8);
        TESTEX("ann_train_begin() succeeds", (ann_train_begin(net) == ERR_OK));
        TESTEX("A batch predicted mid-training returns ERR_OK",
               (ann_predict_batch(net, big_in, big_out, ROWS) == ERR_OK));
        TESTEX("Mid-training prediction leaves the training batch size",
               (net->current_batch_size == 8 && net->layers[0].t_batch_values->rows == 8));

        worst = 0.0;
        for (int s = 0; s < ROWS; s++) {
            real y[2];
            ann_predict(net, &big_in[s * 3], y);
            for (int k = 0; k < 2; k++)
                if (fabs(y[k] - big_out[s * 2 + k]) > worst) worst = fabs(y[k] - big_out[s * 2 + k]);
        }
        TESTEX("Mid-training batch rows match ann_predict()", worst < FP_TOL);

        real targets[8 * 2];
        for (int i = 0; i < 8; i++) { targets[i * 2] = (real)(i & 1); targets[i * 2 + 1] = (real)!(i & 1); }
        real loss = ann_train_step(net, big_in, targets, 8);
        TESTEX("Training continues after a mid-training prediction", (loss > 0.0f && !isnan(loss)));
        ann_train_end(net);
        ann_free_network(net);
    }

    SUITE("ann_confusion_matrix() Uses Batched Prediction");
    {
        enum { ROWS = 300 };
        PTensor x = tensor_create(ROWS, 3), t = tensor_zeros(ROWS, 2);
        for (int i = 0; i < ROWS * 3; i++)
            x->values[i] = (real)sin(0.37 * i);

        PNetwork net = fp_make_net(ACTIVATION_TANH, ACTIVATION_SOFTMAX);
        int etp = 0, efp = 0, etn = 0, efn = 0;
        for (int s = 0; s < ROWS; s++) {
            real y[2];
            int actual = (s % 3) == 0;
            t->values[s * 2 + actual] = 1.0f;
            ann_predict(net, &x->values[s * 3], y);
            int predicted = y[1] > y[0];
            if (actual && predicted) etp++;
            else if (!actual && predicted) efp++;
            else if (!actual) etn++;
            else efn++;
        }

        int tp, fp, tn, fn;
        ann_confusion_matrix(net, x, t, &tp, &fp, &tn, &fn);
        TESTEX("Counts match per-sample predictions",
               (tp == etp && fp == efp && tn == etn && fn == efn));
        TESTEX("Counts cover every row", (tp + fp + tn + fn == ROWS));

        ann_free_network(net);
        tensor_free(x);
        tensor_free(t);
    }

    SUITE("Boundary Conditions");
    {
        PNetwork net = fp_make_net(ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX);
        real big[3] = {80.0f, -80.0f, 80.0f}, y[2];
        ann_predict(net, big, y);
        TESTEX("Large inputs give finite outputs",
               (!isnan(y[0]) && !isinf(y[0]) && !isnan(y[1]) && !isinf(y[1])));
        TESTEX("ann_predict(NULL net) returns ERR_NULL_PTR",
               (ann_predict(NULL, big, y) == ERR_NULL_PTR));
        ann_free_network(net);
    }
}
