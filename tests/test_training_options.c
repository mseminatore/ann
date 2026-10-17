//-----------------------------------------------------------
// Training options: learning-rate schedulers, weight initialization,
// per-layer dropout, training mode, and the confusion matrix printout.
//-----------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

static int near(double a, double b, double tol)
{
    return fabs(a - b) <= tol;
}

// Mean and standard deviation of a tensor's values
static void tensor_stats(PTensor t, double *mean, double *std)
{
    int n = t->rows * t->cols;
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < n; i++) sum += t->values[i];
    *mean = sum / n;
    for (int i = 0; i < n; i++) sq += (t->values[i] - *mean) * (t->values[i] - *mean);
    *std = sqrt(sq / n);
}

// 200 -> 100 (hidden_act) -> 2, initialized with the given scheme
static PNetwork make_init_net(Weight_init_type init, Activation_type hidden_act)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 200, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 100, LAYER_HIDDEN, hidden_act);
    ann_add_layer(net, 2,   LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_weight_init(net, init);
    srand(11);
    ann_train_begin(net);   // initializes the weights
    ann_train_end(net);
    return net;
}

static int all_zero(PTensor t)
{
    for (int i = 0; i < t->rows * t->cols; i++)
        if (t->values[i] != 0.0f) return 0;
    return 1;
}

// Captures everything a network prints
static char g_printed[4096];
static void capture_print(const char *msg)
{
    size_t used = strlen(g_printed);
    if (used + strlen(msg) < sizeof(g_printed))
        strcat(g_printed, msg);
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Training Options");

    // ========================================================================
    SUITE("Learning Rate Schedulers");

    LRStepParams step = { 3, 0.5f };
    TESTEX("Step: epochs 1-3 run at the base rate",
           near(ann_lr_scheduler_step(1, 0.1f, &step), 0.1, 1e-7) &&
           near(ann_lr_scheduler_step(3, 0.1f, &step), 0.1, 1e-7));
    TESTEX("Step: epochs 4-6 are scaled by gamma", near(ann_lr_scheduler_step(4, 0.1f, &step), 0.05, 1e-7));
    TESTEX("Step: epoch 7 is scaled by gamma^2",   near(ann_lr_scheduler_step(7, 0.1f, &step), 0.025, 1e-7));

    LRExponentialParams expo = { 0.9f };
    TESTEX("Exponential: epoch 1 runs at the base rate", near(ann_lr_scheduler_exponential(1, 0.2f, &expo), 0.2, 1e-7));
    TESTEX("Exponential: epoch 3 is base * gamma^2",     near(ann_lr_scheduler_exponential(3, 0.2f, &expo), 0.162, 1e-6));

    LRCosineParams cosp = { 10, 0.001f };
    TESTEX("Cosine: halfway is the midpoint of base and min",
           near(ann_lr_scheduler_cosine(5, 0.1f, &cosp), 0.0505, 1e-6));
    TESTEX("Cosine: epoch T_max reaches min_lr", near(ann_lr_scheduler_cosine(10, 0.1f, &cosp), 0.001, 1e-6));
    TESTEX("Cosine: epochs past T_max stay at min_lr", near(ann_lr_scheduler_cosine(25, 0.1f, &cosp), 0.001, 1e-6));

    TESTEX("NULL parameters return the base rate",
           near(ann_lr_scheduler_step(5, 0.1f, NULL), 0.1, 1e-7) &&
           near(ann_lr_scheduler_exponential(5, 0.1f, NULL), 0.1, 1e-7) &&
           near(ann_lr_scheduler_cosine(5, 0.1f, NULL), 0.1, 1e-7));
    LRStepParams step0 = { 0, 0.5f };
    LRCosineParams cos0 = { 0, 0.001f };
    TESTEX("Step size 0 returns the base rate (no division by zero)",
           near(ann_lr_scheduler_step(5, 0.1f, &step0), 0.1, 1e-7));
    TESTEX("Cosine T_max 0 returns the base rate (no NaN)",
           near(ann_lr_scheduler_cosine(5, 0.1f, &cos0), 0.1, 1e-7));

    // Training applies the scheduler once per epoch, from the base rate
    {
        real xin[] = {0, 0, 0, 1, 1, 0, 1, 1}, xout[] = {0, 1, 1, 0};
        PTensor tin = tensor_create_from_array(4, 2, xin), tout = tensor_create_from_array(4, 1, xout);
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.2f);
        ann_set_convergence(net, 1e-9f);
        ann_set_epoch_limit(net, 7);
        net->print_func = capture_print;
        ann_set_lr_scheduler(net, ann_lr_scheduler_step, &step);
        ann_train_network(net, tin, tout, 4);

        int ok = (net->history_count == 7);
        for (unsigned e = 0; ok && e < net->history_count; e++)
            ok = near(net->lr_history[e], ann_lr_scheduler_step(e + 1, 0.2f, &step), 1e-7);
        TESTEX("ann_train_network() uses the scheduled rate every epoch", ok);
        TESTEX("The final learning rate is the last scheduled value", near(net->learning_rate, 0.05, 1e-7));

        ann_free_network(net);
        tensor_free(tin);
        tensor_free(tout);
    }

    // ========================================================================
    SUITE("Weight Initialization");
    {
        double mean, std;

        PNetwork net = make_init_net(WEIGHT_INIT_HE, ACTIVATION_SIGMOID);
        tensor_stats(net->layers[0].t_weights, &mean, &std);
        TESTEX("He: zero mean", near(mean, 0.0, 0.01));
        TESTEX("He: std = sqrt(2 / fan_in)", near(std, sqrt(2.0 / 200), 0.005));
        TESTEX("He: biases start at zero", all_zero(net->layers[0].t_bias));
        ann_free_network(net);

        net = make_init_net(WEIGHT_INIT_XAVIER, ACTIVATION_RELU);
        tensor_stats(net->layers[0].t_weights, &mean, &std);
        TESTEX("Xavier: std = sqrt(2 / (fan_in + fan_out))", near(std, sqrt(2.0 / 300), 0.004));
        TESTEX("Xavier: biases start at zero", all_zero(net->layers[0].t_bias));
        ann_free_network(net);

        net = make_init_net(WEIGHT_INIT_UNIFORM, ACTIVATION_RELU);
        real limit = net->weight_limit;
        PTensor w = net->layers[0].t_weights;
        int in_range = 1;
        for (int i = 0; i < w->rows * w->cols; i++)
            if (fabs(w->values[i]) > limit) in_range = 0;
        tensor_stats(w, &mean, &std);
        TESTEX("Uniform: weights within [-limit, limit]", in_range);
        TESTEX("Uniform: std = limit / sqrt(3)", near(std, limit / sqrt(3.0), 0.05 * limit));
        ann_free_network(net);

        net = make_init_net(WEIGHT_INIT_AUTO, ACTIVATION_RELU);
        tensor_stats(net->layers[0].t_weights, &mean, &std);
        TESTEX("Auto picks He for ReLU", near(std, sqrt(2.0 / 200), 0.005));
        ann_free_network(net);

        net = make_init_net(WEIGHT_INIT_AUTO, ACTIVATION_TANH);
        tensor_stats(net->layers[0].t_weights, &mean, &std);
        TESTEX("Auto picks Xavier for tanh", near(std, sqrt(2.0 / 300), 0.004));
        ann_free_network(net);
    }

    // ========================================================================
    SUITE("Per-Layer Dropout");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 4,  LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(net, 50, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 50, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 2,  LAYER_OUTPUT, ACTIVATION_SIGMOID);

        ann_set_layer_dropout(net, 1, 0.5f);
        TESTEX("Sets the rate on a hidden layer", near(net->layers[1].dropout_rate, 0.5, 1e-7));
        TESTEX("Leaves other hidden layers alone", (net->layers[2].dropout_rate == 0.0f));
        ann_set_layer_dropout(net, 0, 0.5f);
        ann_set_layer_dropout(net, 3, 0.5f);
        TESTEX("Ignores the input and output layers",
               (net->layers[0].dropout_rate == 0.0f && net->layers[3].dropout_rate == 0.0f));
        ann_set_layer_dropout(net, 2, 1.5f);
        TESTEX("Clamps rates of 1 or more below 1", (net->layers[2].dropout_rate < 1.0f));
        ann_set_layer_dropout(net, 2, -0.3f);
        TESTEX("Clamps negative rates to 0", (net->layers[2].dropout_rate == 0.0f));
        ann_set_layer_dropout(net, 99, 0.5f);   // out of range: no effect, no crash

        // One training step: only layer 1 drops neurons, at about the set rate
        real in[64 * 4], target[64 * 2];
        for (int i = 0; i < 64 * 4; i++) in[i] = (real)((i * 7) % 10) / 10.0f;
        for (int i = 0; i < 64 * 2; i++) target[i] = (real)(i % 2);
        srand(5);
        ann_train_begin(net);
        ann_train_step(net, in, target, 64);
        ann_train_end(net);

        PTensor mask = net->layers[1].t_dropout_mask;
        int dropped = 0;
        for (int i = 0; mask && i < mask->rows * mask->cols; i++)
            if (mask->values[i] == 0.0f) dropped++;
        TESTEX("The dropout layer drops about half its neurons",
               (mask != NULL && near((double)dropped / (64 * 50), 0.5, 0.05)));
        TESTEX("Kept neurons are scaled by 1 / (1 - rate)",
               (mask != NULL && (mask->values[0] == 0.0f || near(mask->values[0], 2.0, 1e-6))));
        TESTEX("Layers without dropout get no mask", (net->layers[2].t_dropout_mask == NULL));
        ann_free_network(net);
    }

    // ========================================================================
    SUITE("Training Mode");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 3,  LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(net, 20, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 1,  LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_dropout(net, 0.5f);
        ann_train_begin(net);
        ann_train_end(net);

        TESTEX("Networks start in inference mode", (net->is_training == 0));
        ann_set_training_mode(net, 1);
        TESTEX("ann_set_training_mode(1) enables training mode", (net->is_training == 1));

        real x[3] = {0.4f, -0.2f, 0.9f}, y1[1], y2[1];
        ann_predict(net, x, y1);
        ann_predict(net, x, y2);
        TESTEX("Predictions in training mode are deterministic (no dropout)", (y1[0] == y2[0]));
        TESTEX("ann_predict() restores training mode", (net->is_training == 1));

        ann_set_training_mode(net, 0);
        TESTEX("ann_set_training_mode(0) returns to inference mode", (net->is_training == 0));
        ann_free_network(net);
    }

    // ========================================================================
    SUITE("Confusion Matrix Printout");
    {
        // One input, two linear outputs (-x, x): predicts class 1 when x > 0
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 1, LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_NULL);
        net->layers[0].t_weights->values[0] = -1.0f;
        net->layers[0].t_weights->values[1] =  1.0f;
        net->layers[0].t_bias->values[0] = 0.0f;
        net->layers[0].t_bias->values[1] = 0.0f;
        net->print_func = capture_print;

        // TP: x=2, x=1   TN: x=-1   FN: x=-2   FP: x=3
        real xs[] = {2, 1, -1, -2, 3};
        real labels[] = {0, 1,  0, 1,  1, 0,  0, 1,  1, 0};
        PTensor in = tensor_create_from_array(5, 1, xs);
        PTensor out = tensor_create_from_array(5, 2, labels);

        g_printed[0] = '\0';
        ann_print_confusion_matrix(net, in, out);
        TESTEX("Prints TP and FN for the positive row", (strstr(g_printed, "    2       1") != NULL));
        TESTEX("Prints FP and TN for the negative row", (strstr(g_printed, "    1       1") != NULL));
        TESTEX("Prints the MCC (1/6)", (strstr(g_printed, "0.1667") != NULL));

        // Wrong label shape: the counts are rejected and print as zero
        PTensor bad = tensor_create_from_array(5, 1, xs);
        g_printed[0] = '\0';
        ann_print_confusion_matrix(net, in, bad);
        TESTEX("Rejected input prints zero counts", (strstr(g_printed, "    0       0") != NULL));

        tensor_free(bad);
        tensor_free(in);
        tensor_free(out);
        ann_free_network(net);
    }
}
