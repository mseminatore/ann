#include <stdio.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#if defined(USE_CBLAS)
#	if defined(USE_MKL)
#		include <mkl_cblas.h>
#	else
#		include <cblas.h>
#	endif
#endif

static int g_errors = 0;
static void count_errors(int code, const char *message, const char *function_name)
{
    (void)code; (void)message; (void)function_name;
    g_errors++;
}

// 1 input -> 1 sigmoid output with fixed weights that predict ~1 for every
// sample, so it is right on targets of 1 and confidently wrong on targets of 0
static PNetwork make_always_one_net(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 1, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    net->layers[1].t_weights->values[0] = 0.0f;
    net->layers[1].t_bias->values[0] = 12.0f;     // sigmoid(12) ~ 0.999994
    net->weights_set = 1;                         // keep these weights
    ann_set_learning_rate(net, 1e-9f);            // and leave them unchanged
    ann_set_epoch_limit(net, 1);
    ann_set_convergence(net, 0.01f);
    return net;
}

// Can a network with this output activation be trained with cross-entropy?
static int ce_trainable(Activation_type output_act)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 2, LAYER_OUTPUT, output_act);
    int ok = (ann_train_begin(net) == ERR_OK);
    ann_train_end(net);
    ann_free_network(net);
    return ok;
}

// 3 inputs -> 4 tanh -> 5 linear outputs, MSE with the given reduction, SGD
static PNetwork make_regression_net(Loss_reduction reduction)
{
    ann_set_seed(7);
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 3, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_TANH);
    ann_add_layer(net, 5, LAYER_OUTPUT, ACTIVATION_NULL);
    ann_set_loss_reduction(net, reduction);
    return net;
}

// Does one SGD step move every parameter by -rate * factor * dL/dp, where L
// is the loss ann_evaluate_loss reports (the mean squared error over the
// batch and the outputs) and dL/dp its central-difference gradient?
// ROWS = 6 samples of OUTS = 5 outputs
static int step_matches_gradient(Loss_reduction reduction, real factor)
{
    enum { ROWS = 6, INS = 3, OUTS = 5, MAXP = 32 };
    real x[ROWS * INS], t[ROWS * OUTS];
    for (int i = 0; i < ROWS * INS; i++)
        x[i] = (real)sin(0.7 * i + 0.3);
    for (int i = 0; i < ROWS * OUTS; i++)
        t[i] = (real)cos(1.3 * i);
    PTensor inputs = tensor_create_from_array(ROWS, INS, x);
    PTensor targets = tensor_create_from_array(ROWS, OUTS, t);

    PNetwork net = make_regression_net(reduction);
    const real rate = (real)0.01, h = (real)1e-3;
    ann_set_learning_rate(net, rate);
    ann_train_begin(net);

    int ok = 1;
    real before[2][2][MAXP], grad[2][2][MAXP];
    int counts[2][2];
    for (int layer = 1; layer <= 2; layer++)
        for (int k = 0; k < 2; k++)
        {
            int rows, cols;
            ann_get_layer_param_shape(net, layer, k, &rows, &cols);
            int n = counts[layer - 1][k] = rows * cols;
            real *p = before[layer - 1][k], *g = grad[layer - 1][k];
            ann_get_layer_param(net, layer, k, p, n);
            for (int j = 0; j < n; j++)
            {
                real keep = p[j];
                p[j] = keep + h;
                ann_set_layer_param(net, layer, k, p, n);
                real up = ann_evaluate_loss(net, inputs, targets);
                p[j] = keep - h;
                ann_set_layer_param(net, layer, k, p, n);
                real down = ann_evaluate_loss(net, inputs, targets);
                p[j] = keep;
                ann_set_layer_param(net, layer, k, p, n);
                g[j] = (up - down) / (2 * h);
            }
        }

    ann_train_step(net, x, t, ROWS);
    for (int layer = 1; layer <= 2; layer++)
        for (int k = 0; k < 2; k++)
        {
            real after[MAXP];
            int n = counts[layer - 1][k];
            ann_get_layer_param(net, layer, k, after, n);
            for (int j = 0; j < n; j++)
            {
                real expected = -rate * factor * grad[layer - 1][k][j];
                real step = after[j] - before[layer - 1][k][j];
                // float central differences carry noise of a few 1e-7 per unit of rate * factor
                if (fabs(step - expected) > 5e-3 * fabs(expected) + 2e-6 * factor)
                    ok = 0;
            }
        }

    ann_train_end(net);
    ann_free_network(net);
    tensor_free(inputs);
    tensor_free(targets);
    return ok;
}

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
#endif

    MODULE("Loss Functions");

    SUITE("MSE Loss Training");
    {
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real labels[] = {0, 0, 1, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets = tensor_create_from_array(4, 1, labels);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        TESTEX("MSE network created", net != NULL);
        
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.5f);
        
        real final_loss = ann_train_network(net, inputs, targets, 4);
        
        TESTEX("MSE loss improves", final_loss < 0.5f);
        TESTEX("MSE converges", final_loss < 0.3f);
        
        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(targets);
    }

    SUITE("Cross-Entropy Loss Training");
    {
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real labels[] = {0, 1, 1, 0};
        
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets = tensor_create_from_array(4, 1, labels);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
        TESTEX("CE network created", net != NULL);
        
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 6, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.05f);
        
        real final_loss = ann_train_network(net, inputs, targets, 4);
        
        TESTEX("CE loss improves", final_loss < 1.0f);
        TESTEX("CE converges", final_loss < 0.8f);
        
        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(targets);
    }

    SUITE("Loss Computation Properties");
    {
        real simple_data[] = {1, 0};
        real simple_targets[] = {1};
        
        PTensor input = tensor_create_from_array(1, 2, simple_data);
        PTensor target = tensor_create_from_array(1, 1, simple_targets);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        
        ann_train_network(net, input, target, 1);
        real loss = ann_train_network(net, input, target, 1);
        
        TESTEX("Loss non-negative", loss >= 0.0f);
        TESTEX("Loss is finite", !isnan(loss) && !isinf(loss));
        TESTEX("Loss reasonable magnitude", loss < 10.0f);
        
        ann_free_network(net);
        tensor_free(input);
        tensor_free(target);
    }

    SUITE("Cross-Entropy Output Requirements");
    {
        // Half the targets are 0. Categorical cross-entropy (-t * log(y))
        // ignores them and reports ~0, which used to stop training as
        // "converged". Binary cross-entropy counts them.
        real x[] = {0, 1, 2, 3};
        real t[] = {1, 0, 1, 0};
        PTensor inputs = tensor_create_from_array(4, 1, x);
        PTensor targets = tensor_create_from_array(4, 1, t);
        PNetwork net = make_always_one_net();
        real loss = ann_train_network(net, inputs, targets, 4);
        TESTEX("Sigmoid output: wrong predictions on 0 targets count in the loss", loss > 1.0f);
        TESTEX("Sigmoid output: the loss stays finite", !isnan(loss) && !isinf(loss));
        ann_free_network(net);

        // A confidently wrong softmax prediction must cost the most, not nothing
        PNetwork sm = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_layer(sm, 1, LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(sm, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
        sm->layers[1].t_weights->values[0] = 0.0f;
        sm->layers[1].t_weights->values[1] = 0.0f;
        sm->layers[1].t_bias->values[0] = 30.0f;   // predicts class 0 with p ~ 1
        sm->layers[1].t_bias->values[1] = -30.0f;
        sm->weights_set = 1;
        ann_set_learning_rate(sm, 1e-9f);
        ann_set_epoch_limit(sm, 1);
        real one_hot[] = {0, 1, 0, 1, 0, 1, 0, 1};  // every sample is class 1
        PTensor sm_targets = tensor_create_from_array(4, 2, one_hot);
        loss = ann_train_network(sm, inputs, sm_targets, 4);
        TESTEX("Softmax output: a confidently wrong prediction has a large loss", loss > 10.0f);
        ann_free_network(sm);
        tensor_free(sm_targets);
        tensor_free(inputs);
        tensor_free(targets);

        TESTEX("Softmax output can be trained with cross-entropy", ce_trainable(ACTIVATION_SOFTMAX));
        TESTEX("Sigmoid output can be trained with cross-entropy", ce_trainable(ACTIVATION_SIGMOID));

        ann_set_error_log_callback(count_errors);
        g_errors = 0;
        TESTEX("Linear output is rejected", !ce_trainable(ACTIVATION_NULL));
        TESTEX("Tanh output is rejected",   !ce_trainable(ACTIVATION_TANH));
        TESTEX("ReLU output is rejected",   !ce_trainable(ACTIVATION_RELU));
        TESTEX("Each rejection reports an error", g_errors == 3);

        // ann_train_network() and ann_train_step() refuse too
        PNetwork lin = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_layer(lin, 1, LAYER_INPUT,  ACTIVATION_NULL);
        ann_add_layer(lin, 1, LAYER_OUTPUT, ACTIVATION_NULL);
        PTensor in1 = tensor_create_from_array(4, 1, x);
        PTensor out1 = tensor_create_from_array(4, 1, t);
        g_errors = 0;
        TESTEX("ann_train_network() returns 0 for a linear output",
               ann_train_network(lin, in1, out1, 4) == 0.0f);
        TESTEX("ann_train_step() returns 0 for a linear output",
               ann_train_step(lin, x, t, 1) == 0.0f);
        TESTEX("Both report an error", g_errors == 2);
        ann_clear_error_log_callback();
        tensor_free(in1);
        tensor_free(out1);
        ann_free_network(lin);
    }

    SUITE("MSE loss reduction");
    {
        PNetwork net = make_regression_net(LOSS_REDUCTION_SUM);
        TESTEX("A new network sums the outputs", ann_get_loss_reduction(net) == LOSS_REDUCTION_SUM);
        TESTEX("Setting the mean works", ann_set_loss_reduction(net, LOSS_REDUCTION_MEAN) == ERR_OK);
        TESTEX("and reads back", ann_get_loss_reduction(net) == LOSS_REDUCTION_MEAN);
        ann_set_error_log_callback(count_errors);
        g_errors = 0;
        TESTEX("An unknown reduction is rejected", ann_set_loss_reduction(net, (Loss_reduction)7) == ERR_INVALID);
        TESTEX("and reported", g_errors == 1);
        TESTEX("and leaves the setting alone", ann_get_loss_reduction(net) == LOSS_REDUCTION_MEAN);
        TESTEX("A NULL network is rejected", ann_set_loss_reduction(NULL, LOSS_REDUCTION_MEAN) == ERR_NULL_PTR);
        ann_clear_error_log_callback();
        ann_free_network(net);

        // the reported loss is the mean over the batch and the outputs; its
        // gradient is what the mean reduction trains on, and the sum
        // reduction's step is batch_size * outputs / 2 = 6 * 5 / 2 times larger
        TESTEX("Mean: an SGD step is -rate * the reported loss's gradient",
               step_matches_gradient(LOSS_REDUCTION_MEAN, (real)1.0));
        TESTEX("Sum: an SGD step is batch_size * outputs / 2 times that",
               step_matches_gradient(LOSS_REDUCTION_SUM, (real)15.0));
    }

    TESTEX("Loss function tests completed", 1);
}
