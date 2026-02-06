/**
 * @file test_optimizers.c
 * @brief Unit tests for optimizer implementations (Adam, AdaGrad, RMSProp, Momentum, SGD)
 *
 * Tests verify that each optimizer can successfully train a network on the XOR problem,
 * which requires learning non-linear decision boundaries.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

// XOR training data (embedded to avoid file I/O in tests)
static real xor_inputs[] = {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
};

static real xor_outputs[] = {
    0.0f,
    1.0f,
    1.0f,
    0.0f
};

// Suppress training output during tests
static void silent_print(const char *msg) {
    (void)msg;
}

/**
 * @brief Create and configure a network for XOR problem with specified optimizer
 */
static PNetwork create_xor_network(Optimizer_type opt) {
    PNetwork pnet = ann_make_network(opt, LOSS_MSE);
    if (!pnet) return NULL;

    // Silence training output
    pnet->print_func = silent_print;

    // XOR needs a hidden layer to learn non-linear boundary
    // Use more neurons for more reliable convergence
    ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(pnet, 16, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    // Configure for faster convergence in tests
    ann_set_convergence(pnet, 0.001f);
    pnet->epochLimit = 10000;

    return pnet;
}

/**
 * @brief Train network on XOR and return final loss
 */
static real train_xor(PNetwork pnet) {
    PTensor x_train = tensor_create_from_array(4, 2, xor_inputs);
    PTensor y_train = tensor_create_from_array(4, 1, xor_outputs);

    if (!x_train || !y_train) {
        tensor_free(x_train);
        tensor_free(y_train);
        return 999.0f;
    }

    real loss = ann_train_network(pnet, x_train, y_train, 4);

    tensor_free(x_train);
    tensor_free(y_train);

    return loss;
}

/**
 * @brief Test if network learned XOR correctly (within tolerance)
 */
static int verify_xor_predictions(PNetwork pnet, real tolerance) {
    real output[1];
    int correct = 0;

    // Test all 4 XOR cases
    for (int i = 0; i < 4; i++) {
        ann_predict(pnet, &xor_inputs[i * 2], output);
        real expected = xor_outputs[i];
        real prediction = output[0] > 0.5f ? 1.0f : 0.0f;
        
        if (fabs(prediction - expected) < tolerance) {
            correct++;
        }
    }

    return correct;
}

void test_main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    MODULE("Optimizer Tests");

    // ========================================================================
    // SGD OPTIMIZER TESTS
    // ========================================================================
    SUITE("SGD Optimizer");
    COMMENT("Testing basic stochastic gradient descent...");

    PNetwork net_sgd = create_xor_network(OPT_SGD);
    TESTEX("SGD network created", (net_sgd != NULL));

    if (net_sgd) {
        ann_set_learning_rate(net_sgd, 0.5f);
        real loss_sgd = train_xor(net_sgd);
        TESTEX("SGD training completed with finite loss", (isfinite(loss_sgd)));
        TESTEX("SGD achieved reasonable loss (<0.3)", (loss_sgd < 0.3f));
        
        ann_free_network(net_sgd);
    }

    // ========================================================================
    // MOMENTUM OPTIMIZER TESTS
    // ========================================================================
    SUITE("Momentum Optimizer");
    COMMENT("Testing SGD with momentum...");

    PNetwork net_momentum = create_xor_network(OPT_MOMENTUM);
    TESTEX("Momentum network created", (net_momentum != NULL));

    if (net_momentum) {
        ann_set_learning_rate(net_momentum, 0.1f);
        real loss_momentum = train_xor(net_momentum);
        TESTEX("Momentum training completed with finite loss", (isfinite(loss_momentum)));
        TESTEX("Momentum achieved reasonable loss (<0.3)", (loss_momentum < 0.3f));

        ann_free_network(net_momentum);
    }

    // ========================================================================
    // ADAGRAD OPTIMIZER TESTS
    // ========================================================================
    SUITE("AdaGrad Optimizer");
    COMMENT("Testing adaptive gradient optimizer...");

    PNetwork net_adagrad = create_xor_network(OPT_ADAGRAD);
    TESTEX("AdaGrad network created", (net_adagrad != NULL));

    if (net_adagrad) {
        TESTEX("AdaGrad default learning rate is 0.01",
               (fabs(net_adagrad->learning_rate - 0.01f) < 1e-6));

        // AdaGrad often needs higher initial LR since it decays aggressively
        ann_set_learning_rate(net_adagrad, 0.5f);
        real loss_adagrad = train_xor(net_adagrad);
        TESTEX("AdaGrad training completed with finite loss", (isfinite(loss_adagrad)));
        TESTEX("AdaGrad achieved reasonable loss (<0.3)", (loss_adagrad < 0.3f));

        // Verify accumulated gradient state exists for hidden layer
        TESTEX("AdaGrad velocity tensor allocated for hidden layer",
               (net_adagrad->layers[1].t_v != NULL));

        ann_free_network(net_adagrad);
    }

    // ========================================================================
    // RMSPROP OPTIMIZER TESTS
    // ========================================================================
    SUITE("RMSProp Optimizer");
    COMMENT("Testing RMSProp adaptive learning rate optimizer...");

    PNetwork net_rmsprop = create_xor_network(OPT_RMSPROP);
    TESTEX("RMSProp network created", (net_rmsprop != NULL));

    if (net_rmsprop) {
        TESTEX("RMSProp default learning rate is 0.001",
               (fabs(net_rmsprop->learning_rate - 0.001f) < 1e-6));

        ann_set_learning_rate(net_rmsprop, 0.01f);
        real loss_rmsprop = train_xor(net_rmsprop);
        TESTEX("RMSProp training completed with finite loss", (isfinite(loss_rmsprop)));
        TESTEX("RMSProp achieved reasonable loss (<0.3)", (loss_rmsprop < 0.3f));

        // Verify moving average state exists
        TESTEX("RMSProp velocity tensor allocated for hidden layer",
               (net_rmsprop->layers[1].t_v != NULL));

        ann_free_network(net_rmsprop);
    }

    // ========================================================================
    // ADAM OPTIMIZER TESTS
    // ========================================================================
    SUITE("Adam Optimizer");
    COMMENT("Testing Adam adaptive moment estimation optimizer...");

    PNetwork net_adam = create_xor_network(OPT_ADAM);
    TESTEX("Adam network created", (net_adam != NULL));

    if (net_adam) {
        TESTEX("Adam default learning rate is 0.001",
               (fabs(net_adam->learning_rate - 0.001f) < 1e-6));

        // Adam works well with its default LR
        real loss_adam = train_xor(net_adam);
        TESTEX("Adam training completed with finite loss", (isfinite(loss_adam)));
        TESTEX("Adam achieved low loss (<0.1)", (loss_adam < 0.1f));

        // Verify both momentum and velocity states exist
        TESTEX("Adam momentum tensor allocated for hidden layer",
               (net_adam->layers[1].t_m != NULL));
        TESTEX("Adam velocity tensor allocated for hidden layer",
               (net_adam->layers[1].t_v != NULL));

        // Verify bias momentum/velocity for Adam
        TESTEX("Adam bias momentum tensor allocated for hidden layer",
               (net_adam->layers[1].t_bias_m != NULL));
        TESTEX("Adam bias velocity tensor allocated for hidden layer",
               (net_adam->layers[1].t_bias_v != NULL));

        ann_free_network(net_adam);
    }

    // ========================================================================
    // OPTIMIZER COMPARISON TESTS
    // ========================================================================
    SUITE("Optimizer Comparison");
    COMMENT("Comparing optimizer convergence characteristics...");

    // Create networks with each optimizer
    PNetwork nets[4];
    Optimizer_type opts[] = {OPT_SGD, OPT_MOMENTUM, OPT_ADAGRAD, OPT_ADAM};
    const char *names[] = {"SGD", "Momentum", "AdaGrad", "Adam"};
    real losses[4];
    real learning_rates[] = {0.5f, 0.1f, 0.5f, 0.001f};

    for (int i = 0; i < 4; i++) {
        nets[i] = create_xor_network(opts[i]);
        if (nets[i]) {
            ann_set_learning_rate(nets[i], learning_rates[i]);
            losses[i] = train_xor(nets[i]);
            ann_free_network(nets[i]);
        } else {
            losses[i] = 999.0f;
        }
    }

    // All optimizers should achieve reasonable loss on XOR
    for (int i = 0; i < 4; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "%s achieved low final loss (<0.1)", names[i]);
        // Note: Using snprintf result in test would need capturing, just use hardcoded
    }
    TESTEX("SGD achieved reasonable loss", (losses[0] < 0.25f));
    TESTEX("Momentum achieved reasonable loss", (losses[1] < 0.25f));
    TESTEX("AdaGrad achieved reasonable loss", (losses[2] < 0.25f));
    TESTEX("Adam achieved reasonable loss", (losses[3] < 0.25f));

    // ========================================================================
    // GRADIENT CLIPPING WITH OPTIMIZERS
    // ========================================================================
    SUITE("Gradient Clipping with Optimizers");
    COMMENT("Testing gradient clipping works with all optimizers...");

    PNetwork net_clip = create_xor_network(OPT_ADAM);
    TESTEX("Network for gradient clipping created", (net_clip != NULL));

    if (net_clip) {
        ann_set_gradient_clip(net_clip, 1.0f);
        TESTEX("Gradient clipping threshold set", 
               (fabs(net_clip->max_gradient - 1.0f) < 1e-6));

        real loss_clipped = train_xor(net_clip);
        TESTEX("Training with gradient clipping completed", (isfinite(loss_clipped)));
        TESTEX("Gradient clipping still allows learning (<0.3)", (loss_clipped < 0.3f));

        ann_free_network(net_clip);
    }

    // ========================================================================
    // EDGE CASES
    // ========================================================================
    SUITE("Optimizer Edge Cases");
    COMMENT("Testing optimizer behavior with edge case configurations...");

    // Very small learning rate
    PNetwork net_small_lr = create_xor_network(OPT_ADAM);
    if (net_small_lr) {
        ann_set_learning_rate(net_small_lr, 0.0001f);
        net_small_lr->epochLimit = 100;  // Short training
        real loss = train_xor(net_small_lr);
        TESTEX("Very small LR produces finite loss", (isfinite(loss)));
        ann_free_network(net_small_lr);
    }

    // Large learning rate (should still not explode with Adam)
    PNetwork net_large_lr = create_xor_network(OPT_ADAM);
    if (net_large_lr) {
        ann_set_learning_rate(net_large_lr, 0.1f);
        real loss = train_xor(net_large_lr);
        TESTEX("Large LR with Adam still converges", (isfinite(loss) && loss < 1.0f));
        ann_free_network(net_large_lr);
    }

    TESTEX("Optimizer tests completed", 1);
}
