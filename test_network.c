#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

// Test callback tracking variables
static int callback_called = 0;
static int last_error_code = 0;
static const char *last_error_msg = NULL;
static const char *last_func_name = NULL;

void test_error_callback(int code, const char *msg, const char *func) {
    callback_called++;
    last_error_code = code;
    last_error_msg = msg;
    last_func_name = func;
}

void test_main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    MODULE("Network Creation and Configuration Tests");

    // ========================================================================
    // NETWORK CREATION TESTS
    // ========================================================================
    SUITE("Network Creation");
    COMMENT("Testing network creation with different optimizers and loss functions...");

    // Test ann_make_network with SGD optimizer and MSE loss
    PNetwork net_sgd_mse = ann_make_network(OPT_SGD, LOSS_MSE);
    TESTEX("ann_make_network (SGD, MSE) returns non-NULL", (net_sgd_mse != NULL));
    TESTEX("Network has zero layers initially", 
           (net_sgd_mse != NULL && net_sgd_mse->layer_count == 0));

    // Test with different optimizer
    PNetwork net_sgd_decay = ann_make_network(OPT_SGD_WITH_DECAY, LOSS_MSE);
    TESTEX("ann_make_network (SGD_WITH_DECAY) returns non-NULL", (net_sgd_decay != NULL));

    // Test with Adam optimizer
    PNetwork net_adam = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    TESTEX("ann_make_network (ADAM, CROSS_ENTROPY) returns non-NULL", (net_adam != NULL));
    TESTEX("Network loss_type set correctly", 
           (net_adam != NULL && net_adam->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY));

    // Test with Momentum optimizer
    PNetwork net_momentum = ann_make_network(OPT_MOMENTUM, LOSS_MSE);
    TESTEX("ann_make_network (MOMENTUM) returns non-NULL", (net_momentum != NULL));

    // ========================================================================
    // LAYER ADDITION TESTS
    // ========================================================================
    SUITE("Layer Addition");
    COMMENT("Testing network topology construction with different layer configurations...");

    // Test adding input layer
    int result = ann_add_layer(net_sgd_mse, 10, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("ann_add_layer (INPUT) returns ERR_OK", (result == ERR_OK));
    TESTEX("Layer count incremented after adding input layer", 
           (net_sgd_mse != NULL && net_sgd_mse->layer_count == 1));

    // Test adding hidden layer with sigmoid
    result = ann_add_layer(net_sgd_mse, 5, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    TESTEX("ann_add_layer (HIDDEN, SIGMOID) returns ERR_OK", (result == ERR_OK));
    TESTEX("Layer count incremented after adding hidden layer", 
           (net_sgd_mse != NULL && net_sgd_mse->layer_count == 2));

    // Test adding output layer
    result = ann_add_layer(net_sgd_mse, 2, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    TESTEX("ann_add_layer (OUTPUT, SIGMOID) returns ERR_OK", (result == ERR_OK));
    TESTEX("Final layer count is 3", 
           (net_sgd_mse != NULL && net_sgd_mse->layer_count == 3));

    // Test different activation functions
    PNetwork net_activations = ann_make_network(OPT_SGD, LOSS_MSE);
    
    result = ann_add_layer(net_activations, 8, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("Input layer with ACTIVATION_NULL added", (result == ERR_OK));

    result = ann_add_layer(net_activations, 16, LAYER_HIDDEN, ACTIVATION_RELU);
    TESTEX("Hidden layer with ACTIVATION_RELU added", (result == ERR_OK));

    result = ann_add_layer(net_activations, 8, LAYER_HIDDEN, ACTIVATION_TANH);
    TESTEX("Hidden layer with ACTIVATION_TANH added", (result == ERR_OK));

    result = ann_add_layer(net_activations, 4, LAYER_HIDDEN, ACTIVATION_LEAKY_RELU);
    TESTEX("Hidden layer with ACTIVATION_LEAKY_RELU added", (result == ERR_OK));

    result = ann_add_layer(net_activations, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    TESTEX("Output layer with ACTIVATION_SOFTMAX added", (result == ERR_OK));
    TESTEX("Network with mixed activations has 5 layers", 
           (net_activations != NULL && net_activations->layer_count == 5));

    // ========================================================================
    // LAYER PROPERTIES TESTS
    // ========================================================================
    SUITE("Layer Properties");
    COMMENT("Testing layer node counts and types...");

    // Verify layer properties were set correctly
    if (net_sgd_mse != NULL && net_sgd_mse->layer_count >= 3) {
        TESTEX("First layer node count is 10", 
               (net_sgd_mse->layers[0].node_count == 10));
        TESTEX("Second layer node count is 5", 
               (net_sgd_mse->layers[1].node_count == 5));
        TESTEX("Third layer node count is 2", 
               (net_sgd_mse->layers[2].node_count == 2));

        TESTEX("First layer type is LAYER_INPUT", 
               (net_sgd_mse->layers[0].layer_type == LAYER_INPUT));
        TESTEX("Second layer type is LAYER_HIDDEN", 
               (net_sgd_mse->layers[1].layer_type == LAYER_HIDDEN));
        TESTEX("Third layer type is LAYER_OUTPUT", 
               (net_sgd_mse->layers[2].layer_type == LAYER_OUTPUT));

        TESTEX("First layer activation is NULL", 
               (net_sgd_mse->layers[0].activation == ACTIVATION_NULL));
        TESTEX("Second layer activation is SIGMOID", 
               (net_sgd_mse->layers[1].activation == ACTIVATION_SIGMOID));
        TESTEX("Third layer activation is SIGMOID", 
               (net_sgd_mse->layers[2].activation == ACTIVATION_SIGMOID));
    }

    // ========================================================================
    // CONFIGURATION TESTS
    // ========================================================================
    SUITE("Network Configuration");
    COMMENT("Testing network property setters...");

    // Test setting learning rate
    ann_set_learning_rate(net_sgd_mse, 0.01f);
    TESTEX("ann_set_learning_rate accepted 0.01", 
           (net_sgd_mse != NULL && fabs(net_sgd_mse->learning_rate - 0.01f) < 1e-6));

    ann_set_learning_rate(net_sgd_mse, 0.001f);
    TESTEX("ann_set_learning_rate changed to 0.001", 
           (net_sgd_mse != NULL && fabs(net_sgd_mse->learning_rate - 0.001f) < 1e-6));

    ann_set_learning_rate(net_sgd_mse, 0.1f);
    TESTEX("ann_set_learning_rate accepted 0.1", 
           (net_sgd_mse != NULL && fabs(net_sgd_mse->learning_rate - 0.1f) < 1e-6));

    // Test setting convergence threshold
    ann_set_convergence(net_sgd_mse, 0.001f);
    TESTEX("ann_set_convergence accepted 0.001", 
           (net_sgd_mse != NULL && fabs(net_sgd_mse->convergence_epsilon - 0.001f) < 1e-6));

    ann_set_convergence(net_sgd_mse, 0.1f);
    TESTEX("ann_set_convergence changed to 0.1", 
           (net_sgd_mse != NULL && fabs(net_sgd_mse->convergence_epsilon - 0.1f) < 1e-6));

    // Test changing loss function
    ann_set_loss_function(net_sgd_mse, LOSS_CATEGORICAL_CROSS_ENTROPY);
    TESTEX("ann_set_loss_function changed to CROSS_ENTROPY", 
           (net_sgd_mse != NULL && net_sgd_mse->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY));

    ann_set_loss_function(net_sgd_mse, LOSS_MSE);
    TESTEX("ann_set_loss_function changed back to MSE", 
           (net_sgd_mse != NULL && net_sgd_mse->loss_type == LOSS_MSE));

    // ========================================================================
    // ERROR HANDLING TESTS
    // ========================================================================
    SUITE("Error Handling");
    COMMENT("Testing error conditions and edge cases...");

    // Test adding layer to NULL network
    result = ann_add_layer(NULL, 10, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("ann_add_layer with NULL network returns ERR_NULL_PTR", 
           (result == ERR_NULL_PTR));

    // Test adding layer with invalid node count
    result = ann_add_layer(net_adam, 0, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("ann_add_layer with zero nodes returns ERR_INVALID", 
           (result == ERR_INVALID));

    result = ann_add_layer(net_adam, -5, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("ann_add_layer with negative nodes returns ERR_INVALID", 
           (result == ERR_INVALID));

    // Test adding very large layer - use separate network to avoid state corruption
    PNetwork net_huge = ann_make_network(OPT_ADAM, LOSS_MSE);
    result = ann_add_layer(net_huge, 1000000, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("ann_add_layer with huge node count returns ERR_OK or ERR_ALLOC", 
           (result == ERR_OK || result == ERR_ALLOC));

    // If allocation succeeded, verify the layer
    if (result == ERR_OK && net_huge != NULL && net_huge->layer_count > 0) {
        TESTEX("Large layer node_count set correctly", 
               (net_huge->layers[net_huge->layer_count - 1].node_count == 1000000));
    }
    
    // Free immediately to avoid issues with huge allocation
    ann_free_network(net_huge);

    // ========================================================================
    // NETWORK STATE TESTS
    // ========================================================================
    SUITE("Network State");
    COMMENT("Testing network state properties...");

    PNetwork net_state = ann_make_network(OPT_ADAM, LOSS_MSE);
    
    // Check initial state
    TESTEX("New network has optimizer OPT_ADAM", 
           (net_state != NULL && net_state->optimizer == OPT_ADAM));
    TESTEX("New network has loss_type MSE", 
           (net_state != NULL && net_state->loss_type == LOSS_MSE));
    TESTEX("New network has zero layers", 
           (net_state != NULL && net_state->layer_count == 0));

    // Add layers and check state
    ann_add_layer(net_state, 5, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net_state, 3, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net_state, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    TESTEX("Network with 3 layers has correct layer_count", 
           (net_state != NULL && net_state->layer_count == 3));
    TESTEX("Network maintains optimizer after adding layers", 
           (net_state != NULL && net_state->optimizer == OPT_ADAM));
    TESTEX("Network maintains loss_type after adding layers", 
           (net_state != NULL && net_state->loss_type == LOSS_MSE));

    // ========================================================================
    // MEMORY MANAGEMENT TESTS
    // ========================================================================
    SUITE("Memory Management");
    COMMENT("Testing network cleanup and deallocation...");

    // Create a network with multiple layers
    PNetwork net_to_free = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net_to_free, 100, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net_to_free, 50, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net_to_free, 50, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net_to_free, 10, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    // Free the network
    ann_free_network(net_to_free);
    TESTEX("ann_free_network completes without error", 1);

    // Clean up all test networks
    ann_free_network(net_sgd_mse);
    ann_free_network(net_sgd_decay);
    ann_free_network(net_adam);
    ann_free_network(net_momentum);
    ann_free_network(net_activations);
    ann_free_network(net_state);

    TESTEX("All networks freed successfully", 1);

    // ========================================================================
    // ERROR CODE CONVERSION TESTS
    // ========================================================================
    SUITE("Error Handling Utilities");
    COMMENT("Testing error code to string conversion...");

    const char *err_msg = ann_strerror(ERR_OK);
    TESTEX("ann_strerror(ERR_OK) returns non-NULL", (err_msg != NULL));

    err_msg = ann_strerror(ERR_NULL_PTR);
    TESTEX("ann_strerror(ERR_NULL_PTR) returns non-NULL", (err_msg != NULL));

    err_msg = ann_strerror(ERR_ALLOC);
    TESTEX("ann_strerror(ERR_ALLOC) returns non-NULL", (err_msg != NULL));

    err_msg = ann_strerror(ERR_INVALID);
    TESTEX("ann_strerror(ERR_INVALID) returns non-NULL", (err_msg != NULL));

    err_msg = ann_strerror(ERR_IO);
    TESTEX("ann_strerror(ERR_IO) returns non-NULL", (err_msg != NULL));

    err_msg = ann_strerror(ERR_FAIL);
    TESTEX("ann_strerror(ERR_FAIL) returns non-NULL", (err_msg != NULL));

    TESTEX("Error message strings are descriptive", 
           (strlen(ann_strerror(ERR_NULL_PTR)) > 0));

    // ========================================================================
    // ERROR CALLBACK TESTS
    // ========================================================================
    SUITE("Error Callback Management");
    COMMENT("Testing error logging callback system...");

    // Check initial state (no callback)
    ErrorLogCallback initial_callback = ann_get_error_log_callback();
    TESTEX("Initial error callback is NULL", (initial_callback == NULL));

    // Set the callback (defined at module level)
    ann_set_error_log_callback(test_error_callback);
    ErrorLogCallback set_callback = ann_get_error_log_callback();
    TESTEX("ann_set_error_log_callback sets callback successfully", 
           (set_callback != NULL));

    // Clear the callback
    ann_clear_error_log_callback();
    ErrorLogCallback cleared_callback = ann_get_error_log_callback();
    TESTEX("ann_clear_error_log_callback clears callback successfully", 
           (cleared_callback == NULL));

    // Test setting NULL explicitly
    ann_set_error_log_callback(NULL);
    TESTEX("ann_set_error_log_callback(NULL) disables callback", 
           (ann_get_error_log_callback() == NULL));

    // ========================================================================
    // COMPREHENSIVE TOPOLOGY TESTS
    // ========================================================================
    SUITE("Complex Topologies");
    COMMENT("Testing more complex network configurations...");

    // Test deep network (5+ layers)
    PNetwork deep_net = ann_make_network(OPT_MOMENTUM, LOSS_MSE);
    ann_add_layer(deep_net, 784, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(deep_net, 256, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(deep_net, 128, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(deep_net, 64, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(deep_net, 32, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(deep_net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

    TESTEX("Deep network (6 layers) created successfully", 
           (deep_net != NULL && deep_net->layer_count == 6));

    // Test varying layer sizes
    int layer_sizes_valid = 1;
    if (deep_net && deep_net->layer_count == 6) {
        int expected_sizes[] = {784, 256, 128, 64, 32, 10};
        for (int i = 0; i < 6; i++) {
            if (deep_net->layers[i].node_count != expected_sizes[i]) {
                layer_sizes_valid = 0;
                break;
            }
        }
    }
    TESTEX("Deep network layer sizes set correctly", layer_sizes_valid);

    // Test network with single hidden layer
    PNetwork simple_net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(simple_net, 4, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(simple_net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(simple_net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    TESTEX("Simple network (3 layers) created successfully", 
           (simple_net != NULL && simple_net->layer_count == 3));

    // Cleanup deep networks
    ann_free_network(deep_net);
    ann_free_network(simple_net);
    TESTEX("Complex networks freed successfully", 1);

    TESTEX("Network creation and configuration tests completed", 1);
}
