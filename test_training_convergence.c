#include <stdio.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
#endif

    MODULE("Training Convergence");

    SUITE("AND Gate");
    {
        real and_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real and_targets[] = {0, 0, 0, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, and_data);
        PTensor targets = tensor_create_from_array(4, 1, and_targets);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.5f);
        
        real final_loss = ann_train_network(net, inputs, targets, 4);
        
        TESTEX("AND loss reduces", final_loss < 0.5f);
        TESTEX("AND converges", final_loss < 0.1f);
        
        // Test prediction
        real test[] = {1, 1};
        real pred[1];
        ann_predict(net, test, pred);
        TESTEX("AND 1,1 = 1", pred[0] > 0.5f);
        
        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(targets);
    }

    SUITE("OR Gate");
    {
        real or_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real or_targets[] = {0, 1, 1, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, or_data);
        PTensor targets = tensor_create_from_array(4, 1, or_targets);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.5f);
        
        real final_loss = ann_train_network(net, inputs, targets, 4);
        
        TESTEX("OR loss reduces", final_loss < 0.5f);
        TESTEX("OR converges", final_loss < 0.1f);
        
        real test[] = {0, 1};
        real pred[1];
        ann_predict(net, test, pred);
        TESTEX("OR 0,1 = 1", pred[0] > 0.5f);
        
        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(targets);
    }

    SUITE("XOR Gate");
    {
        real xor_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real xor_targets[] = {0, 1, 1, 0};
        
        PTensor inputs = tensor_create_from_array(4, 2, xor_data);
        PTensor targets = tensor_create_from_array(4, 1, xor_targets);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.5f);
        
        real final_loss = ann_train_network(net, inputs, targets, 4);
        
        TESTEX("XOR loss reduces", final_loss < 0.5f);
        
        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(targets);
    }

    SUITE("Learning Rates");
    {
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        // Low LR
        PNetwork net_low = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_low, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net_low, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net_low, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net_low, 0.1f);
        
        real low_loss = ann_train_network(net_low, inputs, targets_t, 4);
        
        // High LR
        PNetwork net_high = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_high, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net_high, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net_high, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net_high, 1.0f);
        
        real high_loss = ann_train_network(net_high, inputs, targets_t, 4);
        
        TESTEX("Low LR converges", low_loss < 0.2f);
        TESTEX("High LR converges", high_loss < 0.2f);
        
        ann_free_network(net_low);
        ann_free_network(net_high);
        tensor_free(inputs);
        tensor_free(targets_t);
    }

    SUITE("Optimizers");
    {
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        // SGD
        PNetwork net_sgd = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_sgd, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net_sgd, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net_sgd, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net_sgd, 0.5f);
        
        real sgd_loss = ann_train_network(net_sgd, inputs, targets_t, 4);
        
        // Momentum
        PNetwork net_mom = ann_make_network(OPT_MOMENTUM, LOSS_MSE);
        ann_add_layer(net_mom, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net_mom, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net_mom, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net_mom, 0.5f);
        
        real mom_loss = ann_train_network(net_mom, inputs, targets_t, 4);
        
        TESTEX("SGD converges", sgd_loss < 0.2f);
        TESTEX("Momentum converges", mom_loss < 0.2f);
        
        ann_free_network(net_sgd);
        ann_free_network(net_mom);
        tensor_free(inputs);
        tensor_free(targets_t);
    }

    SUITE("Network Depth");
    {
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 1};
        
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        // Shallow (input + output only)
        PNetwork shallow = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(shallow, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(shallow, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(shallow, 0.5f);
        
        real shallow_loss = ann_train_network(shallow, inputs, targets_t, 4);
        
        // Deep (3 hidden layers)
        PNetwork deep = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(deep, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(deep, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(deep, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(deep, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(deep, 0.5f);
        
        real deep_loss = ann_train_network(deep, inputs, targets_t, 4);
        
        TESTEX("Shallow converges", shallow_loss < 0.2f);
        TESTEX("Deep converges", deep_loss < 0.2f);
        
        ann_free_network(shallow);
        ann_free_network(deep);
        tensor_free(inputs);
        tensor_free(targets_t);
    }

    TESTEX("Training convergence tests completed", 1);
}
