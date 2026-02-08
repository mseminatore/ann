#include <stdio.h>
#include <math.h>
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

    TESTEX("Loss function tests completed", 1);
}
