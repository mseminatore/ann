#include <stdio.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Forward Pass Correctness");

    SUITE("Single Input");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        
        real input[2] = {0.5f, -0.3f};
        real output[1];
        
        int err = ann_predict(net, input, output);
        TESTEX("Prediction succeeds", err == 0);
        TESTEX("Output in sigmoid range", output[0] >= 0.0f && output[0] <= 1.0f);
        TESTEX("Output not NaN", !isnan(output[0]));
        
        ann_free_network(net);
    }

    SUITE("Multi-Layer Forward Pass");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 3, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 2, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        
        TESTEX("Network has 3 layers", net->layer_count == 3);
        
        real input[] = {0.5f, -0.2f};
        real output[1];
        int err = ann_predict(net, input, output);
        TESTEX("Multi-layer prediction succeeds", err == 0);
        TESTEX("Output valid", output[0] >= 0.0f && output[0] <= 1.0f);
        
        ann_free_network(net);
    }

    SUITE("Activation Functions");
    {
        // Sigmoid
        PNetwork net_sig = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_sig, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        real input1[] = {1.0f};
        real out_sig[1];
        ann_predict(net_sig, input1, out_sig);
        TESTEX("Sigmoid in range", out_sig[0] > 0.0f && out_sig[0] < 1.0f);
        ann_free_network(net_sig);
        
        // ReLU
        PNetwork net_relu = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_relu, 1, LAYER_OUTPUT, ACTIVATION_RELU);
        real out_relu[1];
        ann_predict(net_relu, input1, out_relu);
        TESTEX("ReLU non-negative", out_relu[0] >= 0.0f);
        ann_free_network(net_relu);
        
        // Leaky ReLU
        PNetwork net_lrelu = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net_lrelu, 1, LAYER_OUTPUT, ACTIVATION_LEAKY_RELU);
        real out_lrelu[1];
        ann_predict(net_lrelu, input1, out_lrelu);
        TESTEX("Leaky ReLU output valid", !isnan(out_lrelu[0]) && !isinf(out_lrelu[0]));
        ann_free_network(net_lrelu);
    }

    SUITE("Forward Pass Consistency");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        
        real input[] = {0.3f, 0.7f};
        real out1[1], out2[1], out3[1];
        
        ann_predict(net, input, out1);
        ann_predict(net, input, out2);
        ann_predict(net, input, out3);
        
        real diff12 = fabs(out1[0] - out2[0]);
        real diff23 = fabs(out2[0] - out3[0]);
        TESTEX("Consistent predictions", diff12 < 1e-5f && diff23 < 1e-5f);
        
        ann_free_network(net);
    }

    SUITE("Boundary Conditions");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        
        // Zero input
        real zero_in[] = {0.0f};
        real zero_out[1];
        ann_predict(net, zero_in, zero_out);
        TESTEX("Zero input handled", !isnan(zero_out[0]));
        
        // Large positive
        real large_in[] = {50.0f};
        real large_out[1];
        ann_predict(net, large_in, large_out);
        TESTEX("Large positive handled", large_out[0] > 0.5f);
        
        // Large negative
        real neg_in[] = {-50.0f};
        real neg_out[1];
        ann_predict(net, neg_in, neg_out);
        TESTEX("Large negative handled", neg_out[0] < 0.5f);
        
        ann_free_network(net);
    }

    TESTEX("Forward pass tests completed", 1);
}
