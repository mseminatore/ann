#include <stdio.h>
#include <string.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Online Training");

    SUITE("Basic step training (AND gate)");
    {
        // AND gate data: 4 samples, 2 inputs, 1 output
        real data[] = {0,0, 0,1, 1,0, 1,1};
        real targets[] = {0, 0, 0, 1};

        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.01f);

        int rc = ann_train_begin(net);
        TESTEX("train_begin returns ERR_OK", rc == ERR_OK);

        // Train for many steps, feeding all 4 samples as one batch each step
        real loss = 0;
        for (int step = 0; step < 2000; step++)
        {
            loss = ann_train_step(net, data, targets, 4);
        }

        TESTEX("AND loss converges via steps", loss < 0.05f);

        // Predict while still in training session
        real pred[1];
        ann_predict(net, (real[]){1, 1}, pred);
        TESTEX("AND 1,1 ~ 1 mid-training", pred[0] > 0.5f);

        ann_predict(net, (real[]){0, 0}, pred);
        TESTEX("AND 0,0 ~ 0 mid-training", pred[0] < 0.5f);

        ann_train_end(net);

        // Predict after training session ends
        ann_predict(net, (real[]){1, 1}, pred);
        TESTEX("AND 1,1 ~ 1 post-training", pred[0] > 0.5f);

        ann_free_network(net);
    }

    SUITE("Incremental training (continue after initial train)");
    {
        // Train initially with ann_train_network on OR gate
        real data[] = {0,0, 0,1, 1,0, 1,1};
        real targets[] = {0, 1, 1, 1};

        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor tgts = tensor_create_from_array(4, 1, targets);

        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.01f);

        real initial_loss = ann_train_network(net, inputs, tgts, 4);
        TESTEX("initial training completes", initial_loss < 1.0f);

        // Now do online training with the same data
        int rc = ann_train_begin(net);
        TESTEX("train_begin after train_network", rc == ERR_OK);

        real loss = 0;
        for (int step = 0; step < 500; step++)
        {
            loss = ann_train_step(net, data, targets, 4);
        }

        TESTEX("online loss improves further", loss <= initial_loss + 0.01f);

        ann_train_end(net);

        real pred[1];
        ann_predict(net, (real[]){1, 0}, pred);
        TESTEX("OR 1,0 ~ 1 after online training", pred[0] > 0.5f);

        ann_free_network(net);
        tensor_free(inputs);
        tensor_free(tgts);
    }

    SUITE("Predict safe during dropout training");
    {
        real data[] = {0,0, 0,1, 1,0, 1,1};
        real targets[] = {0, 0, 0, 1};

        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.01f);
        ann_set_dropout(net, 0.3f);

        ann_train_begin(net);

        // Train some steps with dropout active
        for (int step = 0; step < 1000; step++)
        {
            ann_train_step(net, data, targets, 4);
        }

        // Predict mid-training: should give deterministic results (no dropout)
        real pred1[1], pred2[1];
        real test_input[] = {1, 1};
        ann_predict(net, test_input, pred1);
        ann_predict(net, test_input, pred2);
        TESTEX("predict deterministic with dropout", pred1[0] == pred2[0]);

        ann_train_end(net);
        ann_free_network(net);
    }

    SUITE("Single sample training");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.5f);

        ann_train_begin(net);

        // Feed one sample at a time
        real loss = 0;
        real samples[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
        real labels[] = {0, 0, 0, 1};  // AND gate

        for (int epoch = 0; epoch < 2000; epoch++)
        {
            for (int i = 0; i < 4; i++)
            {
                loss = ann_train_step(net, samples[i], &labels[i], 1);
            }
        }

        ann_train_end(net);

        real pred[1];
        ann_predict(net, (real[]){1, 1}, pred);
        TESTEX("single-sample AND 1,1 ~ 1", pred[0] > 0.5f);

        ann_predict(net, (real[]){0, 1}, pred);
        TESTEX("single-sample AND 0,1 ~ 0", pred[0] < 0.5f);

        ann_free_network(net);
    }

    SUITE("Error handling");
    {
        TESTEX("train_begin NULL", ann_train_begin(NULL) == ERR_NULL_PTR);
        TESTEX("train_step NULL net", ann_train_step(NULL, NULL, NULL, 1) == (real)0.0);
        
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_train_begin(net);

        TESTEX("train_step NULL inputs", ann_train_step(net, NULL, NULL, 1) == (real)0.0);
        TESTEX("train_step zero batch", ann_train_step(net, (real[]){1,1}, (real[]){1}, 0) == (real)0.0);

        ann_train_end(net);
        ann_free_network(net);

        // train_end on NULL should not crash
        ann_train_end(NULL);
    }
}
