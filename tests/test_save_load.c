#include <stdio.h>
#include <string.h>
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

#define TEST_FILE_TXT "test_roundtrip.nna"
#define TEST_FILE_BIN "test_roundtrip.nnb"
#define TEST_FILE_ONNX "test_roundtrip.onnx.json"

// returns 1 if all weights and biases of a and b match within tol
static int weights_match(const PNetwork a, const PNetwork b, real tol)
{
    if (a->layer_count != b->layer_count)
        return 0;

    for (int layer = 0; layer < a->layer_count - 1; layer++)
    {
        PTensor wa = a->layers[layer].t_weights, wb = b->layers[layer].t_weights;
        PTensor ba = a->layers[layer].t_bias,    bb = b->layers[layer].t_bias;

        if (wa->rows != wb->rows || wa->cols != wb->cols || ba->cols != bb->cols)
            return 0;

        for (int i = 0; i < wa->rows * wa->cols; i++)
            if (fabs(wa->values[i] - wb->values[i]) > tol)
                return 0;

        for (int i = 0; i < ba->cols; i++)
            if (fabs(ba->values[i] - bb->values[i]) > tol)
                return 0;
    }

    return 1;
}

// verify a loaded network has weights_set and that ann_train_begin()
// does not re-randomize the loaded weights (issue #37)
// (macro because TESTEX requires string-literal names)
#define CHECK_LOADED_WEIGHTS(fmt, original, load_expr)                                      \
    do {                                                                                    \
        PNetwork loaded_ = (load_expr);                                                     \
        TESTEX(fmt ": load succeeds", loaded_ != NULL);                                     \
        if (loaded_) {                                                                      \
            TESTEX(fmt ": weights_set after load", loaded_->weights_set == 1);              \
            TESTEX(fmt ": loaded weights/biases match original",                            \
                   weights_match(original, loaded_, (real)1e-4));                           \
            ann_train_begin(loaded_);                                                       \
            TESTEX(fmt ": ann_train_begin preserves loaded weights",                        \
                   weights_match(original, loaded_, (real)1e-4));                           \
            ann_train_end(loaded_);                                                         \
            ann_free_network(loaded_);                                                      \
        }                                                                                   \
    } while (0)

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    #if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

    MODULE("Save/Load Serialization");

    SUITE("Text Format");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 2, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.01f);
        
        int save_err = ann_save_network(net, TEST_FILE_TXT);
        TESTEX("Save to text", save_err == 0);
        
        PNetwork loaded = ann_load_network(TEST_FILE_TXT);
        TESTEX("Load from text", loaded != NULL);
        
        if (loaded != NULL) {
            TESTEX("Text: layers preserved", loaded->layer_count == net->layer_count);
            TESTEX("Text: loss type preserved", loaded->loss_type == net->loss_type);
            ann_free_network(loaded);
        }
        
        ann_free_network(net);
        remove(TEST_FILE_TXT);
    }

    SUITE("Binary Format");
    {
        PNetwork net = ann_make_network(OPT_MOMENTUM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.025f);
        
        int save_err = ann_save_network_binary(net, TEST_FILE_BIN);
        TESTEX("Save to binary", save_err == 0);
        
        PNetwork loaded = ann_load_network_binary(TEST_FILE_BIN);
        TESTEX("Load from binary", loaded != NULL);
        
        if (loaded != NULL) {
            TESTEX("Binary: layers preserved", loaded->layer_count == net->layer_count);
            TESTEX("Binary: loss preserved", loaded->loss_type == net->loss_type);
            ann_free_network(loaded);
        }
        
        ann_free_network(net);
        remove(TEST_FILE_BIN);
    }

    SUITE("Round Trip Predictions");
    {
        PNetwork original = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(original, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(original, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(original, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(original, 0.5f);
        ann_set_epoch_limit(original, 500);  // Limit epochs to prevent long runs
        ann_set_convergence(original, 0.01f);  // Stop when loss is low enough
        
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 0};
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        // Single training call - let convergence/epoch limit handle termination
        ann_train_network(original, inputs, targets_t, 4);
        
        real test_input[] = {0.5f, 0.5f};
        real original_output[1];
        ann_predict(original, test_input, original_output);
        
        ann_save_network(original, TEST_FILE_TXT);
        PNetwork loaded = ann_load_network(TEST_FILE_TXT);
        if (loaded != NULL) {
            real loaded_output[1];
            ann_predict(loaded, test_input, loaded_output);
            real diff = fabsf(original_output[0] - loaded_output[0]);
            TESTEX("Predictions match after save/load", diff < 0.01f);
            ann_free_network(loaded);
        }
        
        ann_free_network(original);
        tensor_free(inputs);
        tensor_free(targets_t);
        remove(TEST_FILE_TXT);
    }

    SUITE("Loaded Weights Preserved");
    {
        PNetwork original = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(original, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(original, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(original, 3, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(original, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

        // initialize weights without training
        ann_train_begin(original);
        ann_train_end(original);

        // He/Xavier init zeroes biases; make them distinct so bias loading is verified
        for (int layer = 0; layer < original->layer_count - 1; layer++)
        {
            PTensor bias = original->layers[layer].t_bias;
            for (int i = 0; i < bias->cols; i++)
                bias->values[i] = (real)(0.1 * (layer + 1) + 0.01 * (i + 1));
        }

        ann_save_network(original, TEST_FILE_TXT);
        ann_save_network_binary(original, TEST_FILE_BIN);
        ann_export_onnx(original, TEST_FILE_ONNX);

        CHECK_LOADED_WEIGHTS("Text", original, ann_load_network(TEST_FILE_TXT));
        CHECK_LOADED_WEIGHTS("Binary", original, ann_load_network_binary(TEST_FILE_BIN));
        CHECK_LOADED_WEIGHTS("ONNX", original, ann_import_onnx(TEST_FILE_ONNX));

        ann_free_network(original);
        remove(TEST_FILE_TXT);
        remove(TEST_FILE_BIN);
        remove(TEST_FILE_ONNX);
    }

    SUITE("Error Handling");
    {
        PNetwork non_exist = ann_load_network("nonexistent_xyz.nna");
        TESTEX("Load nonexistent returns NULL", non_exist == NULL);
        
        int save_null = ann_save_network(NULL, TEST_FILE_TXT);
        TESTEX("Save NULL returns error", save_null != 0);
        
        remove(TEST_FILE_TXT);
    }

    TESTEX("Save/load tests completed", 1);
}
