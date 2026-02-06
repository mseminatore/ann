#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

#define TEST_FILE_TXT "test_roundtrip.nna"
#define TEST_FILE_BIN "test_roundtrip.nnb"

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
        ann_add_layer(original, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(original, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(original, 0.3f);
        
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 0};
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        for (int i = 0; i < 50; i++) {
            ann_train_network(original, inputs, targets_t, 4);
        }
        
        real test_input[] = {0.5f, 0.5f};
        real original_output[1];
        ann_predict(original, test_input, original_output);
        
        ann_save_network(original, TEST_FILE_TXT);
        PNetwork loaded = ann_load_network(TEST_FILE_TXT);
        if (loaded != NULL) {
            real loaded_output[1];
            ann_predict(loaded, test_input, loaded_output);
            real diff = fabs(original_output[0] - loaded_output[0]);
            TESTEX("Predictions match after save/load", diff < 0.01f);
            ann_free_network(loaded);
        }
        
        ann_free_network(original);
        tensor_free(inputs);
        tensor_free(targets_t);
        remove(TEST_FILE_TXT);
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
