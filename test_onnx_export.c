#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

#ifdef _WIN32
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif

// Helper to check if file exists
static int file_exists(const char *filename) {
    return access(filename, F_OK) == 0;
}

// Helper to check if file contains a string
static int file_contains(const char *filename, const char *needle) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *content = (char *)malloc(size + 1);
    if (!content) { fclose(f); return 0; }
    
    size_t read = fread(content, 1, size, f);
    content[read] = '\0';
    fclose(f);
    
    int found = strstr(content, needle) != NULL;
    free(content);
    return found;
}

// Helper to remove test file
static void remove_file(const char *filename) {
    remove(filename);
}

void test_main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    MODULE("ONNX Export Tests");

    const char *test_file = "test_model.onnx.json";

    // ========================================================================
    // BASIC EXPORT TESTS
    // ========================================================================
    SUITE("Basic ONNX Export");
    COMMENT("Testing ann_export_onnx with various network configurations...");

    // Create a simple 4-8-3 network with ReLU and Softmax
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    TESTEX("Create network for export", (net != NULL));

    int result = ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
    TESTEX("Add input layer (4 nodes)", (result == ERR_OK));

    result = ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
    TESTEX("Add hidden layer (8 nodes, ReLU)", (result == ERR_OK));

    result = ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    TESTEX("Add output layer (3 nodes, Softmax)", (result == ERR_OK));

    // Export to ONNX JSON
    result = ann_export_onnx(net, test_file);
    TESTEX("ann_export_onnx returns ERR_OK", (result == ERR_OK));
    TESTEX("ONNX JSON file was created", file_exists(test_file));

    // Verify JSON structure
    COMMENT("Verifying ONNX JSON structure...");
    TESTEX("File contains ir_version", file_contains(test_file, "\"ir_version\""));
    TESTEX("File contains opset_import", file_contains(test_file, "\"opset_import\""));
    TESTEX("File contains producer_name", file_contains(test_file, "\"producer_name\": \"ann-library\""));
    TESTEX("File contains graph", file_contains(test_file, "\"graph\""));
    TESTEX("File contains initializer", file_contains(test_file, "\"initializer\""));
    TESTEX("File contains node", file_contains(test_file, "\"node\""));
    TESTEX("File contains input", file_contains(test_file, "\"input\""));
    TESTEX("File contains output", file_contains(test_file, "\"output\""));

    // Verify operations
    COMMENT("Verifying ONNX operations...");
    TESTEX("File contains MatMul op", file_contains(test_file, "\"op_type\": \"MatMul\""));
    TESTEX("File contains Add op", file_contains(test_file, "\"op_type\": \"Add\""));
    TESTEX("File contains Relu op", file_contains(test_file, "\"op_type\": \"Relu\""));
    TESTEX("File contains Softmax op", file_contains(test_file, "\"op_type\": \"Softmax\""));

    // Verify weights and biases
    COMMENT("Verifying weights and biases...");
    TESTEX("File contains weight_0", file_contains(test_file, "\"name\": \"weight_0\""));
    TESTEX("File contains bias_0", file_contains(test_file, "\"name\": \"bias_0\""));
    TESTEX("File contains weight_1", file_contains(test_file, "\"name\": \"weight_1\""));
    TESTEX("File contains bias_1", file_contains(test_file, "\"name\": \"bias_1\""));
    TESTEX("File contains float_data", file_contains(test_file, "\"float_data\""));
    TESTEX("File contains dims", file_contains(test_file, "\"dims\""));

    remove_file(test_file);
    ann_free_network(net);

    // ========================================================================
    // ERROR HANDLING TESTS  
    // ========================================================================
    SUITE("ONNX Export Error Handling");
    COMMENT("Testing error conditions...");

    // Test NULL network
    result = ann_export_onnx(NULL, test_file);
    TESTEX("NULL network returns ERR_NULL_PTR", (result == ERR_NULL_PTR));

    // Test NULL filename
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    result = ann_export_onnx(net, NULL);
    TESTEX("NULL filename returns ERR_NULL_PTR", (result == ERR_NULL_PTR));

    // Test network with insufficient layers
    PNetwork net_small = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net_small, 4, LAYER_INPUT, ACTIVATION_NULL);
    result = ann_export_onnx(net_small, test_file);
    TESTEX("Single layer network returns ERR_INVALID", (result == ERR_INVALID));
    ann_free_network(net_small);

    ann_free_network(net);

    // ========================================================================
    // DIFFERENT ACTIVATION TESTS
    // ========================================================================
    SUITE("ONNX Export with Different Activations");
    COMMENT("Testing export with various activation functions...");

    // Test with Sigmoid
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    result = ann_export_onnx(net, test_file);
    TESTEX("Export with Sigmoid succeeds", (result == ERR_OK));
    TESTEX("File contains Sigmoid op", file_contains(test_file, "\"op_type\": \"Sigmoid\""));
    remove_file(test_file);
    ann_free_network(net);

    // Test with Tanh
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_TANH);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
    result = ann_export_onnx(net, test_file);
    TESTEX("Export with Tanh succeeds", (result == ERR_OK));
    TESTEX("File contains Tanh op", file_contains(test_file, "\"op_type\": \"Tanh\""));
    remove_file(test_file);
    ann_free_network(net);

    // Test with LeakyReLU
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_LEAKY_RELU);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
    result = ann_export_onnx(net, test_file);
    TESTEX("Export with LeakyReLU succeeds", (result == ERR_OK));
    TESTEX("File contains LeakyRelu op", file_contains(test_file, "\"op_type\": \"LeakyRelu\""));
    TESTEX("LeakyRelu has alpha attribute", file_contains(test_file, "\"alpha\""));
    remove_file(test_file);
    ann_free_network(net);

    // Test with no activation (null)
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
    result = ann_export_onnx(net, test_file);
    TESTEX("Export with no activation succeeds", (result == ERR_OK));
    remove_file(test_file);
    ann_free_network(net);

    // ========================================================================
    // PIKCHR EXPORT TESTS
    // ========================================================================
    SUITE("PIKCHR Export");
    COMMENT("Testing PIKCHR diagram export...");

    const char *pikchr_file = "test_network.pikchr";

    // Test simple mode (large network)
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 784, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 128, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    result = ann_export_pikchr(net, pikchr_file);
    TESTEX("PIKCHR export (simple mode) succeeds", (result == ERR_OK));
    TESTEX("PIKCHR file contains 'box'", file_contains(pikchr_file, "box"));
    TESTEX("PIKCHR file contains 'Input'", file_contains(pikchr_file, "Input"));
    TESTEX("PIKCHR file contains 'Hidden'", file_contains(pikchr_file, "Hidden"));
    TESTEX("PIKCHR file contains 'Softmax'", file_contains(pikchr_file, "Softmax"));
    remove_file(pikchr_file);
    ann_free_network(net);

    // Test detailed mode (small network)
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    result = ann_export_pikchr(net, pikchr_file);
    TESTEX("PIKCHR export (detailed mode) succeeds", (result == ERR_OK));
    TESTEX("PIKCHR detailed file contains 'circle'", file_contains(pikchr_file, "circle"));
    TESTEX("PIKCHR detailed file contains 'line'", file_contains(pikchr_file, "line"));
    remove_file(pikchr_file);
    ann_free_network(net);

    // Test error cases
    result = ann_export_pikchr(NULL, pikchr_file);
    TESTEX("PIKCHR export with NULL network returns ERR_NULL_PTR", (result == ERR_NULL_PTR));

    net = ann_make_network(OPT_SGD, LOSS_MSE);
    result = ann_export_pikchr(net, pikchr_file);
    TESTEX("PIKCHR export with no layers returns ERR_INVALID", (result == ERR_INVALID));
    ann_free_network(net);

    // ========================================================================
    // ONNX IMPORT TESTS
    // ========================================================================
    SUITE("ONNX Import");
    COMMENT("Testing ONNX JSON import (round-trip)...");

    const char *import_file = "test_import.onnx.json";

    // Create a network, export it, then import it back
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    
    // Set some known weights for verification
    net->layers[0].t_weights->values[0] = 0.123f;
    net->layers[0].t_weights->values[1] = 0.456f;
    net->layers[0].t_bias->values[0] = 0.789f;
    net->layers[1].t_weights->values[0] = 0.321f;
    net->layers[1].t_bias->values[0] = 0.654f;
    
    result = ann_export_onnx(net, import_file);
    TESTEX("Export for import test succeeds", (result == ERR_OK));
    
    // Import the network
    PNetwork imported = ann_import_onnx(import_file);
    TESTEX("Import ONNX succeeds", (imported != NULL));
    
    if (imported)
    {
        TESTEX("Imported network has correct layer count", 
               (imported->layer_count == net->layer_count));
        TESTEX("Imported input layer size matches", 
               (imported->layers[0].node_count == 4));
        TESTEX("Imported hidden layer size matches", 
               (imported->layers[1].node_count == 8));
        TESTEX("Imported output layer size matches", 
               (imported->layers[2].node_count == 3));
        TESTEX("Imported hidden activation is RELU", 
               (imported->layers[1].activation == ACTIVATION_RELU));
        TESTEX("Imported output activation is SOFTMAX", 
               (imported->layers[2].activation == ACTIVATION_SOFTMAX));
        
        // Verify weights match
        real w0 = imported->layers[0].t_weights->values[0];
        real w1 = imported->layers[0].t_weights->values[1];
        real b0 = imported->layers[0].t_bias->values[0];
        TESTEX("Imported weight[0][0] matches", (w0 > 0.12f && w0 < 0.13f));
        TESTEX("Imported weight[0][1] matches", (w1 > 0.45f && w1 < 0.46f));
        TESTEX("Imported bias[0][0] matches", (b0 > 0.78f && b0 < 0.80f));
        
        ann_free_network(imported);
    }
    
    ann_free_network(net);
    
    // Test import error cases
    imported = ann_import_onnx(NULL);
    TESTEX("Import with NULL filename returns NULL", (imported == NULL));
    
    imported = ann_import_onnx("nonexistent_file.json");
    TESTEX("Import of nonexistent file returns NULL", (imported == NULL));
    
    remove_file(import_file);

    // Cleanup any remaining test files
    remove_file(test_file);
    remove_file(pikchr_file);
}
