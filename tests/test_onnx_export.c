#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"
#include "json.h"

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

    // Test with GELU: needs opset 20 and the tanh approximation attribute
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_GELU);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
    result = ann_export_onnx(net, test_file);
    TESTEX("Export with GELU succeeds", (result == ERR_OK));
    TESTEX("File contains Gelu op", file_contains(test_file, "\"op_type\": \"Gelu\""));
    TESTEX("Gelu uses opset 20", file_contains(test_file, "\"version\": 20"));
    TESTEX("Gelu has tanh approximate attribute", file_contains(test_file, "\"approximate\""));
    {
        PNetwork back = ann_import_onnx(test_file);
        TESTEX("Gelu model imports", back != NULL);
        TESTEX("Imported hidden layer is GELU",
               back && ann_get_layer_activation(back, 1) == ACTIVATION_GELU);
        if (back) ann_free_network(back);
    }
    remove_file(test_file);
    ann_free_network(net);

    // Models without GELU keep opset 17
    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_NULL);
    ann_export_onnx(net, test_file);
    TESTEX("Models without GELU keep opset 17", file_contains(test_file, "\"version\": 17"));
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
    net->layers[1].t_weights->values[0] = 0.123f;
    net->layers[1].t_weights->values[1] = 0.456f;
    net->layers[1].t_bias->values[0] = 0.789f;
    net->layers[2].t_weights->values[0] = 0.321f;
    net->layers[2].t_bias->values[0] = 0.654f;
    
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
        real w0 = imported->layers[1].t_weights->values[0];
        real w1 = imported->layers[1].t_weights->values[1];
        real b0 = imported->layers[1].t_bias->values[0];
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

    // ========================================================================
    // NaN/Inf HANDLING TESTS
    // ========================================================================
    SUITE("ONNX Weight Layout");
    COMMENT("weight_N must be [in, out] for ONNX MatMul; 1.0 files stored libann's [out, in]...");
    {
        // 2 inputs -> 3 outputs, W[out][in] = 10 * out + in
        const char *layout_file = "test_layout.onnx.json";
        net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_NULL);
        for (int o = 0; o < 3; o++)
            for (int i = 0; i < 2; i++)
                net->layers[1].t_weights->values[o * 2 + i] = (real)(10 * o + i);
        ann_export_onnx(net, layout_file);
        TESTEX("Export is marked producer_version 1.1", file_contains(layout_file, "\"producer_version\": \"1.1\""));

        // Row-major [in=2, out=3]: row i holds W[0][i], W[1][i], W[2][i]
        const double expected[6] = {0, 10, 20, 1, 11, 21};
        int layout_ok = 0;
        JsonValue root;
        if (json_parse_file(layout_file, &root) == 0) {
            JsonValue *w = json_at(json_get(json_get(&root, "graph"), "initializer"), 0);
            JsonValue *dims = json_get(w, "dims");
            JsonValue *data = json_get(w, "float_data");
            int d0 = 0, d1 = 0;
            json_int(json_at(dims, 0), &d0);
            json_int(json_at(dims, 1), &d1);
            layout_ok = d0 == 2 && d1 == 3 && json_array_length(data) == 6;
            for (int k = 0; layout_ok && k < 6; k++) {
                double v = -1;
                json_number(json_at(data, k), &v);
                layout_ok = (v == expected[k]);
            }
            json_free(&root);
        }
        TESTEX("weight_0 is written transposed as [in, out]", layout_ok);

        PNetwork back = ann_import_onnx(layout_file);
        int same = back != NULL;
        for (int k = 0; same && k < 6; k++)
            same = back->layers[1].t_weights->values[k] == net->layers[1].t_weights->values[k];
        TESTEX("Import transposes the weights back", same);
        ann_free_network(back);
        remove(layout_file);

        // A 1.0 file stored the [out, in] data under dims [in, out]; it must still load
        FILE *f = fopen(layout_file, "w");
        if (f) {
            fputs("{\"ir_version\": 8, \"opset_import\": [{\"version\": 17}],\n"
                  " \"producer_name\": \"ann-library\", \"producer_version\": \"1.0\",\n"
                  " \"graph\": {\"name\": \"ann_model\",\n"
                  "  \"initializer\": [\n"
                  "   {\"name\": \"weight_0\", \"data_type\": 1, \"dims\": [2, 3], \"float_data\": [0, 1, 10, 11, 20, 21]},\n"
                  "   {\"name\": \"bias_0\", \"data_type\": 1, \"dims\": [3], \"float_data\": [0, 0, 0]}],\n"
                  "  \"node\": [\n"
                  "   {\"op_type\": \"MatMul\", \"name\": \"matmul_0\", \"input\": [\"input\", \"weight_0\"], \"output\": [\"matmul_0_out\"]},\n"
                  "   {\"op_type\": \"Add\", \"name\": \"add_0\", \"input\": [\"matmul_0_out\", \"bias_0\"], \"output\": [\"add_0_out\"]}]}}\n", f);
            fclose(f);
        }
        back = ann_import_onnx(layout_file);
        same = back != NULL;
        for (int k = 0; same && k < 6; k++)
            same = back->layers[1].t_weights->values[k] == net->layers[1].t_weights->values[k];
        TESTEX("Legacy 1.0 file imports without transposing", same);
        ann_free_network(back);
        remove(layout_file);
        ann_free_network(net);
    }

    SUITE("ONNX Export NaN/Inf Handling");
    COMMENT("Testing that NaN/Inf values are serialized as valid JSON...");

    net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    // Inject NaN and Inf into weights
    net->layers[1].t_weights->values[0] = NAN;
    net->layers[1].t_weights->values[1] = INFINITY;
    net->layers[1].t_bias->values[0] = -INFINITY;

    result = ann_export_onnx(net, test_file);
    TESTEX("Export with NaN/Inf values returns ERR_OK", (result == ERR_OK));
    TESTEX("Output does not contain 'nan'", !file_contains(test_file, "nan"));
    TESTEX("Output does not contain 'inf'", !file_contains(test_file, "inf"));
    TESTEX("Output contains 0.0 replacement", file_contains(test_file, "0.0"));

    remove_file(test_file);
    ann_free_network(net);

    // Cleanup any remaining test files
    remove_file(test_file);
    remove_file(pikchr_file);
}
