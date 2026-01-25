#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

// Global counters
static int callback_count = 0;
static int last_error_code = 0;
static const char *last_function = NULL;

void test_callback(int error_code, const char *error_message, const char *function_name) {
    callback_count++;
    last_error_code = error_code;
    last_function = function_name;
    printf("Callback #%d: %s() returned %s\n", 
           callback_count, 
           function_name, 
           error_message);
}

int main(void) {
    printf("=== Testing Comprehensive Error Callback Coverage ===\n\n");
    
    // Set up callback
    ann_set_error_log_callback(test_callback);
    
    // Test 1: ann_predict with NULL network
    printf("Test 1: ann_predict with NULL network\n");
    real outputs[10];
    real inputs[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    ann_predict(NULL, inputs, outputs);
    if (callback_count != 1) printf("  FAIL: Expected 1 callback, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 2: ann_predict with NULL inputs
    printf("Test 2: ann_predict with NULL inputs\n");
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 5, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 3, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_predict(net, NULL, outputs);
    if (callback_count != 2) printf("  FAIL: Expected 2 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 3: ann_predict with invalid network (empty)
    printf("Test 3: ann_predict with invalid network (layer_count = 0)\n");
    PNetwork badnet = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_predict(badnet, inputs, outputs);
    if (callback_count != 3) printf("  FAIL: Expected 3 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 4: ann_load_csv with NULL filename
    printf("Test 4: ann_load_csv with NULL filename\n");
    real *data;
    int rows, stride;
    ann_load_csv(NULL, 0, &data, &rows, &stride);
    if (callback_count != 4) printf("  FAIL: Expected 4 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 5: ann_load_csv with non-existent file
    printf("Test 5: ann_load_csv with non-existent file\n");
    ann_load_csv("/nonexistent/file.csv", 0, &data, &rows, &stride);
    if (callback_count != 5) printf("  FAIL: Expected 5 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 6: ann_save_network with NULL network
    printf("Test 6: ann_save_network with NULL network\n");
    ann_save_network(NULL, "/tmp/test.txt");
    if (callback_count != 6) printf("  FAIL: Expected 6 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    // Test 7: ann_save_network_binary with NULL filename
    printf("Test 7: ann_save_network_binary with NULL filename\n");
    ann_save_network_binary(net, NULL);
    if (callback_count != 7) printf("  FAIL: Expected 7 callbacks, got %d\n", callback_count);
    else printf("  PASS\n\n");
    
    printf("=== Summary ===\n");
    printf("Total callbacks invoked: %d\n", callback_count);
    printf("All error points have callbacks: %s\n", 
           callback_count >= 7 ? "YES" : "NO");
    
    // Cleanup
    ann_free_network(net);
    ann_free_network(badnet);
    ann_clear_error_log_callback();
    
    return callback_count >= 7 ? 0 : 1;
}
