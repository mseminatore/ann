//-----------------------------------------------------------
// Error callback coverage.
//
// Each public entry point below is called with bad arguments and must both
// return an error code and invoke the installed error log callback exactly
// once, reporting the same code.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "ann.h"
#include "testy/test.h"

static int callback_count = 0;
static int last_error_code = 0;
static const char *last_function = NULL;

static void test_callback(int error_code, const char *error_message, const char *function_name)
{
    (void)error_message;
    callback_count++;
    last_error_code = error_code;
    last_function = function_name;
}

// Number of callbacks since the previous call
static int callbacks_since(int *mark)
{
    int n = callback_count - *mark;
    *mark = callback_count;
    return n;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Error Callback Coverage");

    ann_set_error_log_callback(test_callback);
    int mark = 0;

    real outputs[10];
    real inputs[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 5, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 3, LAYER_HIDDEN, ACTIVATION_RELU);
    PNetwork empty_net = ann_make_network(OPT_SGD, LOSS_MSE);

    SUITE("ann_predict()");
    int rc = ann_predict(NULL, inputs, outputs);
    TESTEX("NULL network returns ERR_NULL_PTR", (rc == ERR_NULL_PTR));
    TESTEX("NULL network invokes the callback once", (callbacks_since(&mark) == 1));
    TESTEX("Callback reports ERR_NULL_PTR", (last_error_code == ERR_NULL_PTR));
    TESTEX("Callback names the function", (last_function != NULL));

    rc = ann_predict(net, NULL, outputs);
    TESTEX("NULL inputs returns ERR_NULL_PTR", (rc == ERR_NULL_PTR));
    TESTEX("NULL inputs invokes the callback once", (callbacks_since(&mark) == 1));

    rc = ann_predict(empty_net, inputs, outputs);
    TESTEX("Network with no layers returns ERR_INVALID", (rc == ERR_INVALID));
    TESTEX("Network with no layers invokes the callback once", (callbacks_since(&mark) == 1));
    TESTEX("Callback reports ERR_INVALID", (last_error_code == ERR_INVALID));

    SUITE("ann_load_csv()");
    real *data = NULL;
    int rows = 0, stride = 0;
    rc = ann_load_csv(NULL, 0, &data, &rows, &stride);
    TESTEX("NULL filename returns ERR_NULL_PTR", (rc == ERR_NULL_PTR));
    TESTEX("NULL filename invokes the callback once", (callbacks_since(&mark) == 1));

    rc = ann_load_csv("/nonexistent/file.csv", 0, &data, &rows, &stride);
    TESTEX("Missing file returns ERR_IO", (rc == ERR_IO));
    TESTEX("Missing file invokes the callback once", (callbacks_since(&mark) == 1));
    TESTEX("Callback reports ERR_IO", (last_error_code == ERR_IO));

    SUITE("ann_save_network() / ann_save_network_binary()");
    rc = ann_save_network(NULL, "unused.nna");
    TESTEX("NULL network returns ERR_NULL_PTR", (rc == ERR_NULL_PTR));
    TESTEX("NULL network invokes the callback once", (callbacks_since(&mark) == 1));

    rc = ann_save_network_binary(net, NULL);
    TESTEX("NULL filename returns ERR_NULL_PTR", (rc == ERR_NULL_PTR));
    TESTEX("NULL filename invokes the callback once", (callbacks_since(&mark) == 1));

    SUITE("Callback Removal");
    ann_clear_error_log_callback();
    ann_predict(NULL, inputs, outputs);
    TESTEX("No callback after ann_clear_error_log_callback()", (callbacks_since(&mark) == 0));

    ann_free_network(net);
    ann_free_network(empty_net);
}
