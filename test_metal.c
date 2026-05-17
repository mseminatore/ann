/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* See LICENSE file for terms.                                                    */
/**********************************************************************************/

//
// test_metal.c - Unit tests for Metal GPU inference backend
//
// Only built and run when USE_METAL is defined (cmake -DUSE_METAL=1).
// Tests verify that GPU inference produces results matching CPU inference.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

#ifdef USE_METAL

//-----------------------------------------------------------
// Helper: build and train a tiny XOR network (2-2-1)
//-----------------------------------------------------------
static PNetwork make_xor_net(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    real xor_in[]  = {0,0, 0,1, 1,0, 1,1};
    real xor_out[] = {0,   1,   1,   0  };

    PTensor tin  = tensor_create_from_array(4, 2, xor_in);
    PTensor tout = tensor_create_from_array(4, 1, xor_out);

    ann_set_convergence(net, 0.01f);
    ann_train_network(net, tin, tout, 4);

    tensor_free(tin);
    tensor_free(tout);
    return net;
}

//-----------------------------------------------------------
// Helper: build a small softmax classifier (4-8-3)
//-----------------------------------------------------------
static PNetwork make_small_classifier(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    if (!net) return NULL;

    ann_set_learning_rate(net, 0.001f);
    ann_add_layer(net, 4, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

    // Simple training data: 3 classes, 4 features
    real inputs[] = {
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1,
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1
    };
    real targets[] = {
        1,0,0,  0,1,0,  0,0,1,  1,0,0,
        1,0,0,  0,1,0,  0,0,1,  1,0,0
    };

    PTensor tin  = tensor_create_from_array(8, 4, inputs);
    PTensor tout = tensor_create_from_array(8, 3, targets);

    ann_set_convergence(net, 0.1f);
    ann_train_network(net, tin, tout, 8);

    tensor_free(tin);
    tensor_free(tout);
    return net;
}

//-----------------------------------------------------------
// Helper: compare two float arrays within tolerance
//-----------------------------------------------------------
static int arrays_close(const real *a, const real *b, int n, real tol)
{
    for (int i = 0; i < n; i++)
    {
        real diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) return 0;
    }
    return 1;
}

#define TOLERANCE 1e-4f

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    MODULE("Metal GPU Inference Tests");

    // ========================================================================
    SUITE("GPU Initialization");
    // ========================================================================

    int gpu_ok = ann_gpu_init();
    TESTEX("ann_gpu_init() succeeds on Metal-capable hardware", (gpu_ok == 1));

    if (!gpu_ok)
    {
        COMMENT("Metal GPU not available - skipping remaining GPU tests");
        return;
    }

    // ========================================================================
    SUITE("GPU Upload and Free");
    // ========================================================================

    PNetwork net = make_xor_net();
    TESTEX("Network creation succeeds", (net != NULL));

    int upload_ok = ann_gpu_upload_network(net);
    TESTEX("ann_gpu_upload_network() returns ERR_OK", (upload_ok == ERR_OK));
    TESTEX("Layer 0 weights have gpu_buf after upload",
           (net->layers[0].t_weights != NULL && net->layers[0].t_weights->gpu_buf != NULL));
    TESTEX("Layer 0 bias has gpu_buf after upload",
           (net->layers[0].t_bias != NULL && net->layers[0].t_bias->gpu_buf != NULL));

    ann_gpu_free_network(net);
    TESTEX("gpu_buf is NULL after ann_gpu_free_network()",
           (net->layers[0].t_weights->gpu_buf == NULL));

    ann_free_network(net);

    // ========================================================================
    SUITE("GPU Single-Sample Inference Matches CPU");
    // ========================================================================

    PNetwork net2 = make_xor_net();
    TESTEX("XOR network created", (net2 != NULL));

    real xor_tests[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    real cpu_out[1], gpu_out[1];

    // Run CPU inference for all 4 XOR inputs
    real cpu_results[4];
    for (int i = 0; i < 4; i++)
    {
        ann_predict(net2, xor_tests[i], cpu_results + i);
    }

    // Upload to GPU and re-run
    ann_gpu_upload_network(net2);

    int gpu_matches_cpu = 1;
    for (int i = 0; i < 4; i++)
    {
        ann_predict(net2, xor_tests[i], gpu_out);
        real diff = gpu_out[0] - cpu_results[i];
        if (diff < 0) diff = -diff;
        if (diff > TOLERANCE) { gpu_matches_cpu = 0; break; }
    }
    TESTEX("GPU single-sample inference matches CPU for XOR network", (gpu_matches_cpu));

    ann_gpu_free_network(net2);
    ann_free_network(net2);

    // ========================================================================
    SUITE("ann_predict_batch() - GPU Batch Inference");
    // ========================================================================

    PNetwork net3 = make_small_classifier();
    TESTEX("Classifier network created", (net3 != NULL));

    real test_inputs[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    int n_samples = 4;
    int out_nodes = 3;

    // CPU baseline: sequential predictions
    real cpu_batch[12];  // 4 samples x 3 outputs
    for (int i = 0; i < n_samples; i++)
    {
        ann_predict(net3, test_inputs + i * 4, cpu_batch + i * out_nodes);
    }

    // GPU batch prediction
    ann_gpu_upload_network(net3);
    real gpu_batch[12];
    int batch_err = ann_predict_batch(net3, test_inputs, gpu_batch, n_samples);

    TESTEX("ann_predict_batch() returns ERR_OK", (batch_err == ERR_OK));
    TESTEX("GPU batch outputs match CPU outputs",
           (arrays_close(cpu_batch, gpu_batch, n_samples * out_nodes, TOLERANCE)));

    ann_gpu_free_network(net3);
    ann_free_network(net3);

    // ========================================================================
    SUITE("CPU Fallback When Not Uploaded");
    // ========================================================================

    PNetwork net4 = make_xor_net();
    TESTEX("Network for fallback test created", (net4 != NULL));

    // Do NOT call ann_gpu_upload_network -- should fall back to CPU
    real fallback_out[4];
    int fallback_err = ann_predict_batch(net4, (real[]){0,0, 0,1, 1,0, 1,1},
                                         fallback_out, 4);
    TESTEX("ann_predict_batch() with no GPU upload falls back to CPU (ERR_OK)",
           (fallback_err == ERR_OK));

    ann_free_network(net4);

    // ========================================================================
    SUITE("Error Handling");
    // ========================================================================

    int null_err = ann_predict_batch(NULL, (real[]){1,0}, (real[]){0}, 1);
    TESTEX("ann_predict_batch(NULL, ...) returns ERR_NULL_PTR", (null_err == ERR_NULL_PTR));

    int zero_batch = ann_predict_batch(net4, (real[]){1,0}, (real[]){0}, 0);
    TESTEX("ann_predict_batch(..., batch_size=0) returns error", (zero_batch != ERR_OK));

    int upload_null = ann_gpu_upload_network(NULL);
    TESTEX("ann_gpu_upload_network(NULL) returns ERR_NULL_PTR", (upload_null == ERR_NULL_PTR));
}

#else  /* USE_METAL not defined */

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    MODULE("Metal GPU Inference Tests");
    SUITE("Build Configuration");
    COMMENT("USE_METAL not defined - Metal GPU tests skipped");
    COMMENT("Build with: cmake -DUSE_METAL=1 .. to enable Metal GPU support");
}

#endif /* USE_METAL */
