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

#define GC_GPU_SUPPORTS_CNN
#include "gpu_consistency.h"

//-----------------------------------------------------------
// Helper: build and train a tiny XOR network (2-2-1)
//-----------------------------------------------------------
static PNetwork make_xor_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    if (!net) return NULL;

    ann_set_learning_rate(net, 0.01f);
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

// ann_print_props() output, for checking its Backend line
static char g_props[2048];
static void capture_props(const char *msg)
{
    size_t len = strlen(g_props);
    snprintf(g_props + len, sizeof(g_props) - len, "%s", msg);
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

//-----------------------------------------------------------
// CNN helpers. The network covers each layer kind and the window cases the
// kernels handle: a padded, strided conv, overlapping pool windows, a second
// conv, flatten and dense layers.
//-----------------------------------------------------------
#define CNN_SAMPLES 32
#define CNN_INPUTS  (2 * 9 * 9)
#define CNN_CLASSES 3

static PNetwork make_cnn(real conv_dropout)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    if (!net) return NULL;

    ann_add_input(net, 2, 9, 9);
    ann_add_conv2d(net, 4, 3, 2, 1, ACTIVATION_RELU);   // 4x5x5
    ann_add_maxpool2d(net, 3, 1);                       // 4x3x3, overlapping windows
    ann_add_conv2d(net, 5, 2, 1, 0, ACTIVATION_TANH);   // 5x2x2
    ann_add_flatten(net);                               // 20
    ann_add_layer(net, 6, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);     // run every epoch
    ann_set_epoch_limit(net, 5);
    ann_set_batch_size(net, 8);
    if (conv_dropout > 0.0f)
        ann_set_layer_dropout(net, 1, conv_dropout);
    return net;
}

static void make_cnn_data(PTensor *inputs, PTensor *targets)
{
    srand(1234);
    *inputs  = tensor_create(CNN_SAMPLES, CNN_INPUTS);
    *targets = tensor_zeros(CNN_SAMPLES, CNN_CLASSES);
    for (int i = 0; i < CNN_SAMPLES * CNN_INPUTS; i++)
        (*inputs)->values[i] = (real)rand() / (real)RAND_MAX;
    for (int i = 0; i < CNN_SAMPLES; i++)
        (*targets)->values[i * CNN_CLASSES + rand() % CNN_CLASSES] = 1.0f;
}

// Train with a fixed seed so weight init, shuffling and dropout masks match
static real train_cnn(PNetwork net, PTensor inputs, PTensor targets)
{
    srand(42);
    return ann_train_network(net, inputs, targets, CNN_SAMPLES);
}

// Compare every parameter, plus optimizer state when with_state is set
static int cnn_nets_close(PNetwork a, PNetwork b, real tol, int with_state)
{
    for (int l = 1; l < a->layer_count; l++) {
        PLayer la = &a->layers[l], lb = &b->layers[l];
        for (int p = 0; p < la->param_count; p++) {
            if (!gc_tensors_close(la->params[p].value, lb->params[p].value, tol))
                return 0;
            if (with_state && (!gc_tensors_close(la->params[p].m, lb->params[p].m, tol) ||
                               !gc_tensors_close(la->params[p].v, lb->params[p].v, tol)))
                return 0;
        }
    }
    return 1;
}

// The first parameter of the first layer that has any: what dispatch checks
static PTensor first_param(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        if (net->layers[l].param_count > 0)
            return net->layers[l].params[0].value;
    return NULL;
}

// GPU ann_predict()/ann_predict_batch() agree with a CPU forward pass over
// the same weights (forced by hiding the buffer dispatch checks)
static int cnn_gpu_matches_cpu(PNetwork net, const PTensor inputs, real tol)
{
    int in_nodes = net->layers[0].node_count;
    int out_nodes = net->layers[net->layer_count - 1].node_count;
    int n = inputs->rows * out_nodes;
    real *gpu = malloc(n * sizeof(real)), *gpu_batch = malloc(n * sizeof(real)), *cpu = malloc(n * sizeof(real));
    int ok = gpu && gpu_batch && cpu;

    if (ok) {
        for (int i = 0; i < inputs->rows; i++)
            ann_predict(net, &inputs->values[i * in_nodes], &gpu[i * out_nodes]);
        ok = ann_predict_batch(net, inputs->values, gpu_batch, inputs->rows) == ERR_OK;

        PTensor w = first_param(net);
        void *saved = w->gpu_buf;
        w->gpu_buf = NULL;
        for (int i = 0; i < inputs->rows; i++)
            ann_predict(net, &inputs->values[i * in_nodes], &cpu[i * out_nodes]);
        w->gpu_buf = saved;

        ok = ok && arrays_close(gpu, cpu, n, tol) && arrays_close(gpu_batch, cpu, n, tol);
    }
    free(gpu);
    free(gpu_batch);
    free(cpu);
    return ok;
}

// Fixed, varied values for every parameter of a layer
static void set_known_params(PNetwork net, int layer)
{
    for (int p = 0; p < ann_get_layer_param_count(net, layer); p++) {
        int r, c;
        ann_get_layer_param_shape(net, layer, p, &r, &c);
        real *v = malloc(r * c * sizeof(real));
        for (int i = 0; i < r * c; i++)
            v[i] = 0.3f * (real)sin(1.1 * i + 0.7 * layer + 1.9 * p);
        ann_set_layer_param(net, layer, p, v, r * c);
        free(v);
    }
}

// 1 if every parameter of a layer still has set_known_params' values
static int has_known_params(PNetwork net, int layer)
{
    int ok = 1;
    for (int p = 0; ok && p < ann_get_layer_param_count(net, layer); p++) {
        int r, c;
        ann_get_layer_param_shape(net, layer, p, &r, &c);
        real *v = malloc(r * c * sizeof(real));
        ok = v && ann_get_layer_param(net, layer, p, v, r * c) == ERR_OK;
        for (int i = 0; ok && i < r * c; i++)
            ok = v[i] == 0.3f * (real)sin(1.1 * i + 0.7 * layer + 1.9 * p);
        free(v);
    }
    return ok;
}

#define TOLERANCE 1e-4f

//-----------------------------------------------------------
// Text helpers: CNN_SAMPLES rows of TEXT_LEN token ids from a vocabulary of
// TEXT_VOCAB, 0 padding them (after the text, or before it in every third
// row; one row is all padding)
//-----------------------------------------------------------
#define TEXT_LEN   12
#define TEXT_VOCAB 20

static void make_text_data(PTensor *ids, PTensor *targets)
{
    srand(77);
    *ids = tensor_zeros(CNN_SAMPLES, TEXT_LEN);
    *targets = tensor_zeros(CNN_SAMPLES, CNN_CLASSES);
    for (int i = 0; i < CNN_SAMPLES; i++) {
        int len = i == 5 ? 0 : 1 + rand() % TEXT_LEN, start = i % 3 == 0 ? TEXT_LEN - len : 0;
        for (int t = 0; t < len; t++)
            (*ids)->values[i * TEXT_LEN + start + t] = (real)(1 + rand() % (TEXT_VOCAB - 1));
        (*targets)->values[i * CNN_CLASSES + rand() % CNN_CLASSES] = 1.0f;
    }
}

// 0: embedding -> conv1d -> global max pool; 1: masked embedding -> conv1d
// -> batch norm -> conv1d (whose windows read masked steps, which must be
// zero) -> max pool -> conv1d -> global average pool; 2: as 1 with global
// max pooling and dropout after the first conv
static PNetwork make_text_net(int variant)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, TEXT_LEN, LAYER_INPUT, ACTIVATION_NULL);
    if (variant == 0)
        ann_add_embedding(net, TEXT_VOCAB, 5, 0);
    else
        ann_add_embedding_masked(net, TEXT_VOCAB, 5, 0);   // 5x1x12
    ann_add_conv1d(net, 6, 3, 1, 1, ACTIVATION_RELU);      // 6x1x12
    if (variant > 0) {
        ann_add_batchnorm(net, ACTIVATION_RELU);
        ann_add_conv1d(net, 6, 3, 1, 1, ACTIVATION_RELU);  // 6x1x12
        ann_add_maxpool1d(net, 2, 2);                       // 6x1x6
        ann_add_conv1d(net, 4, 3, 1, 0, ACTIVATION_TANH);  // 4x1x4
    }
    if (variant == 1)
        ann_add_global_avgpool2d(net);
    else
        ann_add_global_maxpool2d(net);
    ann_add_layer(net, CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_epoch_limit(net, 4);
    ann_set_batch_size(net, 8);
    if (variant == 2)
        ann_set_layer_dropout(net, 2, 0.25f);
    return net;
}

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

    {
        Backend_info info;
        TESTEX("ann_get_backend_info reports Metal, started",
               ann_get_backend_info(&info) == ERR_OK && strcmp(info.gpu, "Metal") == 0 && info.gpu_active == 1);

        g_props[0] = '\0';
        PNetwork net = make_small_classifier();
        net->print_func = capture_props;
        ann_gpu_upload_network(net);
        ann_print_props(net);
        TESTEX("An uploaded network prints Backend: Metal GPU", strstr(g_props, "Metal GPU") != NULL);
        ann_free_network(net);
    }

    // ========================================================================
    SUITE("GPU Upload and Free");
    // ========================================================================

    PNetwork net = make_xor_net();
    TESTEX("Network creation succeeds", (net != NULL));

    int upload_ok = ann_gpu_upload_network(net);
    TESTEX("ann_gpu_upload_network() returns ERR_OK", (upload_ok == ERR_OK));
    TESTEX("Layer 1 weights have gpu_buf after upload",
           (net->layers[1].t_weights != NULL && net->layers[1].t_weights->gpu_buf != NULL));
    TESTEX("Layer 1 bias has gpu_buf after upload",
           (net->layers[1].t_bias != NULL && net->layers[1].t_bias->gpu_buf != NULL));

    ann_gpu_free_network(net);
    TESTEX("gpu_buf is NULL after ann_gpu_free_network()",
           (net->layers[1].t_weights->gpu_buf == NULL));

    ann_free_network(net);

    // ========================================================================
    SUITE("GPU Single-Sample Inference Matches CPU");
    // ========================================================================

    PNetwork net2 = make_xor_net();
    TESTEX("XOR network created", (net2 != NULL));

    real xor_tests[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    real gpu_out[1];

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

    // ========================================================================
    SUITE("GPU Training - Adam");
    // ========================================================================

    // Create fresh network (NOT pre-trained)
    PNetwork train_net = ann_make_network(OPT_ADAM, LOSS_MSE);
    ann_add_layer(train_net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(train_net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(train_net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(train_net, 0.01f);
    ann_set_convergence(train_net, 0.05f);
    ann_set_epoch_limit(train_net, 500);

    TESTEX("XOR network for GPU training created", (train_net != NULL));

    int train_upload = ann_gpu_upload_network(train_net);
    TESTEX("ann_gpu_upload_network() succeeds for training", (train_upload == ERR_OK));

    // XOR training data (2 inputs, 1 output, 4 samples)
    real xor_data[] = {
        0, 0,  0,
        0, 1,  1,
        1, 0,  1,
        1, 1,  0
    };
    PTensor xor_inputs = tensor_create(4, 2);
    PTensor xor_targets = tensor_create(4, 1);
    for (int i = 0; i < 4; i++) {
        xor_inputs->values[i * 2 + 0] = xor_data[i * 3 + 0];
        xor_inputs->values[i * 2 + 1] = xor_data[i * 3 + 1];
        xor_targets->values[i] = xor_data[i * 3 + 2];
    }

    real loss = ann_train_network(train_net, xor_inputs, xor_targets, 4);

    ann_gpu_sync_weights(train_net);
    TESTEX("GPU training completed with loss < 0.1", (loss < 0.1f));

    // Verify predictions after GPU training
    real pred[1];
    ann_predict(train_net, (real[]){0, 0}, pred);
    int pred_ok = (pred[0] < 0.5f); // XOR(0,0) = 0
    TESTEX("After GPU training, XOR(0,0) predicts 0", (pred_ok));

    tensor_free(xor_inputs);
    tensor_free(xor_targets);
    ann_free_network(train_net);

    // ========================================================================
    SUITE("GPU Sync Weights");
    // ========================================================================

    PNetwork sync_net = ann_make_network(OPT_ADAM, LOSS_MSE);
    ann_add_layer(sync_net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(sync_net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(sync_net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(sync_net, 0.01f);
    ann_set_convergence(sync_net, 0.05f);
    ann_set_epoch_limit(sync_net, 500);

    TESTEX("Network for sync test created", (sync_net != NULL));

    ann_gpu_upload_network(sync_net);
    
    // Create small training set
    PTensor sync_inputs = tensor_create(4, 2);
    PTensor sync_targets = tensor_create(4, 1);
    for (int i = 0; i < 4; i++) {
        sync_inputs->values[i * 2 + 0] = xor_data[i * 3 + 0];
        sync_inputs->values[i * 2 + 1] = xor_data[i * 3 + 1];
        sync_targets->values[i] = xor_data[i * 3 + 2];
    }

    ann_train_network(sync_net, sync_inputs, sync_targets, 4);
    ann_gpu_sync_weights(sync_net);

    // After sync, CPU predict should work correctly
    real sync_pred[1];
    ann_predict(sync_net, (real[]){1, 1}, sync_pred);
    int sync_ok = (sync_pred[0] < 0.5f); // XOR(1,1) = 0
    TESTEX("After GPU train + sync, CPU predict works", (sync_ok));

    tensor_free(sync_inputs);
    tensor_free(sync_targets);
    ann_free_network(sync_net);

    // ========================================================================
    SUITE("GPU CNN Training and Inference");
    // ========================================================================
    {
        PTensor inputs, targets;
        make_cnn_data(&inputs, &targets);

        PNetwork cpu_net = make_cnn(0.0f), gpu_net = make_cnn(0.0f);
        TESTEX("Uploading a CNN returns ERR_OK", ann_gpu_upload_network(gpu_net) == ERR_OK);
        TESTEX("and puts the conv weights on the GPU", gpu_net->layers[1].params[0].value->gpu_buf != NULL);

        real cpu_loss = train_cnn(cpu_net, inputs, targets);
        real gpu_loss = train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU CNN training loss is finite and matches the CPU's",
               isfinite(gpu_loss) && fabsf(gpu_loss - cpu_loss) < 1e-3f);
        TESTEX("GPU CNN training matches CPU training (weights and Adam state)",
               cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("GPU CNN inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // dropout on the first conv layer: masks drawn on the CPU, applied on the GPU
        cpu_net = make_cnn(0.25f);
        gpu_net = make_cnn(0.25f);
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU CNN dropout training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // a wide conv (32 filters over 32 channels): its weight gradient is
        // large enough to use the MPS GEMM rather than split sums
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork wide[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(wide[i], 2, 9, 9);
            ann_add_conv2d(wide[i], 32, 3, 1, 1, ACTIVATION_RELU);  // 32x9x9
            ann_add_conv2d(wide[i], 32, 3, 2, 0, ACTIVATION_RELU);  // 32x4x4, 32*32*3*3 weights
            ann_add_layer(wide[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(wide[i], 0.001f);
            ann_set_convergence(wide[i], 0.0f);
            ann_set_epoch_limit(wide[i], 3);
            ann_set_batch_size(wide[i], 8);
        }
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU wide-conv training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("GPU wide-conv inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // split-sum weight gradients with wide patches: 20 filters over 32*3*3
        // patches (patches split across threadgroups, a part block of filters)
        // and 6 filters, not a multiple of 4, over 20*3*3
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork split[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(split[i], 2, 9, 9);
            ann_add_conv2d(split[i], 32, 3, 1, 1, ACTIVATION_RELU);  // 32x9x9
            ann_add_conv2d(split[i], 20, 3, 1, 1, ACTIVATION_RELU);  // 20x9x9, 20*32*3*3 weights
            ann_add_conv2d(split[i], 6, 3, 2, 0, ACTIVATION_RELU);   // 6x4x4, 6*20*3*3 weights
            ann_add_layer(split[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(split[i], 0.001f);
            ann_set_convergence(split[i], 0.0f);
            ann_set_epoch_limit(split[i], 3);
            ann_set_batch_size(split[i], 8);
        }
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU split-sum conv training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // each activation, applied as conv and dense layers add their bias
        // (GELU also saves the values before activation for its derivative)
        const Activation_type acts[] = { ACTIVATION_GELU, ACTIVATION_LEAKY_RELU, ACTIVATION_SOFTSIGN,
                                         ACTIVATION_TANH, ACTIVATION_SIGMOID };
        const char *act_names[] = { "GELU", "leaky ReLU", "softsign", "tanh", "sigmoid" };
        int acts_match = 1;
        for (int a = 0; a < 5; a++) {
            cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
            gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
            PNetwork act_nets[2] = { cpu_net, gpu_net };
            for (int i = 0; i < 2; i++) {
                ann_add_input(act_nets[i], 2, 9, 9);
                ann_add_conv2d(act_nets[i], 4, 3, 2, 1, acts[a]);   // 4x5x5
                ann_add_flatten(act_nets[i]);
                ann_add_layer(act_nets[i], 6, LAYER_HIDDEN, acts[a]);
                ann_add_layer(act_nets[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
                ann_set_learning_rate(act_nets[i], 0.01f);
                ann_set_convergence(act_nets[i], 0.0f);
                ann_set_epoch_limit(act_nets[i], 3);
                ann_set_batch_size(act_nets[i], 8);
            }
            ann_gpu_upload_network(gpu_net);
            train_cnn(cpu_net, inputs, targets);
            train_cnn(gpu_net, inputs, targets);
            if (!cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1)) {
                printf("\t(%s training differs)\n", act_names[a]);
                acts_match = 0;
            }
            ann_free_network(cpu_net);
            ann_free_network(gpu_net);
        }
        TESTEX("GPU conv and dense training matches the CPU's with each activation", acts_match);

        // input noise is drawn on the CPU for each batch, so both paths see the same inputs
        cpu_net = make_cnn(0.0f);
        gpu_net = make_cnn(0.0f);
        ann_set_input_noise(cpu_net, NOISE_GAUSSIAN, 0.3f, 0.0f, 1.0f);
        ann_set_input_noise(gpu_net, NOISE_GAUSSIAN, 0.3f, 0.0f, 1.0f);
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU training with input noise matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // a conv autoencoder (pool 3, upsample 3) trained to reproduce its
        // inputs, with MSE and with binary cross-entropy (the output conv
        // layer's delta then has no activation derivative)
        const Loss_type ae_losses[] = { LOSS_MSE, LOSS_BINARY_CROSS_ENTROPY };
        for (int k = 0; k < 2; k++) {
            cpu_net = ann_make_network(OPT_ADAM, ae_losses[k]);
            gpu_net = ann_make_network(OPT_ADAM, ae_losses[k]);
            PNetwork ae[2] = { cpu_net, gpu_net };
            for (int i = 0; i < 2; i++) {
                ann_add_input(ae[i], 2, 9, 9);
                ann_add_conv2d(ae[i], 4, 3, 1, 1, ACTIVATION_RELU);     // 4x9x9
                ann_add_maxpool2d(ae[i], 3, 3);                         // 4x3x3
                ann_add_upsample2d(ae[i], 3);                           // 4x9x9
                ann_add_conv2d(ae[i], 2, 3, 1, 1, ACTIVATION_SIGMOID);  // 2x9x9
                ann_set_learning_rate(ae[i], 0.01f);
                ann_set_convergence(ae[i], 0.0f);
                ann_set_epoch_limit(ae[i], 3);
                ann_set_batch_size(ae[i], 8);
            }
            int uploaded = ann_gpu_upload_network(gpu_net) == ERR_OK;
            if (k == 0) {
                TESTEX("A network with upsampling uploads", uploaded);
            }
            train_cnn(cpu_net, inputs, inputs);
            train_cnn(gpu_net, inputs, inputs);
            if (k == 0) {
                TESTEX("GPU autoencoder training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
                TESTEX("GPU autoencoder inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
            } else {
                TESTEX("GPU autoencoder training with cross-entropy matches CPU training",
                       uploaded && cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
            }
            ann_free_network(cpu_net);
            ann_free_network(gpu_net);
        }

        // a dense bottleneck reshaped to an image, upsampled and convolved
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork bottleneck[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(bottleneck[i], 2, 9, 9);
            ann_add_layer(bottleneck[i], 8, LAYER_HIDDEN, ACTIVATION_TANH);
            ann_add_reshape(bottleneck[i], 2, 2, 2);
            ann_add_upsample2d(bottleneck[i], 2);                         // 2x4x4
            ann_add_conv2d(bottleneck[i], 3, 3, 1, 1, ACTIVATION_RELU);   // 3x4x4
            ann_add_layer(bottleneck[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(bottleneck[i], 0.01f);
            ann_set_convergence(bottleneck[i], 0.0f);
            ann_set_epoch_limit(bottleneck[i], 3);
            ann_set_batch_size(bottleneck[i], 8);
        }
        TESTEX("A network with a reshape uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU dense -> reshape -> upsample training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // global average pooling in place of pool + flatten
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork gap[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(gap[i], 2, 9, 9);
            ann_add_conv2d(gap[i], 4, 3, 1, 1, ACTIVATION_RELU);   // 4x9x9
            ann_add_conv2d(gap[i], 6, 3, 2, 0, ACTIVATION_TANH);   // 6x4x4
            ann_add_global_avgpool2d(gap[i]);                      // 6x1x1
            ann_add_layer(gap[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(gap[i], 0.01f);
            ann_set_convergence(gap[i], 0.0f);
            ann_set_epoch_limit(gap[i], 3);
            ann_set_batch_size(gap[i], 8);
        }
        TESTEX("A network with global average pooling uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU conv -> global avg pool training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // batch norm after a conv (relu) and a dense layer (gelu), trained with
        // the batch's statistics; inference uses the running ones. SGD, since
        // a bias before a batch norm (and here the first batch norm's beta,
        // which the second cancels) gets only rounding noise as its gradient,
        // and Adam scales that noise up to full steps that differ by path.
        cpu_net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork bn[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(bn[i], 2, 9, 9);
            ann_add_conv2d(bn[i], 4, 3, 1, 1, ACTIVATION_NULL);    // 4x9x9
            ann_add_batchnorm(bn[i], ACTIVATION_RELU);
            ann_add_maxpool2d(bn[i], 3, 3);                         // 4x3x3
            ann_add_layer(bn[i], 8, LAYER_HIDDEN, ACTIVATION_NULL);
            ann_add_batchnorm(bn[i], ACTIVATION_GELU);
            ann_add_layer(bn[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(bn[i], 0.05f);
            ann_set_convergence(bn[i], 0.0f);
            ann_set_epoch_limit(bn[i], 3);
            ann_set_batch_size(bn[i], 8);
        }
        TESTEX("A network with batch norm uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU batch norm training (weights and running statistics) matches CPU training",
               cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // batch norm as the output layer with MSE, and straight off the input
        cpu_net = ann_make_network(OPT_SGD, LOSS_MSE);
        gpu_net = ann_make_network(OPT_SGD, LOSS_MSE);
        PNetwork bn_out[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(bn_out[i], 2, 9, 9);
            ann_add_batchnorm(bn_out[i], ACTIVATION_NULL);
            ann_add_conv2d(bn_out[i], 3, 3, 2, 0, ACTIVATION_TANH);  // 3x4x4
            ann_add_layer(bn_out[i], CNN_CLASSES, LAYER_HIDDEN, ACTIVATION_NULL);
            ann_add_batchnorm(bn_out[i], ACTIVATION_SIGMOID);
            ann_set_learning_rate(bn_out[i], 0.05f);
            ann_set_convergence(bn_out[i], 0.0f);
            ann_set_epoch_limit(bn_out[i], 3);
            ann_set_batch_size(bn_out[i], 8);
        }
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU batch norm on the input and as an MSE output matches CPU training",
               cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1) && cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // grouped and depthwise convolutions: a depthwise conv (x2, stride 2),
        // a grouped 1x1 conv, and a depthwise conv straight off the input
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork dw[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(dw[i], 2, 9, 9);
            ann_add_depthwise_conv2d(dw[i], 2, 3, 1, 1, ACTIVATION_RELU);     // 4x9x9
            ann_add_conv2d(dw[i], 6, 3, 1, 1, ACTIVATION_GELU);               // 6x9x9
            ann_add_depthwise_conv2d(dw[i], 2, 3, 2, 1, ACTIVATION_TANH);     // 12x5x5
            ann_add_conv2d_grouped(dw[i], 6, 1, 1, 0, 3, ACTIVATION_RELU);    // 6x5x5
            ann_add_global_avgpool2d(dw[i]);
            ann_add_layer(dw[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(dw[i], 0.01f);
            ann_set_convergence(dw[i], 0.0f);
            ann_set_epoch_limit(dw[i], 3);
            ann_set_batch_size(dw[i], 8);
        }
        TESTEX("A network with grouped and depthwise convs uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU grouped/depthwise conv training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // non-square windows: a 2x3 conv with strides (1, 2) and padding (1, 0),
        // a 3x1 pool, 1-D convs along the width (one grouped) and a 1-D pool
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork rect[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(rect[i], 2, 9, 9);
            ann_add_conv2d_ex(rect[i], 6, 2, 3, 1, 2, 1, 0, 1, ACTIVATION_RELU);   // 6x10x4
            ann_add_maxpool2d_ex(rect[i], 3, 1, 3, 1);                             // 6x3x4
            ann_add_conv1d(rect[i], 4, 3, 1, 1, ACTIVATION_TANH);                   // 4x3x4
            ann_add_conv2d_ex(rect[i], 4, 1, 3, 1, 2, 0, 1, 2, ACTIVATION_GELU);   // grouped: 4x3x2
            ann_add_maxpool1d(rect[i], 2, 1);                                       // 4x3x1
            ann_add_layer(rect[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(rect[i], 0.01f);
            ann_set_convergence(rect[i], 0.0f);
            ann_set_epoch_limit(rect[i], 3);
            ann_set_batch_size(rect[i], 8);
        }
        TESTEX("A network with non-square windows uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU non-square and 1-D conv training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // a wider conv (8 filters, past the direct-conv cutoffs) with a 1x5 window
        // and a 3x2 pool whose windows tile the input
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork wide_rect[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(wide_rect[i], 2, 9, 9);
            ann_add_conv2d_ex(wide_rect[i], 20, 1, 5, 1, 1, 0, 2, 1, ACTIVATION_RELU);  // 20x9x9
            ann_add_maxpool2d_ex(wide_rect[i], 3, 2, 3, 2);                             // 20x3x4
            ann_add_conv2d_ex(wide_rect[i], 12, 3, 2, 1, 1, 1, 0, 1, ACTIVATION_TANH);  // 12x3x3
            ann_add_layer(wide_rect[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(wide_rect[i], 0.01f);
            ann_set_convergence(wide_rect[i], 0.0f);
            ann_set_epoch_limit(wide_rect[i], 3);
            ann_set_batch_size(wide_rect[i], 8);
        }
        ann_gpu_upload_network(gpu_net);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU non-square im2col convs and a tiled 3x2 pool match CPU training",
               cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1) && cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // global max pooling over 1-D conv features and over an image
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork gmp[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(gmp[i], 2, 9, 9);
            ann_add_conv2d(gmp[i], 6, 3, 1, 1, ACTIVATION_RELU);    // 6x9x9
            ann_add_conv1d(gmp[i], 5, 3, 2, 1, ACTIVATION_TANH);    // 5x9x5
            ann_add_global_maxpool2d(gmp[i]);                       // 5
            ann_add_layer(gmp[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(gmp[i], 0.01f);
            ann_set_convergence(gmp[i], 0.0f);
            ann_set_epoch_limit(gmp[i], 3);
            ann_set_batch_size(gmp[i], 8);
        }
        TESTEX("A network with global max pooling uploads", ann_gpu_upload_network(gpu_net) == ERR_OK);
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU global max pool training matches CPU training", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        TESTEX("and its inference matches the CPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // embeddings, with and without padding masks
        {
            PTensor ids, text_targets;
            make_text_data(&ids, &text_targets);
            int uploaded[3], trained[3], inferred[3];
            for (int v = 0; v < 3; v++) {
                PNetwork cpu_text = make_text_net(v), gpu_text = make_text_net(v);
                uploaded[v] = ann_gpu_upload_network(gpu_text) == ERR_OK;
                srand(42);
                ann_train_network(cpu_text, ids, text_targets, CNN_SAMPLES);
                srand(42);
                ann_train_network(gpu_text, ids, text_targets, CNN_SAMPLES);
                trained[v] = cnn_nets_close(cpu_text, gpu_text, 1e-3f, 1);
                inferred[v] = cnn_gpu_matches_cpu(gpu_text, ids, TOLERANCE);
                ann_free_network(cpu_text);
                ann_free_network(gpu_text);
            }
            TESTEX("A network with an embedding uploads", uploaded[0] && uploaded[1] && uploaded[2]);
            TESTEX("GPU embedding training matches CPU training", trained[0]);
            TESTEX("and its inference matches the CPU", inferred[0]);
            TESTEX("GPU masked embedding, batch norm, pooling and global average pool training matches the CPU", trained[1]);
            TESTEX("and its inference matches the CPU", inferred[1]);
            TESTEX("GPU masked training with dropout and global max pooling matches the CPU", trained[2]);
            TESTEX("and its inference matches the CPU", inferred[2]);
            tensor_free(ids);
            tensor_free(text_targets);

            // ids that come from another layer stay on the CPU
            PNetwork emb = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
            ann_add_layer(emb, 8, LAYER_INPUT, ACTIVATION_NULL);
            ann_add_layer(emb, 8, LAYER_HIDDEN, ACTIVATION_RELU);
            ann_add_embedding(emb, 20, 4, 0);
            ann_add_global_maxpool2d(emb);
            ann_add_layer(emb, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            TESTEX("An embedding above layer 1 is not uploaded (it trains on the CPU)",
                   ann_gpu_upload_network(emb) == ERR_INVALID);
            ann_free_network(emb);
        }

        // pooling straight off the input: the first layer has no parameters
        PNetwork pool_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_input(pool_net, 2, 9, 9);
        ann_add_maxpool2d(pool_net, 2, 2);                  // 2x4x4
        ann_add_layer(pool_net, CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
        ann_set_epoch_limit(pool_net, 2);
        train_cnn(pool_net, inputs, targets);
        TESTEX("A network starting with a pool layer uploads", ann_gpu_upload_network(pool_net) == ERR_OK);
        TESTEX("and its GPU inference matches the CPU", cnn_gpu_matches_cpu(pool_net, inputs, TOLERANCE));
        ann_free_network(pool_net);

        // non-overlapping pooling that leaves the last input row and column
        // out of every window: their gradients must be zero
        cpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        gpu_net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        PNetwork edge[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            ann_add_input(edge[i], 2, 9, 9);
            ann_add_conv2d(edge[i], 4, 3, 1, 1, ACTIVATION_RELU);   // 4x9x9
            ann_add_maxpool2d(edge[i], 2, 2);                       // 4x4x4
            ann_add_flatten(edge[i]);
            ann_add_layer(edge[i], CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(edge[i], 0.01f);
            ann_set_convergence(edge[i], 0.0f);
            ann_set_epoch_limit(edge[i], 3);
            ann_set_batch_size(edge[i], 8);
        }
        int edge_uploaded = ann_gpu_upload_network(gpu_net) == ERR_OK;
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("GPU training through a 2x2 pool with uncovered edges matches CPU training",
               edge_uploaded && cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        // frozen layers (a conv and a dense one) stay as set while the rest
        // trains on the GPU, as on the CPU
        cpu_net = make_cnn(0.0f);
        gpu_net = make_cnn(0.0f);
        PNetwork frozen[2] = { cpu_net, gpu_net };
        for (int i = 0; i < 2; i++) {
            set_known_params(frozen[i], 1);
            set_known_params(frozen[i], 5);
            ann_set_layer_trainable(frozen[i], 1, 0);
            ann_set_layer_trainable(frozen[i], 5, 0);
        }
        int frozen_uploaded = ann_gpu_upload_network(gpu_net) == ERR_OK;
        train_cnn(cpu_net, inputs, targets);
        train_cnn(gpu_net, inputs, targets);
        TESTEX("Frozen layers keep their values through GPU training",
               frozen_uploaded && has_known_params(gpu_net, 1) && has_known_params(gpu_net, 5));
        TESTEX("and the layers around them train as on the CPU", cnn_nets_close(cpu_net, gpu_net, 1e-3f, 1));

        // parameters set while the network is uploaded reach the GPU
        set_known_params(gpu_net, 6);
        TESTEX("Parameters set on an uploaded network reach the GPU", cnn_gpu_matches_cpu(gpu_net, inputs, TOLERANCE));
        ann_free_network(cpu_net);
        ann_free_network(gpu_net);

        tensor_free(inputs);
        tensor_free(targets);

        // a batch bigger than one forward pass's memory cap (about 600 rows
        // of this network) is evaluated in pieces
        enum { BIG_ROWS = 1000, BIG_IN = 48 * 48 };
        PTensor big_in = tensor_create(BIG_ROWS, BIG_IN), big_out = tensor_zeros(BIG_ROWS, CNN_CLASSES);
        srand(5);
        for (int i = 0; i < BIG_ROWS * BIG_IN; i++)
            big_in->values[i] = (real)rand() / (real)RAND_MAX;
        for (int i = 0; i < BIG_ROWS; i++)
            big_out->values[i * CNN_CLASSES + rand() % CNN_CLASSES] = 1.0f;
        PNetwork big = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_input(big, 1, 48, 48);
        ann_add_conv2d(big, 16, 3, 1, 1, ACTIVATION_RELU);     // 16x48x48
        ann_add_maxpool2d(big, 2, 2);                          // 16x24x24
        ann_add_layer(big, CNN_CLASSES, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
        ann_set_epoch_limit(big, 1);
        ann_set_batch_size(big, 50);
        ann_train_network(big, big_in, big_out, BIG_ROWS);
        ann_gpu_upload_network(big);
        TESTEX("A large GPU batch is evaluated in pieces and matches the CPU",
               cnn_gpu_matches_cpu(big, big_in, TOLERANCE));
        ann_free_network(big);
        tensor_free(big_in);
        tensor_free(big_out);

        // GEMMs with a dimension over 16384 run as blocks: MPS on Intel GPUs
        // runs some large ones through textures and asserts when they are more
        // than 16384 high (this conv's shape does, unsplit). 300 texts of 200
        // ids make a conv GEMM of 60000 x 320 x 128 when predicting, and
        // batches of 256 one of 51200 rows forward and over 51200 positions
        // for the weight gradient
        enum { TEXTS = 300, STEPS = 200 };
        PTensor ids = tensor_zeros(TEXTS, STEPS), labels = tensor_zeros(TEXTS, 2);
        srand(11);
        for (int i = 0; i < TEXTS; i++) {
            int len = 20 + rand() % (STEPS - 20);
            for (int t = 0; t < len; t++)
                ids->values[i * STEPS + t] = (real)(1 + rand() % 499);
            labels->values[i * 2 + rand() % 2] = 1.0f;
        }
        PNetwork long_text[2];
        for (int i = 0; i < 2; i++) {
            long_text[i] = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
            ann_add_layer(long_text[i], STEPS, LAYER_INPUT, ACTIVATION_NULL);
            ann_add_embedding_masked(long_text[i], 500, 64, 0);              // 64x1x200
            ann_add_conv1d(long_text[i], 128, 5, 1, 2, ACTIVATION_RELU);     // im2col 200 x 320 per text: 65 MB for 256
            ann_add_global_maxpool2d(long_text[i]);
            ann_add_layer(long_text[i], 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(long_text[i], 0.01f);
            ann_set_convergence(long_text[i], 0.0f);
            ann_set_epoch_limit(long_text[i], 2);
            ann_set_batch_size(long_text[i], 256);
        }
        int long_uploaded = ann_gpu_upload_network(long_text[1]) == ERR_OK;
        srand(42);
        ann_train_network(long_text[0], ids, labels, TEXTS);
        srand(42);
        ann_train_network(long_text[1], ids, labels, TEXTS);
        TESTEX("GPU training in batches of 256 texts of 200 steps (GEMMs split into blocks) matches the CPU",
               long_uploaded && cnn_nets_close(long_text[0], long_text[1], 1e-3f, 0));
        TESTEX("and predicting all 300 at once matches the CPU", cnn_gpu_matches_cpu(long_text[1], ids, TOLERANCE));
        ann_free_network(long_text[0]);
        ann_free_network(long_text[1]);
        tensor_free(ids);
        tensor_free(labels);

        // a dense layer of 20000 nodes: split columns forward, rows and the
        // inner dimension backward
        PTensor wide_in = tensor_create(64, 8), wide_out = tensor_zeros(64, 2);
        for (int i = 0; i < 64 * 8; i++)
            wide_in->values[i] = (real)rand() / (real)RAND_MAX;
        for (int i = 0; i < 64; i++)
            wide_out->values[i * 2 + i % 2] = 1.0f;
        PNetwork wide_net[2];
        for (int i = 0; i < 2; i++) {
            wide_net[i] = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
            ann_add_layer(wide_net[i], 8, LAYER_INPUT, ACTIVATION_NULL);
            ann_add_layer(wide_net[i], 20000, LAYER_HIDDEN, ACTIVATION_RELU);
            ann_add_layer(wide_net[i], 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
            ann_set_learning_rate(wide_net[i], 0.001f);
            ann_set_convergence(wide_net[i], 0.0f);
            ann_set_epoch_limit(wide_net[i], 2);
            ann_set_batch_size(wide_net[i], 32);
        }
        int wide_uploaded = ann_gpu_upload_network(wide_net[1]) == ERR_OK;
        srand(42);
        ann_train_network(wide_net[0], wide_in, wide_out, 64);
        srand(42);
        ann_train_network(wide_net[1], wide_in, wide_out, 64);
        TESTEX("GPU training with a 20000-node dense layer (GEMMs split into blocks) matches the CPU",
               wide_uploaded && cnn_nets_close(wide_net[0], wide_net[1], 1e-3f, 0));
        TESTEX("and so does its inference", cnn_gpu_matches_cpu(wide_net[1], wide_in, TOLERANCE));
        ann_free_network(wide_net[0]);
        ann_free_network(wide_net[1]);
        tensor_free(wide_in);
        tensor_free(wide_out);
    }

    run_gpu_consistency_tests();
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
