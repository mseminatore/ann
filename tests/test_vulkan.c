/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* See LICENSE file for terms.                                                    */
/**********************************************************************************/

//
// test_vulkan.c - Unit tests for the Vulkan GPU inference + training backend
//
// Only built and run when USE_VULKAN is defined (cmake -DUSE_VULKAN=1).
// Tests verify that GPU inference/training produces results matching the CPU path.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

#ifdef USE_VULKAN

#include "gpu_consistency.h"

//-----------------------------------------------------------
// Helper: build and train a tiny XOR network (2-4-1)
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

static void quiet(const char *s) { (void)s; }

static real rand_unit(void)
{
    return (real)rand() / (real)RAND_MAX;
}

// A dense network with the given layer sizes; weights drawn from rand() after srand(seed),
// so two calls with the same arguments build the same network
static PNetwork make_dense_net(unsigned seed, Optimizer_type opt, Loss_type loss, const int *nodes, int layers,
                               Activation_type hidden, Activation_type output)
{
    srand(seed);
    PNetwork net = ann_make_network(opt, loss);
    if (!net) return NULL;
    net->print_func = quiet;
    for (int i = 0; i < layers; i++)
        ann_add_layer(net, nodes[i], i == 0 ? LAYER_INPUT : (i == layers - 1 ? LAYER_OUTPUT : LAYER_HIDDEN),
                      i == 0 ? ACTIVATION_NULL : (i == layers - 1 ? output : hidden));
    return net;
}

// rows x cols tensor of uniform [0, 1) values, or one-hot rows when one_hot is set
static PTensor random_data(int rows, int cols, int one_hot)
{
    PTensor t = tensor_create(rows, cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            t->values[r * cols + c] = one_hot ? (real)(c == r % cols) : rand_unit();
    return t;
}

// What ann_predict_batch() returns now (on the GPU) against a CPU forward pass over the CPU
// weights, forced by hiding layer 1's weight buffer, which is what dispatch checks
static real max_gpu_cpu_diff(PNetwork net, const real *inputs, int rows)
{
    int out = net->layers[net->layer_count - 1].node_count;
    int in = net->layers[0].node_count;
    real *gpu = (real *)malloc((size_t)rows * out * sizeof(real));
    real *cpu = (real *)malloc((size_t)rows * out * sizeof(real));

    ann_predict_batch(net, inputs, gpu, rows);
    void *saved = net->layers[1].t_weights->gpu_buf;
    net->layers[1].t_weights->gpu_buf = NULL;
    for (int r = 0; r < rows; r++)
        ann_predict(net, inputs + (size_t)r * in, cpu + (size_t)r * out);
    net->layers[1].t_weights->gpu_buf = saved;

    real worst = 0.0f;
    for (int i = 0; i < rows * out; i++)
    {
        real d = fabsf(gpu[i] - cpu[i]);
        if (!(d <= worst)) worst = d;   // NaN counts as the worst
    }
    free(gpu);
    free(cpu);
    return worst;
}

static real max_weight_diff(PNetwork a, PNetwork b)
{
    real worst = 0.0f;
    for (int l = 1; l < a->layer_count; l++)
    {
        PTensor ts[2][2] = { { a->layers[l].t_weights, b->layers[l].t_weights },
                             { a->layers[l].t_bias, b->layers[l].t_bias } };
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < ts[k][0]->rows * ts[k][0]->cols; i++)
            {
                real d = fabsf(ts[k][0]->values[i] - ts[k][1]->values[i]);
                if (!(d <= worst)) worst = d;
            }
    }
    return worst;
}

// Train the same network on the CPU and on the GPU from the same seed; returns the largest
// weight difference and sets the two final losses
static real train_cpu_and_gpu(Optimizer_type opt, Loss_type loss, const int *nodes, int layers,
                              Activation_type hidden, Activation_type output,
                              real l2, real l1, real clip, real *cpu_loss, real *gpu_loss)
{
    int in = nodes[0], out = nodes[layers - 1], rows = 85;
    srand(4242);
    PTensor x = random_data(rows, in, 0);
    PTensor y = random_data(rows, out, loss == LOSS_CATEGORICAL_CROSS_ENTROPY && output == ACTIVATION_SOFTMAX);

    PNetwork nets[2];
    real losses[2];
    for (int i = 0; i < 2; i++)
    {
        nets[i] = make_dense_net(77, opt, loss, nodes, layers, hidden, output);
        ann_set_learning_rate(nets[i], 0.01f);
        ann_set_batch_size(nets[i], 17);         // not a multiple of any tile size
        ann_set_epoch_limit(nets[i], 4);
        ann_set_convergence(nets[i], 1e-9f);     // run every epoch
        if (l2 > 0.0f) ann_set_weight_decay(nets[i], l2);
        if (l1 > 0.0f) ann_set_l1_regularization(nets[i], l1);
        if (clip > 0.0f) ann_set_gradient_clip(nets[i], clip);
        if (i == 1) ann_gpu_upload_network(nets[i]);
        srand(99);
        losses[i] = ann_train_network(nets[i], x, y, rows);
    }
    real diff = (nets[1]->layers[1].t_weights->gpu_buf && !nets[1]->gpu_stale) ? max_weight_diff(nets[0], nets[1])
                                                                                 : 1e9f;   // did not train on the GPU
    *cpu_loss = losses[0];
    *gpu_loss = losses[1];
    ann_free_network(nets[0]);
    ann_free_network(nets[1]);
    tensor_free(x);
    tensor_free(y);
    return diff;
}

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    MODULE("Vulkan GPU Backend Tests");

    // ========================================================================
    SUITE("GPU Initialization");
    // ========================================================================

    int gpu_ok = ann_gpu_init();
    TESTEX("ann_gpu_init() succeeds on Vulkan-capable hardware", (gpu_ok == 1));

    if (!gpu_ok)
    {
        COMMENT("Vulkan GPU not available - skipping remaining GPU tests");
        return;
    }

    Backend_info info;
    ann_get_backend_info(&info);
    TESTEX("ann_get_backend_info() reports the running Vulkan backend",
           strcmp(info.gpu, "Vulkan") == 0 && info.gpu_active);

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

    // CPU baseline
    real cpu_results[4];
    for (int i = 0; i < 4; i++)
        ann_predict(net2, xor_tests[i], cpu_results + i);

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

    real cpu_batch[12];
    for (int i = 0; i < n_samples; i++)
        ann_predict(net3, test_inputs + i * 4, cpu_batch + i * out_nodes);

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

    real fallback_out[4];
    int fallback_err = ann_predict_batch(net4, (real[]){0,0, 0,1, 1,0, 1,1},
                                         fallback_out, 4);
    TESTEX("ann_predict_batch() with no GPU upload falls back to CPU (ERR_OK)",
           (fallback_err == ERR_OK));

    // ========================================================================
    SUITE("Error Handling");
    // ========================================================================

    int null_err = ann_predict_batch(NULL, (real[]){1,0}, (real[]){0}, 1);
    TESTEX("ann_predict_batch(NULL, ...) returns ERR_NULL_PTR", (null_err == ERR_NULL_PTR));

    int zero_batch = ann_predict_batch(net4, (real[]){1,0}, (real[]){0}, 0);
    TESTEX("ann_predict_batch(..., batch_size=0) returns error", (zero_batch != ERR_OK));

    int upload_null = ann_gpu_upload_network(NULL);
    TESTEX("ann_gpu_upload_network(NULL) returns ERR_NULL_PTR", (upload_null == ERR_NULL_PTR));

    ann_free_network(net4);

    // ========================================================================
    SUITE("GPU Training - Adam (XOR converges)");
    // ========================================================================

    PNetwork train_net = ann_make_network(OPT_ADAM, LOSS_MSE);
    ann_add_layer(train_net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(train_net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(train_net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(train_net, 0.01f);
    ann_set_convergence(train_net, 0.05f);
    ann_set_epoch_limit(train_net, 5000);

    TESTEX("XOR network for GPU training created", (train_net != NULL));

    int train_upload = ann_gpu_upload_network(train_net);
    TESTEX("ann_gpu_upload_network() succeeds for training", (train_upload == ERR_OK));

    real xor_data[] = {
        0, 0,  0,
        0, 1,  1,
        1, 0,  1,
        1, 1,  0
    };
    PTensor xor_inputs  = tensor_create(4, 2);
    PTensor xor_targets = tensor_create(4, 1);
    for (int i = 0; i < 4; i++) {
        xor_inputs->values[i * 2 + 0] = xor_data[i * 3 + 0];
        xor_inputs->values[i * 2 + 1] = xor_data[i * 3 + 1];
        xor_targets->values[i]        = xor_data[i * 3 + 2];
    }

    real loss = ann_train_network(train_net, xor_inputs, xor_targets, 4);
    ann_gpu_sync_weights(train_net);
    TESTEX("GPU training completed with loss < 0.1", (loss < 0.1f));

    real pred[1];
    ann_predict(train_net, (real[]){0, 0}, pred);
    TESTEX("After GPU training, XOR(0,0) predicts 0", (pred[0] < 0.5f));

    tensor_free(xor_inputs);
    tensor_free(xor_targets);
    ann_free_network(train_net);

    // ========================================================================
    SUITE("GPU Train + Sync -> CPU Predict");
    // ========================================================================

    PNetwork sync_net = ann_make_network(OPT_ADAM, LOSS_MSE);
    ann_add_layer(sync_net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(sync_net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(sync_net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(sync_net, 0.01f);
    ann_set_convergence(sync_net, 0.05f);
    ann_set_epoch_limit(sync_net, 5000);

    TESTEX("Network for sync test created", (sync_net != NULL));

    ann_gpu_upload_network(sync_net);

    PTensor sync_inputs  = tensor_create(4, 2);
    PTensor sync_targets = tensor_create(4, 1);
    for (int i = 0; i < 4; i++) {
        sync_inputs->values[i * 2 + 0] = xor_data[i * 3 + 0];
        sync_inputs->values[i * 2 + 1] = xor_data[i * 3 + 1];
        sync_targets->values[i]        = xor_data[i * 3 + 2];
    }

    ann_train_network(sync_net, sync_inputs, sync_targets, 4);
    ann_gpu_sync_weights(sync_net);

    real sync_pred[1];
    ann_predict(sync_net, (real[]){1, 1}, sync_pred);
    TESTEX("After GPU train + sync, CPU predict works", (sync_pred[0] < 0.5f));

    tensor_free(sync_inputs);
    tensor_free(sync_targets);
    ann_free_network(sync_net);

    // ========================================================================
    SUITE("GEMM Tiles: Shapes That Are Not Multiples of 64");
    // ========================================================================
    {
        // 257 rows span five 64-row tiles, the last one nearly empty; 63, 65 and 31
        // straddle the 64-wide and 16-deep tile edges
        int nodes[] = { 63, 65, 130, 31 };
        PNetwork net = make_dense_net(5, OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY, nodes, 4,
                                      ACTIVATION_TANH, ACTIVATION_SOFTMAX);
        srand(6);
        PTensor x = random_data(257, 63, 0);
        TESTEX("Odd-shaped network uploads", ann_gpu_upload_network(net) == ERR_OK);
        TESTEX("Batch of 257 through 63-65-130-31 matches the CPU", max_gpu_cpu_diff(net, x->values, 257) < TOLERANCE);
        TESTEX("A single row matches the CPU", max_gpu_cpu_diff(net, x->values, 1) < TOLERANCE);
        tensor_free(x);
        ann_free_network(net);
    }

    // ========================================================================
    SUITE("GPU Training Matches CPU Training: Every Optimizer");
    // ========================================================================
    {
        int nodes[] = { 37, 70, 5 };
        const struct { Optimizer_type opt; const char *name; } opts[] = {
            { OPT_SGD, "SGD" }, { OPT_MOMENTUM, "Momentum" }, { OPT_ADAGRAD, "AdaGrad" },
            { OPT_RMSPROP, "RMSProp" }, { OPT_ADAM, "Adam" },
        };
        for (int i = 0; i < (int)(sizeof(opts) / sizeof(opts[0])); i++)
        {
            real cpu_loss, gpu_loss;
            real diff = train_cpu_and_gpu(opts[i].opt, LOSS_MSE, nodes, 3, ACTIVATION_RELU, ACTIVATION_SIGMOID,
                                          0.0f, 0.0f, 0.0f, &cpu_loss, &gpu_loss);
            printf("\t%s: max weight difference %g, loss %g on the GPU and %g on the CPU\n",
                   opts[i].name, (double)diff, (double)gpu_loss, (double)cpu_loss);
            TESTEX("weights match the CPU", diff < 1e-3f);
            TESTEX("loss matches the CPU", fabsf(gpu_loss - cpu_loss) < 1e-4f + 1e-3f * fabsf(cpu_loss));
        }
    }

    // ========================================================================
    SUITE("GPU Training Matches CPU Training: Losses, Activations, Regularization");
    // ========================================================================
    {
        const struct {
            const char *name;
            Optimizer_type opt; Loss_type loss; Activation_type hidden, output;
            real l2, l1, clip;
        } cases[] = {
            { "Categorical cross-entropy, softmax", OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY,
              ACTIVATION_LEAKY_RELU, ACTIVATION_SOFTMAX, 0.0f, 0.0f, 0.0f },
            { "Binary cross-entropy, sigmoid", OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY,
              ACTIVATION_SOFTSIGN, ACTIVATION_SIGMOID, 0.0f, 0.0f, 0.0f },
            { "MSE through softmax", OPT_SGD, LOSS_MSE, ACTIVATION_TANH, ACTIVATION_SOFTMAX, 0.0f, 0.0f, 0.0f },
            { "MSE, linear output", OPT_MOMENTUM, LOSS_MSE, ACTIVATION_GELU, ACTIVATION_NULL, 0.0f, 0.0f, 0.0f },
            { "L2 + L1 + gradient clipping", OPT_ADAM, LOSS_MSE, ACTIVATION_SIGMOID, ACTIVATION_SIGMOID,
              1e-3f, 1e-4f, 0.05f },
        };
        int nodes[] = { 29, 66, 34, 7 };
        for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++)
        {
            real cpu_loss, gpu_loss;
            real diff = train_cpu_and_gpu(cases[i].opt, cases[i].loss, nodes, 4, cases[i].hidden, cases[i].output,
                                          cases[i].l2, cases[i].l1, cases[i].clip, &cpu_loss, &gpu_loss);
            printf("\t%s: max weight difference %g, loss %g on the GPU and %g on the CPU\n",
                   cases[i].name, (double)diff, (double)gpu_loss, (double)cpu_loss);
            TESTEX("weights match the CPU", diff < 1e-3f);
            TESTEX("loss matches the CPU", fabsf(gpu_loss - cpu_loss) < 1e-4f + 1e-3f * fabsf(cpu_loss));
        }
    }

    run_gpu_consistency_tests();
}

#else  /* USE_VULKAN not defined */

void test_main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    MODULE("Vulkan GPU Backend Tests");
    SUITE("Build Configuration");
    COMMENT("USE_VULKAN not defined - Vulkan GPU tests skipped");
    COMMENT("Build with: cmake -DUSE_VULKAN=1 .. to enable Vulkan GPU support");
}

#endif /* USE_VULKAN */
