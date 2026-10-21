//-----------------------------------------------------------
// Backend-independent GPU consistency tests.
//
// Checks that the CPU and GPU copies of a network's weights stay in step
// across ann_train_network(), dropout, and the online training API.
// Included by each GPU backend's test file (test_metal.c, test_cuda.c);
// call run_gpu_consistency_tests() after ann_gpu_init() succeeds. A backend
// that runs CNNs defines GC_GPU_SUPPORTS_CNN before including this file.
//-----------------------------------------------------------

#ifndef GPU_CONSISTENCY_H
#define GPU_CONSISTENCY_H

#define GPU_CONSISTENCY_TOL 1e-4f

static real gc_xor_in[]  = {0,0, 0,1, 1,0, 1,1};
static real gc_xor_out[] = {0,   1,   1,   0  };

static PNetwork gc_make_net(real dropout)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(net, 0.05f);
    ann_set_convergence(net, 1e-6f);  // run every epoch
    ann_set_epoch_limit(net, 50);
    if (dropout > 0.0f)
        ann_set_dropout(net, dropout);
    return net;
}

static void gc_train(PNetwork net)
{
    PTensor tin  = tensor_create_from_array(4, 2, gc_xor_in);
    PTensor tout = tensor_create_from_array(4, 1, gc_xor_out);
    ann_train_network(net, tin, tout, 4);
    tensor_free(tin);
    tensor_free(tout);
}

// Compare what ann_predict()/ann_predict_batch() return now (GPU when the
// library chooses it) with a CPU forward pass over the CPU weights. The CPU
// pass is forced by hiding layer 1's weight buffer, which is what dispatch checks.
static int gc_gpu_matches_cpu(PNetwork net)
{
    real gpu[4], gpu_batch[4], cpu[4];

    for (int i = 0; i < 4; i++)
        ann_predict(net, &gc_xor_in[i * 2], &gpu[i]);
    ann_predict_batch(net, gc_xor_in, gpu_batch, 4);

    void *saved = net->layers[1].t_weights->gpu_buf;
    net->layers[1].t_weights->gpu_buf = NULL;
    for (int i = 0; i < 4; i++)
        ann_predict(net, &gc_xor_in[i * 2], &cpu[i]);
    net->layers[1].t_weights->gpu_buf = saved;

    for (int i = 0; i < 4; i++) {
        real d1 = gpu[i] - cpu[i], d2 = gpu_batch[i] - cpu[i];
        if (d1 < 0) d1 = -d1;
        if (d2 < 0) d2 = -d2;
        if (d1 > GPU_CONSISTENCY_TOL || d2 > GPU_CONSISTENCY_TOL)
            return 0;
    }
    return 1;
}

// Network with dropout on two hidden layers: sigmoid (derivative uses the
// activation) and ReLU (derivative uses the pre-activation)
static PNetwork gc_make_dropout_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 6, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(net, 0.05f);
    ann_set_convergence(net, 1e-6f);  // run every epoch
    ann_set_epoch_limit(net, 20);
    ann_set_dropout(net, 0.3f);
    return net;
}

// GELU hidden layers (derivative uses the pre-activation)
static PNetwork gc_make_gelu_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    if (!net) return NULL;

    ann_add_layer(net, 2, LAYER_INPUT,  ACTIVATION_NULL);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_GELU);
    ann_add_layer(net, 6, LAYER_HIDDEN, ACTIVATION_GELU);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(net, 0.05f);
    ann_set_convergence(net, 1e-6f);  // run every epoch
    ann_set_epoch_limit(net, 30);
    return net;
}

static int gc_tensors_close(PTensor a, PTensor b, real tol)
{
    if (!a || !b) return a == b;
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) {
        real d = a->values[i] - b->values[i];
        if (d < 0) d = -d;
        if (d > tol) return 0;
    }
    return 1;
}

// Compare weights and biases, plus Adam state when with_state is set
static int gc_nets_close(PNetwork a, PNetwork b, real tol, int with_state)
{
    for (int l = 0; l < a->layer_count - 1; l++) {
        PLayer la = &a->layers[l + 1], lb = &b->layers[l + 1];
        if (!gc_tensors_close(la->t_weights, lb->t_weights, tol) ||
            !gc_tensors_close(la->t_bias, lb->t_bias, tol))
            return 0;
        if (with_state &&
            (!gc_tensors_close(la->t_m, lb->t_m, tol) || !gc_tensors_close(la->t_v, lb->t_v, tol) ||
             !gc_tensors_close(la->t_bias_m, lb->t_bias_m, tol) ||
             !gc_tensors_close(la->t_bias_v, lb->t_bias_v, tol)))
            return 0;
    }
    return 1;
}

static void gc_online_steps(PNetwork net)
{
    srand(99);
    ann_train_begin(net);
    for (int step = 0; step < 20; step++)
        ann_train_step(net, &gc_xor_in[(step % 4) * 2], &gc_xor_out[step % 4], 1);
    ann_train_end(net);
}

static void run_gpu_consistency_tests(void)
{
    // ========================================================================
    SUITE("GPU/CPU Weight Consistency - ann_train_network");
    // ========================================================================

    PNetwork net = gc_make_net(0.0f);
    ann_gpu_upload_network(net);
    gc_train(net);
    TESTEX("After GPU training, CPU weights match the GPU without an explicit sync",
           gc_gpu_matches_cpu(net));
    ann_free_network(net);

    // ========================================================================
    SUITE("GPU Dropout and Optimizer State");
    // ========================================================================

    // Same seed, same network: one trained on the CPU, one on the GPU
    PNetwork cpu_net = gc_make_dropout_net();
    PNetwork gpu_net = gc_make_dropout_net();
    ann_gpu_upload_network(gpu_net);

    srand(1234);
    gc_train(cpu_net);
    srand(1234);
    gc_train(gpu_net);

    TESTEX("Dropout network trains on the GPU (mask uploaded)",
           (gpu_net->layers[1].t_dropout_mask && gpu_net->layers[1].t_dropout_mask->gpu_buf));
    TESTEX("GPU dropout training matches CPU dropout training",
           gc_nets_close(cpu_net, gpu_net, 1e-3f, 0));

    int moments_nonzero = 0;
    for (int i = 0; i < gpu_net->layers[1].t_m->rows * gpu_net->layers[1].t_m->cols; i++)
        if (gpu_net->layers[1].t_m->values[i] != 0.0f) moments_nonzero = 1;
    TESTEX("Adam moments come back from the GPU after training", moments_nonzero);
    TESTEX("GPU Adam moments match CPU training", gc_nets_close(cpu_net, gpu_net, 1e-3f, 1));
    TESTEX("After GPU dropout training, the GPU copy matches the CPU weights",
           gc_gpu_matches_cpu(gpu_net));

    // Online fine-tuning continues from the GPU's optimizer state
    gc_online_steps(cpu_net);
    gc_online_steps(gpu_net);
    TESTEX("Online training after GPU training matches the CPU-only network",
           gc_nets_close(cpu_net, gpu_net, 1e-3f, 1));

    // A second ann_train_network() starts from the CPU's (online-updated) state
    srand(4321);
    gc_train(cpu_net);
    srand(4321);
    gc_train(gpu_net);
    TESTEX("GPU training after online training matches the CPU-only network",
           gc_nets_close(cpu_net, gpu_net, 1e-3f, 1));

    ann_free_network(cpu_net);
    ann_free_network(gpu_net);

    // ========================================================================
    SUITE("GPU GELU and Early Stopping");
    // ========================================================================

    cpu_net = gc_make_gelu_net();
    gpu_net = gc_make_gelu_net();
    ann_gpu_upload_network(gpu_net);
    srand(77);
    gc_train(cpu_net);
    srand(77);
    gc_train(gpu_net);
    TESTEX("GPU GELU training matches CPU GELU training", gc_nets_close(cpu_net, gpu_net, 1e-3f, 0));
    TESTEX("After GPU GELU training, the GPU copy matches the CPU weights",
           gc_gpu_matches_cpu(gpu_net));
    ann_free_network(cpu_net);
    ann_free_network(gpu_net);

    // Validation on flipped targets gets worse as training fits XOR, so
    // early stopping triggers and restores an earlier epoch's weights
    {
        real flipped[] = {1, 0, 0, 1};
        PTensor vin  = tensor_create_from_array(4, 2, gc_xor_in);
        PTensor vout = tensor_create_from_array(4, 1, flipped);

        cpu_net = gc_make_net(0.0f);
        gpu_net = gc_make_net(0.0f);
        ann_set_epoch_limit(cpu_net, 300);
        ann_set_epoch_limit(gpu_net, 300);
        ann_set_validation_data(cpu_net, vin, vout);
        ann_set_validation_data(gpu_net, vin, vout);
        ann_set_early_stopping(cpu_net, 5, 0.0f, 1);
        ann_set_early_stopping(gpu_net, 5, 0.0f, 1);
        ann_gpu_upload_network(gpu_net);

        srand(5);
        gc_train(cpu_net);
        srand(5);
        gc_train(gpu_net);
        TESTEX("GPU early stopping picks the same best epoch as the CPU",
               ann_get_best_epoch(cpu_net) == ann_get_best_epoch(gpu_net) &&
               cpu_net->history_count == gpu_net->history_count);
        TESTEX("GPU restored weights match the CPU's", gc_nets_close(cpu_net, gpu_net, 1e-3f, 0));
        TESTEX("After restoring, the GPU copy matches the CPU weights", gc_gpu_matches_cpu(gpu_net));

        ann_free_network(cpu_net);
        ann_free_network(gpu_net);
        tensor_free(vin);
        tensor_free(vout);
    }

#ifndef GC_GPU_SUPPORTS_CNN
    // ========================================================================
    SUITE("Non-Dense Networks Stay on the CPU");
    // ========================================================================
    {
        PNetwork conv = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_input(conv, 1, 2, 2);
        ann_add_conv2d(conv, 2, 2, 1, 0, ACTIVATION_SIGMOID);   // -> 2x1x1
        ann_add_layer(conv, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_epoch_limit(conv, 5);
        TESTEX("Uploading a conv network returns ERR_INVALID", ann_gpu_upload_network(conv) == ERR_INVALID);
        TESTEX("and leaves its weights off the GPU", conv->layers[1].params[0].value->gpu_buf == NULL);

        // four 1x2x2 images: the XOR inputs in the top row, zeros below
        real images[16] = {0,0,0,0, 0,1,0,0, 1,0,0,0, 1,1,0,0};
        PTensor tin  = tensor_create_from_array(4, 4, images);
        PTensor tout = tensor_create_from_array(4, 1, gc_xor_out);
        real before = 0.0f, out = 0.0f;
        ann_predict(conv, &images[4], &before);
        ann_train_network(conv, tin, tout, 4);
        ann_predict(conv, &images[4], &out);
        TESTEX("The conv network still trains (on the CPU)", isfinite(out) && out != before);
        tensor_free(tin);
        tensor_free(tout);
        ann_free_network(conv);
    }
#endif

    // ========================================================================
    SUITE("GPU/CPU Weight Consistency - Online Training");
    // ========================================================================

    net = gc_make_net(0.0f);
    ann_gpu_upload_network(net);
    gc_train(net);

    ann_train_begin(net);
    for (int step = 0; step < 20; step++)
        ann_train_step(net, &gc_xor_in[(step % 4) * 2], &gc_xor_out[step % 4], 1);
    TESTEX("Mid-session predictions use the updated weights",
           gc_gpu_matches_cpu(net));

    int n_w = net->layers[1].t_weights->rows * net->layers[1].t_weights->cols;
    real *before_sync = (real *)malloc((size_t)n_w * sizeof(real));
    memcpy(before_sync, net->layers[1].t_weights->values, (size_t)n_w * sizeof(real));
    ann_gpu_sync_weights(net);
    TESTEX("ann_gpu_sync_weights() does not overwrite newer CPU weights",
           (memcmp(before_sync, net->layers[1].t_weights->values, (size_t)n_w * sizeof(real)) == 0));
    free(before_sync);

    ann_train_end(net);
    TESTEX("After ann_train_end(), the GPU copy matches the CPU weights",
           gc_gpu_matches_cpu(net));
    TESTEX("After ann_train_end(), weights are still on the GPU",
           (net->layers[1].t_weights->gpu_buf != NULL));
    ann_free_network(net);
}

#endif /* GPU_CONSISTENCY_H */
