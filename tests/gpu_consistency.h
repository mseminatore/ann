//-----------------------------------------------------------
// Backend-independent GPU consistency tests.
//
// Checks that the CPU and GPU copies of a network's weights stay in step
// across ann_train_network(), dropout, and the online training API.
// Included by each GPU backend's test file (test_metal.c, test_cuda.c);
// call run_gpu_consistency_tests() after ann_gpu_init() succeeds.
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
// pass is forced by hiding layer 0's GPU buffer, which is what dispatch checks.
static int gc_gpu_matches_cpu(PNetwork net)
{
    real gpu[4], gpu_batch[4], cpu[4];

    for (int i = 0; i < 4; i++)
        ann_predict(net, &gc_xor_in[i * 2], &gpu[i]);
    ann_predict_batch(net, gc_xor_in, gpu_batch, 4);

    void *saved = net->layers[0].t_weights->gpu_buf;
    net->layers[0].t_weights->gpu_buf = NULL;
    for (int i = 0; i < 4; i++)
        ann_predict(net, &gc_xor_in[i * 2], &cpu[i]);
    net->layers[0].t_weights->gpu_buf = saved;

    for (int i = 0; i < 4; i++) {
        real d1 = gpu[i] - cpu[i], d2 = gpu_batch[i] - cpu[i];
        if (d1 < 0) d1 = -d1;
        if (d2 < 0) d2 = -d2;
        if (d1 > GPU_CONSISTENCY_TOL || d2 > GPU_CONSISTENCY_TOL)
            return 0;
    }
    return 1;
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
    SUITE("GPU/CPU Weight Consistency - Dropout");
    // ========================================================================

    net = gc_make_net(0.5f);
    ann_gpu_upload_network(net);
    gc_train(net);
    TESTEX("Dropout network trains on the CPU path (dropout mask allocated)",
           (net->layers[1].t_dropout_mask != NULL));
    TESTEX("After CPU dropout training, the GPU copy matches the CPU weights",
           gc_gpu_matches_cpu(net));
    ann_free_network(net);

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

    int n_w = net->layers[0].t_weights->rows * net->layers[0].t_weights->cols;
    real *before_sync = (real *)malloc((size_t)n_w * sizeof(real));
    memcpy(before_sync, net->layers[0].t_weights->values, (size_t)n_w * sizeof(real));
    ann_gpu_sync_weights(net);
    TESTEX("ann_gpu_sync_weights() does not overwrite newer CPU weights",
           (memcmp(before_sync, net->layers[0].t_weights->values, (size_t)n_w * sizeof(real)) == 0));
    free(before_sync);

    ann_train_end(net);
    TESTEX("After ann_train_end(), the GPU copy matches the CPU weights",
           gc_gpu_matches_cpu(net));
    TESTEX("After ann_train_end(), weights are still on the GPU",
           (net->layers[0].t_weights->gpu_buf != NULL));
    ann_free_network(net);
}

#endif /* GPU_CONSISTENCY_H */
