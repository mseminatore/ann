//-----------------------------------------------------------
// Reading and setting layer parameters, and freezing layers.
//
// Counts and shapes for each layer kind, argument checks, copying a trained
// network's parameters into a new one (it then predicts the same), values
// set before training are kept as its starting point while the other layers
// are still randomized, and frozen layers stay unchanged through training
// with every optimizer, regularization, clipping and the online steps.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

#define SEQ   12
#define VOCAB 10
#define N     64

static void quiet(const char *s) { (void)s; }
static void quiet_errors(int code, const char *message, const char *function) { (void)code; (void)message; (void)function; }

// layers: input 1x6x6, conv 4x6x6, batch norm, max pool 4x3x3, flatten, dense 8, dense 2
static PNetwork make_cnn(Optimizer_type opt)
{
    PNetwork net = ann_make_network(opt, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_input(net, 1, 6, 6);
    ann_add_conv2d(net, 4, 3, 1, 1, ACTIVATION_NULL);
    ann_add_batchnorm(net, ACTIVATION_RELU);
    ann_add_maxpool2d(net, 2, 2);
    ann_add_flatten(net);
    ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 8);
    ann_set_epoch_limit(net, 3);
    return net;
}

// layers: input of SEQ ids, embedding 4x1xSEQ, conv1d 6x1xSEQ, global max pool 6, dense 2
static PNetwork make_text(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_layer(net, SEQ, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_embedding_masked(net, VOCAB, 4, 0);
    ann_add_conv1d(net, 6, 3, 1, 1, ACTIVATION_RELU);
    ann_add_global_maxpool2d(net);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 8);
    ann_set_epoch_limit(net, 3);
    return net;
}

// images whose label is whether the top half is brighter
static void make_images(real *x, real *y, int n)
{
    for (int s = 0; s < n; s++) {
        real top = 0, bottom = 0;
        for (int i = 0; i < 36; i++) {
            real v = (real)((s * 37 + i * 11) % 17) / 17.0f;
            x[s * 36 + i] = v;
            if (i < 18) top += v; else bottom += v;
        }
        y[s * 2] = top > bottom;
        y[s * 2 + 1] = top <= bottom;
    }
}

// texts of ids padded with 0 at the end, labelled by whether id 3 occurs
static void make_texts(real *x, real *y, int n)
{
    for (int s = 0; s < n; s++) {
        int len = 4 + s % 8, has = 0;
        for (int t = 0; t < SEQ; t++) {
            int id = t < len ? 1 + (s * 7 + t * 5) % (VOCAB - 1) : 0;
            x[s * SEQ + t] = (real)id;
            has |= id == 3;
        }
        y[s * 2] = has;
        y[s * 2 + 1] = !has;
    }
}

// all of a network's parameters, layer by layer, in a new buffer
static real *all_params(PNetwork net, int *count)
{
    int n = 0, r, c;
    for (int l = 1; l < ann_get_layer_count(net); l++)
        for (int p = 0; p < ann_get_layer_param_count(net, l); p++) {
            ann_get_layer_param_shape(net, l, p, &r, &c);
            n += r * c;
        }
    real *buf = malloc(n * sizeof(real)), *dst = buf;
    for (int l = 1; buf && l < ann_get_layer_count(net); l++)
        for (int p = 0; p < ann_get_layer_param_count(net, l); p++) {
            ann_get_layer_param_shape(net, l, p, &r, &c);
            ann_get_layer_param(net, l, p, dst, r * c);
            dst += r * c;
        }
    *count = n;
    return buf;
}

// one layer's parameters, in a new buffer
static real *layer_params(PNetwork net, int layer, int *count)
{
    int n = 0, r, c;
    for (int p = 0; p < ann_get_layer_param_count(net, layer); p++) {
        ann_get_layer_param_shape(net, layer, p, &r, &c);
        n += r * c;
    }
    real *buf = malloc(n * sizeof(real) + 1), *dst = buf;
    for (int p = 0; buf && p < ann_get_layer_param_count(net, layer); p++) {
        ann_get_layer_param_shape(net, layer, p, &r, &c);
        ann_get_layer_param(net, layer, p, dst, r * c);
        dst += r * c;
    }
    *count = n;
    return buf;
}

static int same_values(const real *a, const real *b, int n)
{
    return a && b && memcmp(a, b, n * sizeof(real)) == 0;
}

// copy every parameter of src into dst (same layers)
static int copy_all(PNetwork src, PNetwork dst)
{
    int r, c, ok = 1;
    for (int l = 1; l < ann_get_layer_count(src); l++)
        for (int p = 0; p < ann_get_layer_param_count(src, l); p++) {
            ann_get_layer_param_shape(src, l, p, &r, &c);
            real *v = malloc(r * c * sizeof(real));
            ok &= v && ann_get_layer_param(src, l, p, v, r * c) == ERR_OK &&
                  ann_set_layer_param(dst, l, p, v, r * c) == ERR_OK;
            free(v);
        }
    return ok;
}

static real max_pred_diff(PNetwork a, PNetwork b, const real *x, int n, int in_size)
{
    real ya[2], yb[2], worst = 0;
    for (int s = 0; s < n; s++) {
        ann_predict(a, x + s * in_size, ya);
        ann_predict(b, x + s * in_size, yb);
        for (int k = 0; k < 2; k++)
            if (fabsf(ya[k] - yb[k]) > worst) worst = fabsf(ya[k] - yb[k]);
    }
    return worst;
}

// train a CNN with layer frozen; 1 if the frozen layer's parameters are
// unchanged and every other layer's changed
static int frozen_stays(Optimizer_type opt, int frozen, real decay, real l1, real clip, int online)
{
    static real x[N * 36], y[N * 2];
    make_images(x, y, N);
    srand(5);
    PNetwork net = make_cnn(opt);
    ann_set_weight_decay(net, decay);
    ann_set_l1_regularization(net, l1);
    ann_set_gradient_clip(net, clip);
    int ok = ann_set_layer_trainable(net, frozen, 0) == ERR_OK && !ann_get_layer_trainable(net, frozen);

    // start from known values (training would otherwise initialize them first)
    real *before[8] = { 0 };
    int counts[8];
    ann_train_begin(net);
    for (int l = 1; l < ann_get_layer_count(net); l++)
        before[l] = layer_params(net, l, &counts[l]);
    if (!online)
        ann_train_end(net);

    if (online) {
        for (int s = 0; s + 8 <= N; s += 8)
            ann_train_step(net, x + s * 36, y + s * 2, 8);
        ann_train_end(net);
    } else {
        PTensor tx = tensor_create_from_array(N, 36, x), ty = tensor_create_from_array(N, 2, y);
        ann_train_network(net, tx, ty, N);
        tensor_free(tx);
        tensor_free(ty);
    }

    for (int l = 1; l < ann_get_layer_count(net); l++) {
        int n;
        real *after = layer_params(net, l, &n);
        if (n > 0) {
            // batch norm's running mean and variance always move: compare gamma and beta
            int compared = ann_get_layer_kind(net, l) == LAYER_KIND_BATCHNORM ? 2 * 4 : n;
            int same = same_values(before[l], after, compared);
            ok &= l == frozen ? same : !same;
        }
        free(after);
        free(before[l]);
    }

    // a frozen batch norm still keeps its running statistics
    if (ann_get_layer_kind(net, frozen) == LAYER_KIND_BATCHNORM) {
        real var[4];
        ann_get_layer_param(net, frozen, 3, var, 4);
        ok &= var[0] != 1.0f;
    }
    ann_free_network(net);
    return ok;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Layer Parameters");
    ann_set_error_log_callback(quiet_errors);	// the argument checks report errors

    SUITE("Counts and Shapes");
    {
        PNetwork cnn = make_cnn(OPT_ADAM), text = make_text();
        int r = -1, c = -1;
        TESTEX("The input, pooling and flatten layers have none", ann_get_layer_param_count(cnn, 0) == 0 &&
               ann_get_layer_param_count(cnn, 3) == 0 && ann_get_layer_param_count(cnn, 4) == 0 &&
               ann_get_layer_param_count(text, 3) == 0);
        TESTEX("Dense and conv have weights and a bias, batch norm 4, embedding 1",
               ann_get_layer_param_count(cnn, 1) == 2 && ann_get_layer_param_count(cnn, 2) == 4 &&
               ann_get_layer_param_count(cnn, 5) == 2 && ann_get_layer_param_count(text, 1) == 1 &&
               ann_get_layer_param_count(text, 2) == 2);
        TESTEX("A layer out of range gives -1", ann_get_layer_param_count(cnn, 7) == -1 &&
               ann_get_layer_param_count(cnn, -1) == -1 && ann_get_layer_param_count(NULL, 1) == -1);
        TESTEX("Conv weights are [filters x channels * k * k]",
               ann_get_layer_param_shape(cnn, 1, 0, &r, &c) == ERR_OK && r == 4 && c == 9);
        TESTEX("and the bias [1 x filters]", ann_get_layer_param_shape(cnn, 1, 1, &r, &c) == ERR_OK && r == 1 && c == 4);
        TESTEX("Batch norm's running variance is [1 x channels]",
               ann_get_layer_param_shape(cnn, 2, 3, &r, &c) == ERR_OK && r == 1 && c == 4);
        TESTEX("Dense weights are [nodes x nodes below]",
               ann_get_layer_param_shape(cnn, 5, 0, &r, &c) == ERR_OK && r == 8 && c == 36);
        TESTEX("Embedding vectors are [vocab x dim]",
               ann_get_layer_param_shape(text, 1, 0, &r, &c) == ERR_OK && r == VOCAB && c == 4);
        TESTEX("A 1-D conv's weights are [filters x channels * k]",
               ann_get_layer_param_shape(text, 2, 0, &r, &c) == ERR_OK && r == 6 && c == 12);
        TESTEX("A bad index or layer is ERR_INVALID", ann_get_layer_param_shape(cnn, 1, 2, &r, &c) == ERR_INVALID &&
               ann_get_layer_param_shape(cnn, 3, 0, &r, &c) == ERR_INVALID &&
               ann_get_layer_param_shape(cnn, 9, 0, &r, &c) == ERR_INVALID);
        TESTEX("and NULLs ERR_NULL_PTR", ann_get_layer_param_shape(NULL, 1, 0, &r, &c) == ERR_NULL_PTR &&
               ann_get_layer_param_shape(cnn, 1, 0, NULL, &c) == ERR_NULL_PTR);
        ann_free_network(cnn);
        ann_free_network(text);
    }

    SUITE("Reading and Setting");
    {
        PNetwork net = make_cnn(OPT_ADAM);
        real w[36] = { 0 }, back[36], bad[36];
        for (int i = 0; i < 36; i++)
            w[i] = 0.1f * (real)(i - 18);
        TESTEX("Setting conv weights succeeds", ann_set_layer_param(net, 1, 0, w, 36) == ERR_OK);
        TESTEX("and reads back the same", ann_get_layer_param(net, 1, 0, back, 36) == ERR_OK &&
               same_values(w, back, 36));
        TESTEX("A wrong count is ERR_INVALID", ann_set_layer_param(net, 1, 0, w, 35) == ERR_INVALID &&
               ann_get_layer_param(net, 1, 0, back, 37) == ERR_INVALID);
        TESTEX("and so is a bad index or a layer without parameters",
               ann_set_layer_param(net, 1, 2, w, 4) == ERR_INVALID && ann_set_layer_param(net, 0, 0, w, 1) == ERR_INVALID &&
               ann_set_layer_param(net, 3, 0, w, 1) == ERR_INVALID && ann_get_layer_param(net, 8, 0, w, 1) == ERR_INVALID);
        memcpy(bad, w, sizeof(w));
        bad[7] = (real)NAN;
        int nan_rejected = ann_set_layer_param(net, 1, 0, bad, 36) == ERR_INVALID;
        bad[7] = (real)INFINITY;
        TESTEX("A value that is not finite is rejected", nan_rejected &&
               ann_set_layer_param(net, 1, 0, bad, 36) == ERR_INVALID);
        TESTEX("and changes nothing", ann_get_layer_param(net, 1, 0, back, 36) == ERR_OK && same_values(w, back, 36));
        TESTEX("NULLs are ERR_NULL_PTR", ann_set_layer_param(NULL, 1, 0, w, 36) == ERR_NULL_PTR &&
               ann_set_layer_param(net, 1, 0, NULL, 36) == ERR_NULL_PTR &&
               ann_get_layer_param(net, 1, 0, NULL, 36) == ERR_NULL_PTR);
        ann_free_network(net);
    }

    SUITE("Before Training");
    {
        static real x[N * 36], y[N * 2];
        make_images(x, y, N);
        PTensor tx = tensor_create_from_array(N, 36, x), ty = tensor_create_from_array(N, 2, y);
        srand(9);
        PNetwork net = make_cnn(OPT_ADAM);
        int n1, n2, nonzero = 0;
        real *start = layer_params(net, 5, &n1);
        for (int i = 0; i < n1; i++) nonzero |= start[i] != 0.0f;
        TESTEX("Reading an untrained layer gives initialized (random) weights", nonzero);
        ann_set_learning_rate(net, 0.0f);
        ann_train_network(net, tx, ty, N);
        real *after = layer_params(net, 5, &n2);
        TESTEX("which are the values training starts from", n1 == n2 && same_values(start, after, n1));
        free(start);
        free(after);
        ann_free_network(net);
        tensor_free(tx);
        tensor_free(ty);
    }

    SUITE("Copying a Trained Network");
    {
        static real x[N * 36], y[N * 2], xs[N * SEQ], ys[N * 2];
        make_images(x, y, N);
        make_texts(xs, ys, N);
        PTensor tx = tensor_create_from_array(N, 36, x), ty = tensor_create_from_array(N, 2, y);
        PTensor txs = tensor_create_from_array(N, SEQ, xs), tys = tensor_create_from_array(N, 2, ys);

        srand(2);
        PNetwork trained = make_cnn(OPT_ADAM);
        ann_train_network(trained, tx, ty, N);
        PNetwork copy = make_cnn(OPT_ADAM);
        TESTEX("Every parameter of a trained CNN copies into a new one", copy_all(trained, copy));
        TESTEX("which predicts identically (batch norm's running statistics included)",
               max_pred_diff(trained, copy, x, N, 36) == 0.0f);

        // training the copy starts from the copied values: at learning rate 0 nothing moves
        int n1, n2;
        real *before = all_params(copy, &n1);
        ann_set_learning_rate(copy, 0.0f);
        ann_train_network(copy, tx, ty, N);
        real *after = all_params(copy, &n2);
        int same = n1 == n2;
        for (int l = 1, off = 0; same && l < ann_get_layer_count(copy); l++)
            for (int p = 0; p < ann_get_layer_param_count(copy, l); p++) {
                int r, c;
                ann_get_layer_param_shape(copy, l, p, &r, &c);
                // batch norm's running mean and variance still move
                if (!(ann_get_layer_kind(copy, l) == LAYER_KIND_BATCHNORM && p >= 2))
                    same &= same_values(before + off, after + off, r * c);
                off += r * c;
            }
        TESTEX("and training does not randomize them again", same);
        free(before);
        free(after);
        ann_free_network(trained);
        ann_free_network(copy);

        srand(2);
        trained = make_text();
        ann_train_network(trained, txs, tys, N);
        copy = make_text();
        TESTEX("A masked text net copies too", copy_all(trained, copy));
        TESTEX("and predicts identically", max_pred_diff(trained, copy, xs, N, SEQ) == 0.0f);

        // pre-trained vectors only: the other layers still start random
        PNetwork pre = make_text();
        int nv;
        real *vectors = layer_params(trained, 1, &nv);
        TESTEX("Setting only the embedding's vectors succeeds", ann_set_layer_param(pre, 1, 0, vectors, nv) == ERR_OK);
        ann_set_learning_rate(pre, 0.0f);
        ann_train_network(pre, txs, tys, N);
        int nc, nd;
        real *got = layer_params(pre, 1, &nv), *conv = layer_params(pre, 2, &nc), *dense = layer_params(pre, 4, &nd);
        real *vtrained = layer_params(trained, 1, &nv);
        int conv_set = 0, dense_set = 0;
        for (int i = 0; i < nc; i++) conv_set |= conv[i] != 0.0f;
        for (int i = 0; i < nd; i++) dense_set |= dense[i] != 0.0f;
        TESTEX("training keeps them", same_values(vtrained, got, nv));
        TESTEX("and initializes the layers above", conv_set && dense_set);
        free(vectors);
        free(vtrained);
        free(got);
        free(conv);
        free(dense);
        ann_free_network(pre);

        TESTEX("A saved network keeps the set values", ann_save_network_binary(copy, "test_params.nnb") == ERR_OK);
        PNetwork loaded = ann_load_network_binary("test_params.nnb");
        TESTEX("and loads predicting identically", loaded && max_pred_diff(trained, loaded, xs, N, SEQ) == 0.0f);
        ann_free_network(loaded);
        remove("test_params.nnb");
        ann_free_network(trained);
        ann_free_network(copy);

        tensor_free(tx);
        tensor_free(ty);
        tensor_free(txs);
        tensor_free(tys);
    }

    SUITE("Frozen Layers");
    {
        PNetwork net = make_cnn(OPT_ADAM);
        TESTEX("Layers train unless frozen", ann_get_layer_trainable(net, 1) && ann_get_layer_trainable(net, 5));
        TESTEX("The input layer and layers out of range can't be frozen",
               ann_set_layer_trainable(net, 0, 0) == ERR_INVALID && ann_set_layer_trainable(net, 7, 0) == ERR_INVALID &&
               ann_set_layer_trainable(NULL, 1, 0) == ERR_NULL_PTR);
        TESTEX("Freezing and unfreezing reads back", ann_set_layer_trainable(net, 5, 0) == ERR_OK &&
               !ann_get_layer_trainable(net, 5) && ann_set_layer_trainable(net, 5, 1) == ERR_OK &&
               ann_get_layer_trainable(net, 5));
        ann_free_network(net);

        TESTEX("A frozen dense layer stays put with SGD", frozen_stays(OPT_SGD, 5, 0, 0, 0, 0));
        TESTEX("with Momentum", frozen_stays(OPT_MOMENTUM, 5, 0, 0, 0, 0));
        TESTEX("with AdaGrad", frozen_stays(OPT_ADAGRAD, 5, 0, 0, 0, 0));
        TESTEX("with RMSProp", frozen_stays(OPT_RMSPROP, 5, 0, 0, 0, 0));
        TESTEX("with Adam", frozen_stays(OPT_ADAM, 5, 0, 0, 0, 0));
        TESTEX("with weight decay, L1 and clipping", frozen_stays(OPT_ADAM, 5, 1e-3f, 1e-3f, 0.5f, 0));
        TESTEX("A frozen conv layer stays put", frozen_stays(OPT_ADAM, 1, 1e-3f, 0, 0, 0));
        TESTEX("and in online training", frozen_stays(OPT_ADAM, 1, 1e-3f, 0, 0, 1));
        TESTEX("A frozen batch norm keeps gamma and beta, and updates its statistics",
               frozen_stays(OPT_ADAM, 2, 0, 0, 0, 0));

        // a frozen embedding with pre-trained vectors: the classifier above still learns
        static real xs[N * SEQ], ys[N * 2];
        make_texts(xs, ys, N);
        PTensor txs = tensor_create_from_array(N, SEQ, xs), tys = tensor_create_from_array(N, 2, ys);
        srand(4);
        PNetwork text = make_text();
        real vec[VOCAB * 4];
        for (int i = 0; i < VOCAB * 4; i++)
            vec[i] = (real)sin(1.3 * i);
        ann_set_layer_param(text, 1, 0, vec, VOCAB * 4);
        ann_set_layer_trainable(text, 1, 0);
        ann_set_epoch_limit(text, 30);
        real loss_before = ann_evaluate_loss(text, txs, tys);
        ann_train_network(text, txs, tys, N);
        real got[VOCAB * 4];
        ann_get_layer_param(text, 1, 0, got, VOCAB * 4);
        TESTEX("A frozen embedding keeps its vectors", same_values(vec, got, VOCAB * 4));
        TESTEX("while the layers above it learn", ann_evaluate_loss(text, txs, tys) < 0.5f * loss_before);
        ann_free_network(text);
        tensor_free(txs);
        tensor_free(tys);
    }
}
