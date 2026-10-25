//-----------------------------------------------------------
// Embedding and global max pooling layers, for text.
//
// Shapes and argument checks, lookups (rounding, padding and ids outside the
// vocabulary), finite-difference gradient checks as in test_conv.c, the same
// results for any thread count, a small text classifier, save/load, ONNX
// export and import, and quantization.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "testy/test.h"

#define GC_LR    0.01f
#define GC_EPS   1e-3f
#define GC_ABS   1e-3f
#define GC_REL   2e-2f
#define GC_BATCH 2
#define MAX_OUT  8

static void quiet(const char *s) { (void)s; }

static int file_contains(const char *path, const char *needle)
{
    static char buf[1 << 18];
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

// Every parameter set to fixed, varied values
static void set_params(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++)
                t->values[i] = 0.5f * (real)sin(1.7 * i + 0.9 * l + 2.3 * p + 0.4);
        }
}

// Ids for the gradient checks over a vocabulary of 6 with padding id 0:
// in-vocabulary ids, padding, an id past the vocabulary, and a negative one
static const real gc_ids[GC_BATCH * 7] = { 1, 4, 2, 0, 5, 3, 9,   2, 2, 5, -1, 1, 4, 0 };
static real g_t[GC_BATCH * MAX_OUT];

static double batch_loss(PNetwork net)
{
    int in = net->layers[0].node_count, out = net->layers[net->layer_count - 1].node_count;
    double loss = 0.0;
    real y[MAX_OUT];
    for (int b = 0; b < GC_BATCH; b++) {
        ann_predict(net, &gc_ids[b * in], y);
        for (int k = 0; k < out; k++) {
            real t = g_t[b * out + k];
            loss += net->loss_type == LOSS_CATEGORICAL_CROSS_ENTROPY ? -t * log(y[k]) : 0.5 * (t - y[k]) * (t - y[k]);
        }
    }
    return loss;
}

// Largest scaled error between analytic and numeric gradients; <= 1 when all
// are within tolerance. Frees net.
static real grad_check(PNetwork net)
{
    if (!net) return 1e9f;
    net->print_func = quiet;
    ann_set_learning_rate(net, GC_LR);
    ann_train_begin(net);
    set_params(net);
    int out = net->layers[net->layer_count - 1].node_count;
    for (int b = 0; b < GC_BATCH; b++)
        for (int k = 0; k < out; k++)
            g_t[b * out + k] = k == b % out ? 0.7f : 0.3f / (out - 1);

    static real numeric[8192], before[8192];
    int n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++, n++) {
                real saved = t->values[i];
                t->values[i] = saved + GC_EPS;
                double lp = batch_loss(net);
                t->values[i] = saved - GC_EPS;
                double lm = batch_loss(net);
                t->values[i] = saved;
                numeric[n] = (real)(-(lp - lm) / (2.0 * GC_EPS));
                before[n] = saved;
            }
        }

    ann_train_step(net, gc_ids, g_t, GC_BATCH);

    real worst = 0.0f;
    n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++, n++) {
                real analytic = (t->values[i] - before[n]) / GC_LR;
                real err = (real)fabs(analytic - numeric[n]) / (GC_ABS + GC_REL * (real)fabs(numeric[n]));
                if (err > worst) worst = err;
            }
        }
    ann_train_end(net);
    ann_free_network(net);
    return worst;
}

// 7 ids -> embedding (vocab 6, dim 3, padding 0) -> conv1d(4, k3, padding 1, tanh)
// -> global max pool -> dense(3, softmax), cross-entropy
static PNetwork net_embed_conv(int pool_max)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 7, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_embedding(net, 6, 3, 0);
    ann_add_conv1d(net, 4, 3, 1, 1, ACTIVATION_TANH);
    if (pool_max)
        ann_add_global_maxpool2d(net);
    else
        ann_add_global_avgpool2d(net);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    return net;
}

// 7 ids -> embedding (vocab 6, dim 2, no padding) -> dense(2, sigmoid) -> dense(3, softmax)
static PNetwork net_embed_dense(void)
{
    PNetwork net = ann_make_network(OPT_SGD, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 7, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_embedding(net, 6, 2, -1);
    ann_add_layer(net, 2, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    return net;
}

//------------------------------
// A tiny text task: sequences of 1 to 16 tokens from a vocabulary of 40
// (0 pads them to 16), class 1 when the phrase "7 11" appears anywhere,
// else class 0 (whose sequences often hold 7 or 11 alone)
//------------------------------
enum { VOCAB = 40, SEQ = 16 };

static void make_texts(real *x, real *y, int count, unsigned seed)
{
    for (int n = 0; n < count; n++) {
        unsigned r = seed + 977u * (unsigned)n;
        int len = 4 + (int)((r >> 3) % (SEQ - 3)), cls = n % 2;
        for (int t = 0; t < SEQ; t++) {
            r = r * 1103515245u + 12345u;
            int tok = 1 + (int)((r >> 16) % (VOCAB - 1));
            x[n * SEQ + t] = (real)(t < len ? tok : 0);
        }
        // remove accidental phrases, then plant one (class 1) or decoys (class 0)
        for (int t = 0; t + 1 < len; t++)
            if (x[n * SEQ + t] == 7 && x[n * SEQ + t + 1] == 11)
                x[n * SEQ + t + 1] = 12;
        int at = (int)((r >> 8) % (unsigned)(len - 1));
        if (cls) {
            x[n * SEQ + at] = 7;
            x[n * SEQ + at + 1] = 11;
        } else {
            x[n * SEQ + at] = 11;
            x[n * SEQ + at + 1] = 7;
        }
        y[n * 2 + 0] = (real)!cls;
        y[n * 2 + 1] = (real)cls;
    }
}

static PNetwork make_text_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    ann_add_layer(net, SEQ, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_embedding(net, VOCAB, 8, 0);                  // 8x1x16
    ann_add_conv1d(net, 16, 2, 1, 0, ACTIVATION_RELU);    // 16x1x15: word pairs
    ann_add_global_maxpool2d(net);                        // 16: wherever the pair is
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_batch_size(net, 16);
    return net;
}

static int same_structure(PNetwork a, PNetwork b)
{
    if (!a || !b || a->layer_count != b->layer_count)
        return 0;
    for (int l = 0; l < a->layer_count; l++)
        if (a->layers[l].kind != b->layers[l].kind || a->layers[l].activation != b->layers[l].activation ||
            a->layers[l].shape.c != b->layers[l].shape.c || a->layers[l].shape.h != b->layers[l].shape.h ||
            a->layers[l].shape.w != b->layers[l].shape.w)
            return 0;
    return 1;
}

static real max_pred_diff(PNetwork a, PNetwork b, const real *x, int count)
{
    int in = a->layers[0].node_count, out = a->layers[a->layer_count - 1].node_count;
    real worst = 0.0f, ya[MAX_OUT], yb[MAX_OUT];
    for (int n = 0; n < count; n++) {
        ann_predict(a, &x[n * in], ya);
        ann_predict(b, &x[n * in], yb);
        for (int k = 0; k < out; k++)
            if (fabs(ya[k] - yb[k]) > worst) worst = (real)fabs(ya[k] - yb[k]);
    }
    return worst;
}

// The text net's parameters after a few epochs with the given thread count
static real *train_text_params(int threads, size_t *count)
{
    enum { N = 64 };
    static real x[N * SEQ], y[N * 2];
    make_texts(x, y, N, 5u);
    ann_set_num_threads(threads);
    srand(3);
    PNetwork net = make_text_net();
    ann_set_epoch_limit(net, 3);
    PTensor tx = tensor_create_from_array(N, SEQ, x), ty = tensor_create_from_array(N, 2, y);
    ann_train_network(net, tx, ty, N);
    size_t n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++)
            n += (size_t)net->layers[l].params[p].value->rows * net->layers[l].params[p].value->cols;
    real *out = (real *)malloc(n * sizeof(real));
    n = 0;
    for (int l = 1; out && l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            memcpy(out + n, t->values, (size_t)t->rows * t->cols * sizeof(real));
            n += (size_t)t->rows * t->cols;
        }
    *count = n;
    tensor_free(tx);
    tensor_free(ty);
    ann_free_network(net);
    return out;
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Embedding and Global Max Pooling Layers");

    SUITE("Shapes");
    {
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_layer(net, 8, LAYER_INPUT, ACTIVATION_NULL);
        TESTEX("Embedding 8 ids, vocab 10, dim 4: the sequence 4x1x8", ann_add_embedding(net, 10, 4, 0) == ERR_OK &&
               net->layers[1].shape.c == 4 && net->layers[1].shape.h == 1 && net->layers[1].shape.w == 8 &&
               ann_get_layer_kind(net, 1) == LAYER_KIND_EMBEDDING);
        TESTEX("with weights [vocab x dim], not regularized", net->layers[1].param_count == 1 &&
               net->layers[1].params[0].value->rows == 10 && net->layers[1].params[0].value->cols == 4 &&
               !net->layers[1].params[0].regularize);
        int k, st, p;
        TESTEX("ann_get_layer_settings reports the vocabulary and padding id",
               ann_get_layer_settings(net, 1, &k, &st, &p) == ERR_OK && k == 10 && st == 0 && p == 0);
        TESTEX("Global max pool: 4x1x8 -> 4x1x1", ann_add_global_maxpool2d(net) == ERR_OK &&
               net->layers[2].shape.c == 4 && net->layers[2].shape.h == 1 && net->layers[2].shape.w == 1 &&
               net->layers[2].param_count == 0 && ann_get_layer_kind(net, 2) == LAYER_KIND_GLOBAL_MAXPOOL2D);
        TESTEX("An embedding after a non-flat layer is rejected",
               ann_add_conv1d(net, 2, 1, 1, 0, ACTIVATION_NULL) == ERR_INVALID ||
               ann_add_embedding(net, 10, 4, -1) == ERR_OK);
        ann_free_network(net);

        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        TESTEX("Embedding as the first layer is rejected", ann_add_embedding(net, 10, 4, -1) == ERR_INVALID);
        TESTEX("Global max pool as the first layer is rejected", ann_add_global_maxpool2d(net) == ERR_INVALID);
        ann_add_input(net, 2, 3, 3);
        TESTEX("Embedding over an image is rejected", ann_add_embedding(net, 10, 4, -1) == ERR_INVALID);
        ann_free_network(net);

        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(net, 5, LAYER_INPUT, ACTIVATION_NULL);
        TESTEX("A zero vocabulary is rejected", ann_add_embedding(net, 0, 4, -1) == ERR_INVALID);
        TESTEX("A zero dimension is rejected", ann_add_embedding(net, 10, 0, -1) == ERR_INVALID);
        TESTEX("A padding id outside the vocabulary is rejected", ann_add_embedding(net, 10, 4, 10) == ERR_INVALID &&
               ann_add_embedding(net, 10, 4, -2) == ERR_INVALID);
        TESTEX("Rejected layers are not counted", net->layer_count == 1);
        ann_free_network(net);
    }

    SUITE("Lookups and Pooling");
    {
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_layer(net, 6, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_embedding(net, 5, 2, 0);
        ann_train_begin(net);
        ann_train_end(net);
        PTensor W = net->layers[1].params[0].value;
        TESTEX("The padding row starts at zero", W->values[0] == 0.0f && W->values[1] == 0.0f &&
               W->values[2] != 0.0f);
        for (int i = 0; i < 10; i++) W->values[i] = (real)(i + 1);
        const real ids[6] = { 1, 2.6f, 0, 4, 5, -1 };
        real y[12];
        ann_predict(net, ids, y);
        // channel d at step t: W[id][d]; 2.6 rounds to 3; 0 pads; 5 and -1 lie outside
        const real want[12] = { 3, 7, 0, 9, 0, 0,   4, 8, 0, 10, 0, 0 };
        TESTEX("Each step is its id's row (rounded), zero for padding and ids outside the vocabulary",
               memcmp(y, want, sizeof(want)) == 0);
        ann_free_network(net);

        net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 3, 2, 3);
        ann_add_global_maxpool2d(net);
        ann_train_begin(net);
        ann_train_end(net);
        const real x[18] = { 1, 5, 2, 5, 0, 3,   -4, -2, -9, -3, -2, -8,   0, 0, 0, 0, 0, 0 };
        real m[3];
        ann_predict(net, x, m);
        TESTEX("Global max pool takes each channel's largest value", m[0] == 5 && m[1] == -2 && m[2] == 0);
        ann_free_network(net);
    }

    SUITE("Gradients");
    TESTEX("Gradients: embedding -> conv1d -> global max pool -> softmax", grad_check(net_embed_conv(1)) <= 1.0f);
    TESTEX("Gradients: embedding -> conv1d -> global average pool -> softmax", grad_check(net_embed_conv(0)) <= 1.0f);
    TESTEX("Gradients: embedding (no padding) -> dense -> softmax", grad_check(net_embed_dense()) <= 1.0f);
    {
        // a training step leaves the padding row and the unused ids' rows alone
        PNetwork net = net_embed_conv(1);
        net->print_func = quiet;
        ann_train_begin(net);
        set_params(net);
        memset(net->layers[1].params[0].value->values, 0, 3 * sizeof(real));    // the padding row, as kept
        real before[18];
        memcpy(before, net->layers[1].params[0].value->values, sizeof(before));
        real t[6] = { 1, 0, 0, 0, 1, 0 };
        const real ids[14] = { 1, 4, 2, 0, 4, 2, 9,   2, 2, 4, -1, 1, 4, 0 };   // no 3 or 5
        ann_train_step(net, ids, t, 2);
        const real *after = net->layers[1].params[0].value->values;
        // (a used id's row moves only where max pooling took its positions, so some may not)
        TESTEX("A training step leaves the padding row zero and unused rows unchanged",
               after[0] == 0 && after[1] == 0 && after[2] == 0 && memcmp(after + 9, before + 9, 3 * sizeof(real)) == 0 &&
               memcmp(after + 15, before + 15, 3 * sizeof(real)) == 0 &&
               memcmp(after + 3, before + 3, 6 * sizeof(real)) != 0);
        ann_train_end(net);
        ann_free_network(net);
    }

    SUITE("Threads");
    {
        size_t n1, n4;
        real *one = train_text_params(1, &n1), *four = train_text_params(4, &n4);
        TESTEX("1 and 4 threads train identical weights", one && four && n1 == n4 &&
               memcmp(one, four, n1 * sizeof(real)) == 0);
        free(one);
        free(four);
        ann_set_num_threads(0);
    }

    SUITE("Training, Saving and ONNX");
    {
        enum { N = 256, NT = 128 };
        static real x[N * SEQ], y[N * 2], xt[NT * SEQ], yt[NT * 2];
        make_texts(x, y, N, 1u);
        make_texts(xt, yt, NT, 777u);
        PTensor tx = tensor_create_from_array(N, SEQ, x), ty = tensor_create_from_array(N, 2, y);
        PTensor txt = tensor_create_from_array(NT, SEQ, xt), tyt = tensor_create_from_array(NT, 2, yt);
        srand(1);
        PNetwork net = make_text_net();
        ann_set_epoch_limit(net, 40);
        ann_train_network(net, tx, ty, N);
        real acc = ann_evaluate_accuracy(net, txt, tyt);
        printf("\t  phrase detection with embedding + conv1d + max pool: test accuracy %.2f\n", acc);
        TESTEX("A text classifier learns to find a phrase anywhere in the sequence", acc >= 0.95f);

        TESTEX("Text and binary saves succeed", ann_save_network(net, "test_emb.nna") == ERR_OK &&
               ann_save_network_binary(net, "test_emb.nnb") == ERR_OK);
        PNetwork text = ann_load_network("test_emb.nna");
        PNetwork bin = ann_load_network_binary("test_emb.nnb");
        int k, st, p;
        TESTEX("Loaded networks keep the layers and the padding id", same_structure(net, text) &&
               same_structure(net, bin) && ann_get_layer_settings(bin, 1, &k, &st, &p) == ERR_OK && k == VOCAB && p == 0);
        TESTEX("Binary load predicts identically", bin && max_pred_diff(net, bin, xt, NT) == 0.0f);
        TESTEX("Text load predicts the same (6 decimals)", text && max_pred_diff(net, text, xt, NT) < 1e-3f);
        ann_free_network(text);
        ann_free_network(bin);
        remove("test_emb.nna");
        remove("test_emb.nnb");

        const char *onnx = "test_emb.onnx.json";
        TESTEX("It exports to ONNX", ann_export_onnx(net, onnx) == ERR_OK);
        TESTEX("with Cast, Gather, Transpose and Unsqueeze for the embedding and a GlobalMaxPool",
               file_contains(onnx, "\"op_type\": \"Cast\"") && file_contains(onnx, "\"op_type\": \"Gather\"") &&
               file_contains(onnx, "\"input\": [\"emb_0_weight\", \"emb_0_ids\"]") &&
               file_contains(onnx, "\"op_type\": \"Transpose\"") && file_contains(onnx, "\"op_type\": \"Unsqueeze\"") &&
               file_contains(onnx, "\"op_type\": \"GlobalMaxPool\"") && file_contains(onnx, "\"dims\": [40, 8]"));
        PNetwork back = ann_import_onnx(onnx);
        TESTEX("and imports with the same structure", same_structure(net, back));
        TESTEX("and predicts identically", back && max_pred_diff(net, back, xt, NT) == 0.0f);
        ann_free_network(back);

        PQuantizedNetwork q = ann_quantize(net, tx);
        TESTEX("A network with an embedding cannot be quantized yet", q == NULL &&
               ann_export_c_header(net, NULL, "test_emb.h", "emb") == ERR_INVALID);
        ann_free_quantized(q);
        remove("test_emb.h");
        tensor_free(tx);
        tensor_free(ty);
        tensor_free(txt);
        tensor_free(tyt);
        ann_free_network(net);

        // a PyTorch-style model (protobuf JSON): int64 ids [N, 5], Gather(emb.weight [6, 2]),
        // Transpose, Conv1d(3, k2), Relu, GlobalMaxPool, Flatten, Gemm
        static const char *model =
            "{\"irVersion\":\"8\",\"producerName\":\"pytorch\",\"graph\":{\"node\":["
            "{\"input\":[\"emb.weight\",\"x\"],\"output\":[\"e\"],\"name\":\"/emb/Gather\",\"opType\":\"Gather\"},"
            "{\"input\":[\"e\"],\"output\":[\"s\"],\"opType\":\"Transpose\",\"attribute\":[{\"name\":\"perm\",\"ints\":[\"0\",\"2\",\"1\"],\"type\":\"INTS\"}]},"
            "{\"input\":[\"s\",\"c.w\",\"c.b\"],\"output\":[\"c\"],\"opType\":\"Conv\",\"attribute\":[{\"name\":\"kernel_shape\",\"ints\":[\"2\"],\"type\":\"INTS\"}]},"
            "{\"input\":[\"c\"],\"output\":[\"r\"],\"opType\":\"Relu\"},"
            "{\"input\":[\"r\"],\"output\":[\"m\"],\"opType\":\"GlobalMaxPool\"},"
            "{\"input\":[\"m\"],\"output\":[\"f\"],\"name\":\"/Flatten\",\"opType\":\"Flatten\",\"attribute\":[{\"name\":\"axis\",\"i\":\"1\",\"type\":\"INT\"}]},"
            "{\"input\":[\"f\",\"fc.w\",\"fc.b\"],\"output\":[\"y\"],\"opType\":\"Gemm\",\"attribute\":[{\"name\":\"transB\",\"i\":\"1\",\"type\":\"INT\"}]}],"
            "\"name\":\"g\",\"initializer\":["
            "{\"dims\":[\"6\",\"2\"],\"dataType\":1,\"name\":\"emb.weight\",\"floatData\":[0,0,0.5,-1,1.5,0.25,-0.5,0.75,1,1,-1,0.5]},"
            "{\"dims\":[\"3\",\"2\",\"2\"],\"dataType\":1,\"name\":\"c.w\",\"floatData\":[0.5,-0.25,0.75,0.125,-0.5,1,0.25,-0.75,1.5,0.5,-1,0.25]},"
            "{\"dims\":[\"3\"],\"dataType\":1,\"name\":\"c.b\",\"floatData\":[0.1,-0.2,0.05]},"
            "{\"dims\":[\"1\",\"3\"],\"dataType\":1,\"name\":\"fc.w\",\"floatData\":[1,-1,0.5]},"
            "{\"dims\":[\"1\"],\"dataType\":1,\"name\":\"fc.b\",\"floatData\":[0.5]}],"
            "\"input\":[{\"name\":\"x\",\"type\":{\"tensorType\":{\"elemType\":7,\"shape\":{\"dim\":[{\"dimParam\":\"N\"},{\"dimValue\":\"5\"}]}}}}],"
            "\"output\":[{\"name\":\"y\",\"type\":{\"tensorType\":{\"elemType\":1,\"shape\":{\"dim\":[{\"dimParam\":\"N\"},{\"dimValue\":\"1\"}]}}}}]},"
            "\"opsetImport\":[{\"domain\":\"\",\"version\":\"17\"}]}";
        static const double ew[12] = { 0, 0, 0.5, -1, 1.5, 0.25, -0.5, 0.75, 1, 1, -1, 0.5 };
        static const double cw[12] = { 0.5, -0.25, 0.75, 0.125, -0.5, 1, 0.25, -0.75, 1.5, 0.5, -1, 0.25 };
        static const double cb[3] = { 0.1, -0.2, 0.05 }, fw[3] = { 1, -1, 0.5 };
        FILE *f = fopen(onnx, "w");
        if (f) {
            fputs(model, f);
            fclose(f);
        }
        back = ann_import_onnx(onnx);
        const real ids[5] = { 3, 1, 5, 2, 4 };
        double expected = 0.5;
        for (int fi = 0; fi < 3; fi++) {
            double best = -1e30;
            for (int t = 0; t < 4; t++) {
                double v = cb[fi];
                for (int c = 0; c < 2; c++)
                    for (int j = 0; j < 2; j++)
                        v += cw[(fi * 2 + c) * 2 + j] * ew[(int)ids[t + j] * 2 + c];
                v = v > 0.0 ? v : 0.0;
                if (v > best) best = v;
            }
            expected += fw[fi] * best;
        }
        real out[1];
        int close = back != NULL && ann_get_layer_kind(back, 1) == LAYER_KIND_EMBEDDING;
        if (close) {
            ann_predict(back, ids, out);
            close = fabs(out[0] - expected) < 1e-5;
        }
        TESTEX("External nn.Embedding model (int64 ids) imports and matches a reference", close);
        ann_free_network(back);
        remove(onnx);
    }

    SUITE("Quantization");
    {
        // global max pooling keeps its input's int8 quantization
        enum { N = 64 };
        static real x[N * 2 * 12];
        for (int i = 0; i < N * 24; i++) x[i] = (real)sin(0.37 * i);
        PTensor tx = tensor_create_from_array(N, 24, x);
        PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
        net->print_func = quiet;
        ann_add_input(net, 2, 1, 12);
        ann_add_conv1d(net, 4, 3, 1, 1, ACTIVATION_RELU);
        ann_add_global_maxpool2d(net);
        ann_add_layer(net, 3, LAYER_OUTPUT, ACTIVATION_TANH);
        ann_train_begin(net);
        set_params(net);
        ann_train_end(net);
        PQuantizedNetwork q = ann_quantize(net, tx);
        double worst = 0.0;
        real scale = 1.0f;
        for (int r = 0; q && r < N; r++) {
            real yf[3], yq[3];
            ann_quantized_get_io_params(q, NULL, NULL, &scale, NULL);
            ann_predict(net, &x[r * 24], yf);
            ann_quantized_predict(q, &x[r * 24], yq);
            for (int k = 0; k < 3; k++)
                if (fabs(yf[k] - yq[k]) / scale > worst) worst = fabs(yf[k] - yq[k]) / scale;
        }
        printf("\t  conv1d/global max pool: %.2f steps\n", worst);
        TESTEX("Conv1d and global max pool quantize within 3 steps", q && worst <= 3.0);
        TESTEX("and export a header that uses it", q && ann_export_c_header(NULL, q, "test_gmp.h", "gmp_q") == ERR_OK &&
               file_contains("test_gmp.h", "ANN_RT_GLOBAL_MAXPOOL2D") && file_contains("test_gmp.h", "ANN_RT_VERSION < 6"));
        remove("test_gmp.h");
        ann_free_quantized(q);
        ann_free_network(net);
        tensor_free(tx);
    }
}
