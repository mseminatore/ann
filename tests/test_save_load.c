#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#if defined(USE_CBLAS)
#	if defined(USE_MKL)
#		include <mkl_cblas.h>
#	else
#		include <cblas.h>
#	endif
#endif

#define TEST_FILE_TXT "test_roundtrip.nna"
#define TEST_FILE_BIN "test_roundtrip.nnb"
#define TEST_FILE_ONNX "test_roundtrip.onnx.json"

// The 2-2 dense network the fixture files below describe:
// weights [0.1 0.2; 0.3 0.4], bias [0.5 0.6], sigmoid output, Adam, MSE
static const real fx_weights[] = {0.1f, 0.2f, 0.3f, 0.4f};
static const real fx_bias[] = {0.5f, 0.6f};

// Header and layer records, in file order, for format version 1 or 2
static int fx_ints(int version, int *out)
{
    int n = 0;
    out[n++] = version; out[n++] = OPT_ADAM; out[n++] = LOSS_MSE; out[n++] = 2;
    if (version == 1) {
        out[n++] = 2; out[n++] = LAYER_INPUT;  out[n++] = ACTIVATION_NULL;
        out[n++] = 2; out[n++] = LAYER_OUTPUT; out[n++] = ACTIVATION_SIGMOID;
    } else {
        // kind layer_type activation c h w config_count
        out[n++] = LAYER_KIND_DENSE; out[n++] = LAYER_INPUT;  out[n++] = ACTIVATION_NULL;
        out[n++] = 2; out[n++] = 1; out[n++] = 1; out[n++] = 0;
        out[n++] = LAYER_KIND_DENSE; out[n++] = LAYER_OUTPUT; out[n++] = ACTIVATION_SIGMOID;
        out[n++] = 2; out[n++] = 1; out[n++] = 1; out[n++] = 0;
    }
    return n;
}

// Parameter values in file order: v1 stores the bias first, v2 follows params[] (W, b)
static int fx_reals(int version, real *out)
{
    int n = 0;
    if (version == 1)
        for (int i = 0; i < 2; i++) out[n++] = fx_bias[i];
    for (int i = 0; i < 4; i++) out[n++] = fx_weights[i];
    if (version != 1)
        for (int i = 0; i < 2; i++) out[n++] = fx_bias[i];
    return n;
}

// Write a fixture model file; drop_last leaves the final value out (truncated file)
static void fx_write(const char *path, int version, int binary, int drop_last)
{
    int ints[32];
    real reals[8];
    int ni = fx_ints(version, ints), nr = fx_reals(version, reals) - (drop_last ? 1 : 0);
    FILE *f = fopen(path, binary ? "wb" : "wt");
    if (!f) return;
    for (int i = 0; i < ni; i++) {
        if (binary) fwrite(&ints[i], sizeof(int), 1, f);
        else fprintf(f, "%d\n", ints[i]);
    }
    for (int i = 0; i < nr; i++) {
        if (binary) fwrite(&reals[i], sizeof(real), 1, f);
        else fprintf(f, "%f\n", reals[i]);
    }
    fclose(f);
}

// 1 if the network matches the fixture
static int fx_matches(PNetwork net)
{
    if (!net || net->layer_count != 2 || ann_get_layer_kind(net, 1) != LAYER_KIND_DENSE ||
        net->layers[1].shape.c != 2 || net->layers[1].shape.h != 1 || net->layers[1].shape.w != 1 ||
        net->layers[1].activation != ACTIVATION_SIGMOID)
        return 0;
    for (int i = 0; i < 4; i++)
        if (fabs(net->layers[1].t_weights->values[i] - fx_weights[i]) > 1e-6) return 0;
    for (int i = 0; i < 2; i++)
        if (fabs(net->layers[1].t_bias->values[i] - fx_bias[i]) > 1e-6) return 0;
    return 1;
}

static PNetwork fx_load(const char *path, int binary)
{
    return binary ? ann_load_network_binary(path) : ann_load_network(path);
}

static void quiet_print(const char *s) { (void)s; }

// TESTEX needs a literal message; label it with the format under test
#define FORMAT_TEST(msg, cond) \
    if (binary) { TESTEX(msg " (binary)", (cond)); } else { TESTEX(msg " (text)", (cond)); }

// returns 1 if all weights and biases of a and b match within tol
static int weights_match(const PNetwork a, const PNetwork b, real tol)
{
    if (a->layer_count != b->layer_count)
        return 0;

    for (int layer = 0; layer < a->layer_count - 1; layer++)
    {
        PTensor wa = a->layers[layer + 1].t_weights, wb = b->layers[layer + 1].t_weights;
        PTensor ba = a->layers[layer + 1].t_bias,    bb = b->layers[layer + 1].t_bias;

        if (wa->rows != wb->rows || wa->cols != wb->cols || ba->cols != bb->cols)
            return 0;

        for (int i = 0; i < wa->rows * wa->cols; i++)
            if (fabs(wa->values[i] - wb->values[i]) > tol)
                return 0;

        for (int i = 0; i < ba->cols; i++)
            if (fabs(ba->values[i] - bb->values[i]) > tol)
                return 0;
    }

    return 1;
}

// verify a loaded network has weights_set and that ann_train_begin()
// does not re-randomize the loaded weights (issue #37)
// (macro because TESTEX requires string-literal names)
#define CHECK_LOADED_WEIGHTS(fmt, original, load_expr)                                      \
    do {                                                                                    \
        PNetwork loaded_ = (load_expr);                                                     \
        TESTEX(fmt ": load succeeds", loaded_ != NULL);                                     \
        if (loaded_) {                                                                      \
            TESTEX(fmt ": weights_set after load", loaded_->weights_set == 1);              \
            TESTEX(fmt ": loaded weights/biases match original",                            \
                   weights_match(original, loaded_, (real)1e-4));                           \
            ann_train_begin(loaded_);                                                       \
            TESTEX(fmt ": ann_train_begin preserves loaded weights",                        \
                   weights_match(original, loaded_, (real)1e-4));                           \
            ann_train_end(loaded_);                                                         \
            ann_free_network(loaded_);                                                      \
        }                                                                                   \
    } while (0)

void test_main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    #if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

    MODULE("Save/Load Serialization");

    SUITE("Text Format");
    {
        PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 2, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.01f);
        
        int save_err = ann_save_network(net, TEST_FILE_TXT);
        TESTEX("Save to text", save_err == 0);
        
        PNetwork loaded = ann_load_network(TEST_FILE_TXT);
        TESTEX("Load from text", loaded != NULL);
        
        if (loaded != NULL) {
            TESTEX("Text: layers preserved", loaded->layer_count == net->layer_count);
            TESTEX("Text: loss type preserved", loaded->loss_type == net->loss_type);
            ann_free_network(loaded);
        }
        
        ann_free_network(net);
        remove(TEST_FILE_TXT);
    }

    SUITE("Binary Format");
    {
        PNetwork net = ann_make_network(OPT_MOMENTUM, LOSS_CATEGORICAL_CROSS_ENTROPY);
        ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 8, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(net, 0.025f);
        
        int save_err = ann_save_network_binary(net, TEST_FILE_BIN);
        TESTEX("Save to binary", save_err == 0);
        
        PNetwork loaded = ann_load_network_binary(TEST_FILE_BIN);
        TESTEX("Load from binary", loaded != NULL);
        
        if (loaded != NULL) {
            TESTEX("Binary: layers preserved", loaded->layer_count == net->layer_count);
            TESTEX("Binary: loss preserved", loaded->loss_type == net->loss_type);
            ann_free_network(loaded);
        }
        
        ann_free_network(net);
        remove(TEST_FILE_BIN);
    }

    SUITE("Round Trip Predictions");
    {
        PNetwork original = ann_make_network(OPT_SGD, LOSS_MSE);
        ann_add_layer(original, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(original, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(original, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);
        ann_set_learning_rate(original, 0.5f);
        ann_set_epoch_limit(original, 500);  // Limit epochs to prevent long runs
        ann_set_convergence(original, 0.01f);  // Stop when loss is low enough
        
        real data[] = {0, 0, 0, 1, 1, 0, 1, 1};
        real targets[] = {0, 1, 1, 0};
        PTensor inputs = tensor_create_from_array(4, 2, data);
        PTensor targets_t = tensor_create_from_array(4, 1, targets);
        
        // Single training call - let convergence/epoch limit handle termination
        ann_train_network(original, inputs, targets_t, 4);
        
        real test_input[] = {0.5f, 0.5f};
        real original_output[1];
        ann_predict(original, test_input, original_output);
        
        ann_save_network(original, TEST_FILE_TXT);
        PNetwork loaded = ann_load_network(TEST_FILE_TXT);
        if (loaded != NULL) {
            real loaded_output[1];
            ann_predict(loaded, test_input, loaded_output);
            real diff = fabsf(original_output[0] - loaded_output[0]);
            TESTEX("Predictions match after save/load", diff < 0.01f);
            ann_free_network(loaded);
        }
        
        ann_free_network(original);
        tensor_free(inputs);
        tensor_free(targets_t);
        remove(TEST_FILE_TXT);
    }

    SUITE("Loaded Weights Preserved");
    {
        PNetwork original = ann_make_network(OPT_ADAM, LOSS_MSE);
        ann_add_layer(original, 2, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(original, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
        ann_add_layer(original, 3, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(original, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

        // initialize weights without training
        ann_train_begin(original);
        ann_train_end(original);

        // He/Xavier init zeroes biases; make them distinct so bias loading is verified
        for (int layer = 0; layer < original->layer_count - 1; layer++)
        {
            PTensor bias = original->layers[layer + 1].t_bias;
            for (int i = 0; i < bias->cols; i++)
                bias->values[i] = (real)(0.1 * (layer + 1) + 0.01 * (i + 1));
        }

        ann_save_network(original, TEST_FILE_TXT);
        ann_save_network_binary(original, TEST_FILE_BIN);
        ann_export_onnx(original, TEST_FILE_ONNX);

        CHECK_LOADED_WEIGHTS("Text", original, ann_load_network(TEST_FILE_TXT));
        CHECK_LOADED_WEIGHTS("Binary", original, ann_load_network_binary(TEST_FILE_BIN));
        CHECK_LOADED_WEIGHTS("ONNX", original, ann_import_onnx(TEST_FILE_ONNX));

        ann_free_network(original);
        remove(TEST_FILE_TXT);
        remove(TEST_FILE_BIN);
        remove(TEST_FILE_ONNX);
    }

    SUITE("Format Versions");
    {
        const char *paths[2] = {"test_fixture.nna", "test_fixture.nnb"};

        for (int binary = 0; binary < 2; binary++) {
            const char *path = paths[binary];
            PNetwork net;

            // v1 files written by earlier libann versions still load
            fx_write(path, 1, binary, 0);
            net = fx_load(path, binary);
            FORMAT_TEST("v1 file loads with the right weights", fx_matches(net));
            ann_free_network(net);

            // v2 records each layer's kind and shape
            fx_write(path, 2, binary, 0);
            net = fx_load(path, binary);
            FORMAT_TEST("v2 file loads with the right weights", fx_matches(net));

            // An all-dense network is saved as v1 so older versions can read it
            if (net) {
                net->print_func = quiet_print;
                if (binary) ann_save_network_binary(net, path);
                else ann_save_network(net, path);
            }
            int version = 0;
            FILE *f = fopen(path, binary ? "rb" : "rt");
            if (f) {
                if (binary) { if (fread(&version, sizeof(int), 1, f) != 1) version = 0; }
                else if (fscanf(f, "%d", &version) != 1) version = 0;
                fclose(f);
            }
            FORMAT_TEST("All-dense network saves as v1", version == 1);
            ann_free_network(net);

            // Truncated and unknown-version files are rejected
            fx_write(path, 1, binary, 1);
            net = fx_load(path, binary);
            FORMAT_TEST("Truncated v1 file returns NULL", net == NULL);
            ann_free_network(net);

            fx_write(path, 2, binary, 1);
            net = fx_load(path, binary);
            FORMAT_TEST("Truncated v2 file returns NULL", net == NULL);
            ann_free_network(net);

            remove(path);
        }

        // Unknown version and unknown layer kind
        FILE *f = fopen("test_fixture.nna", "wt");
        if (f) { fprintf(f, "3\n4\n0\n2\n"); fclose(f); }
        TESTEX("Unknown file version returns NULL", ann_load_network("test_fixture.nna") == NULL);

        int ints[32];
        real reals[8];
        int ni = fx_ints(2, ints), nr = fx_reals(2, reals);
        ints[4 + 7] = 99;   // second layer's kind
        f = fopen("test_fixture.nna", "wt");
        if (f) {
            for (int i = 0; i < ni; i++) fprintf(f, "%d\n", ints[i]);
            for (int i = 0; i < nr; i++) fprintf(f, "%f\n", reals[i]);
            fclose(f);
        }
        TESTEX("Unknown layer kind returns NULL", ann_load_network("test_fixture.nna") == NULL);
        remove("test_fixture.nna");
    }

    SUITE("Error Handling");
    {
        PNetwork non_exist = ann_load_network("nonexistent_xyz.nna");
        TESTEX("Load nonexistent returns NULL", non_exist == NULL);
        
        int save_null = ann_save_network(NULL, TEST_FILE_TXT);
        TESTEX("Save NULL returns error", save_null != 0);
        
        remove(TEST_FILE_TXT);
    }

    TESTEX("Save/load tests completed", 1);
}
