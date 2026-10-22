#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "tensor.h"
#include "testy/test.h"

#define N 20000		// values for the statistics checks

static void quiet(const char *s) { (void)s; }
static void no_errors(int code, const char *msg, const char *fn) { (void)code; (void)msg; (void)fn; }

// 8 -> dense(4, sigmoid) -> dense(8, sigmoid): a small autoencoder
static PNetwork make_net(void)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
    net->print_func = quiet;
    ann_add_layer(net, 8, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 8, LAYER_OUTPUT, ACTIVATION_SIGMOID);
    ann_set_learning_rate(net, 0.01f);
    ann_set_convergence(net, 0.0f);
    ann_set_epoch_limit(net, 20);
    ann_set_batch_size(net, 8);
    return net;
}

// 64 samples of 8 values in [0, 1]
static void make_data(real *x)
{
    for (int i = 0; i < 64 * 8; i++)
        x[i] = (real)(0.5 + 0.5 * sin(0.37 * i));
}

// train a fresh network from seed, with the given noise and thread count;
// returns its output-layer weights in w (8 x 4)
static void train(int noise, int threads, real *w)
{
    real x[64 * 8];
    make_data(x);
    ann_set_num_threads(threads);
    srand(7);
    PNetwork net = make_net();
    if (noise)
        ann_set_input_noise(net, NOISE_GAUSSIAN, 0.2f, 0.0f, 1.0f);
    PTensor tx = tensor_create_from_array(64, 8, x);
    ann_train_network(net, tx, tx, 64);
    memcpy(w, net->layers[2].t_weights->values, 32 * sizeof(real));
    tensor_free(tx);
    ann_free_network(net);
    ann_set_num_threads(0);
}

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("Input Noise");

    SUITE("Settings");
    {
        PNetwork net = make_net();
        Noise_type type;
        real amount, low, high;

        TESTEX("Noise is off by default", ann_get_input_noise(net, &type, &amount, &low, &high) == ERR_OK &&
               type == NOISE_NONE && amount == 0.0f && low == 0.0f && high == 0.0f);
        TESTEX("Settings round trip", ann_set_input_noise(net, NOISE_GAUSSIAN, 0.3f, 0.0f, 1.0f) == ERR_OK &&
               ann_get_input_noise(net, &type, &amount, &low, &high) == ERR_OK &&
               type == NOISE_GAUSSIAN && amount == 0.3f && low == 0.0f && high == 1.0f);
        TESTEX("Any output may be NULL", ann_get_input_noise(net, NULL, NULL, NULL, NULL) == ERR_OK);

        ann_set_error_log_callback(no_errors);
        real nan_value = (real)sqrt(-1.0);
        TESTEX("NULL network", ann_set_input_noise(NULL, NOISE_GAUSSIAN, 0.1f, 0.0f, 0.0f) == ERR_NULL_PTR &&
               ann_get_input_noise(NULL, &type, NULL, NULL, NULL) == ERR_NULL_PTR);
        TESTEX("Unknown type", ann_set_input_noise(net, (Noise_type)99, 0.1f, 0.0f, 0.0f) == ERR_INVALID);
        TESTEX("Negative or NaN amount", ann_set_input_noise(net, NOISE_GAUSSIAN, -0.1f, 0.0f, 0.0f) == ERR_INVALID &&
               ann_set_input_noise(net, NOISE_GAUSSIAN, nan_value, 0.0f, 0.0f) == ERR_INVALID);
        TESTEX("Probability above 1", ann_set_input_noise(net, NOISE_MASKING, 1.5f, 0.0f, 0.0f) == ERR_INVALID &&
               ann_set_input_noise(net, NOISE_SALT_PEPPER, 1.5f, 0.0f, 1.0f) == ERR_INVALID);
        TESTEX("Salt-and-pepper needs low < high", ann_set_input_noise(net, NOISE_SALT_PEPPER, 0.1f, 0.0f, 0.0f) == ERR_INVALID);
        TESTEX("low above high", ann_set_input_noise(net, NOISE_GAUSSIAN, 0.1f, 1.0f, 0.0f) == ERR_INVALID);
        TESTEX("A rejected setting keeps the previous one", ann_get_input_noise(net, &type, &amount, NULL, NULL) == ERR_OK &&
               type == NOISE_GAUSSIAN && amount == 0.3f);
        real v[2] = { 1.0f, 2.0f };
        TESTEX("Apply checks its arguments", ann_apply_input_noise(NULL, v, 2) == ERR_NULL_PTR &&
               ann_apply_input_noise(net, NULL, 2) == ERR_NULL_PTR && ann_apply_input_noise(net, v, -1) == ERR_INVALID);
        ann_clear_error_log_callback();
        ann_free_network(net);
    }

    SUITE("Noise Types");
    {
        static real v[N];
        PNetwork net = make_net();
        double mean = 0.0, var = 0.0;
        int count, low_count, high_count, other;

        for (int i = 0; i < N; i++) v[i] = 0.25f;
        TESTEX("NOISE_NONE leaves values alone", ann_apply_input_noise(net, v, N) == ERR_OK && v[0] == 0.25f && v[N - 1] == 0.25f);

        srand(1);
        ann_set_input_noise(net, NOISE_GAUSSIAN, 0.5f, 0.0f, 0.0f);
        for (int i = 0; i < N; i++) v[i] = 0.0f;
        ann_apply_input_noise(net, v, N);
        for (int i = 0; i < N; i++) mean += v[i];
        mean /= N;
        for (int i = 0; i < N; i++) var += (v[i] - mean) * (v[i] - mean);
        var /= N;
        printf("\t  Gaussian: mean %.4f, std %.4f\n", mean, sqrt(var));
        TESTEX("Gaussian noise has mean 0 and the requested standard deviation",
               fabs(mean) < 0.02 && fabs(sqrt(var) - 0.5) < 0.02);

        ann_set_input_noise(net, NOISE_GAUSSIAN, 1.0f, 0.0f, 1.0f);
        for (int i = 0; i < N; i++) v[i] = 0.5f;
        ann_apply_input_noise(net, v, N);
        low_count = high_count = other = 0;
        for (int i = 0; i < N; i++) {
            if (v[i] < 0.0f || v[i] > 1.0f) other++;
            low_count += v[i] == 0.0f;
            high_count += v[i] == 1.0f;
        }
        TESTEX("Clamped Gaussian noise stays in [low, high]", other == 0 && low_count > 0 && high_count > 0);

        ann_set_input_noise(net, NOISE_MASKING, 0.3f, 0.0f, 0.0f);
        for (int i = 0; i < N; i++) v[i] = 1.0f;
        ann_apply_input_noise(net, v, N);
        count = other = 0;
        for (int i = 0; i < N; i++) {
            count += v[i] == 0.0f;
            other += v[i] != 0.0f && v[i] != 1.0f;
        }
        printf("\t  masking: %.4f zeroed\n", (double)count / N);
        TESTEX("Masking zeros the requested fraction and keeps the rest", fabs((double)count / N - 0.3) < 0.02 && other == 0);

        ann_set_input_noise(net, NOISE_SALT_PEPPER, 0.2f, -1.0f, 2.0f);
        for (int i = 0; i < N; i++) v[i] = 0.5f;
        ann_apply_input_noise(net, v, N);
        low_count = high_count = other = 0;
        for (int i = 0; i < N; i++) {
            low_count += v[i] == -1.0f;
            high_count += v[i] == 2.0f;
            other += v[i] != -1.0f && v[i] != 2.0f && v[i] != 0.5f;
        }
        printf("\t  salt-and-pepper: %.4f low, %.4f high\n", (double)low_count / N, (double)high_count / N);
        TESTEX("Salt-and-pepper sets about half the changed inputs to low and half to high",
               fabs((double)low_count / N - 0.1) < 0.015 && fabs((double)high_count / N - 0.1) < 0.015 && other == 0);
        ann_free_network(net);
    }

    SUITE("Training");
    {
        real x[64 * 8], copy[64 * 8], before[8], after[8];
        make_data(x);
        memcpy(copy, x, sizeof(x));

        srand(3);
        PNetwork net = make_net();
        PTensor tx = tensor_create_from_array(64, 8, x);
        ann_train_network(net, tx, tx, 64);
        ann_predict(net, x, before);
        real loss = ann_evaluate_loss(net, tx, tx);

        ann_set_input_noise(net, NOISE_MASKING, 0.5f, 0.0f, 0.0f);
        ann_predict(net, x, after);
        TESTEX("Predictions don't use the noise", memcmp(before, after, sizeof(before)) == 0);
        TESTEX("Neither does evaluation", ann_evaluate_loss(net, tx, tx) == loss);

        ann_train_begin(net);
        ann_train_step(net, x, x, 64);
        int masked = 0;
        for (int i = 0; i < 64 * 8; i++)
            masked += net->layers[0].t_batch_values->values[i] == 0.0f && x[i] != 0.0f;
        ann_train_end(net);
        TESTEX("ann_train_step trains on noisy inputs", masked > 64 * 8 / 4);
        TESTEX("and leaves the caller's data unchanged", memcmp(x, copy, sizeof(x)) == 0);
        tensor_free(tx);
        ann_free_network(net);

        real plain[32], noisy[32], noisy4[32];
        train(0, 1, plain);
        train(1, 1, noisy);
        train(1, 4, noisy4);
        TESTEX("ann_train_network trains on noisy inputs", memcmp(plain, noisy, sizeof(plain)) != 0);
        TESTEX("with the same results for any thread count", memcmp(noisy, noisy4, sizeof(noisy)) == 0);
    }
}
