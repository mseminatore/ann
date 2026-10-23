//-----------------------------------------------------------
// CPU threading: the thread count setting, results that do not depend on the
// thread count, and networks trained at the same time from several
// application threads.
//-----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"
#include "testy/test.h"

#ifndef _WIN32
#	include <pthread.h>
#endif

static void quiet(const char *s) { (void)s; }

static char g_printed[2048];
static void capture(const char *msg)
{
    size_t len = strlen(g_printed);
    snprintf(g_printed + len, sizeof(g_printed) - len, "%s", msg);
}

#define ROWS 64

// A CNN (conv, pool) or a dense network large enough for its GEMMs to be split
static PNetwork make_net(int cnn)
{
    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    net->print_func = quiet;
    if (cnn) {
        ann_add_input(net, 2, 10, 10);
        ann_add_conv2d(net, 6, 3, 1, 1, ACTIVATION_RELU);
        ann_add_maxpool2d(net, 2, 2);
        ann_add_conv2d(net, 8, 3, 1, 0, ACTIVATION_TANH);
        ann_add_layer(net, 4, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    } else {
        ann_add_layer(net, 200, LAYER_INPUT, ACTIVATION_NULL);
        ann_add_layer(net, 256, LAYER_HIDDEN, ACTIVATION_RELU);
        ann_add_layer(net, 4, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
    }
    ann_set_batch_size(net, 32);
    ann_set_epoch_limit(net, 3);
    ann_set_convergence(net, 0.0f);
    return net;
}

static void make_data(PTensor *x, PTensor *y, int inputs)
{
    *x = tensor_create(ROWS, inputs);
    *y = tensor_zeros(ROWS, 4);
    for (int i = 0; i < ROWS * inputs; i++)
        (*x)->values[i] = (real)((i * 37) % 101) / 101.0f;
    for (int r = 0; r < ROWS; r++)
        (*y)->values[r * 4 + r % 4] = 1.0f;
}

// Train one network from a fixed seed and return a copy of every parameter
static real *train_params(int cnn, size_t *count)
{
    PTensor x, y;
    srand(5);
    PNetwork net = make_net(cnn);
    make_data(&x, &y, net->layers[0].node_count);
    ann_train_network(net, x, y, ROWS);

    size_t n = 0;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++)
            n += (size_t)net->layers[l].params[p].value->rows * net->layers[l].params[p].value->cols;
    real *out = malloc(n * sizeof(real)), *dst = out;
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            memcpy(dst, t->values, (size_t)t->rows * t->cols * sizeof(real));
            dst += t->rows * t->cols;
        }

    *count = n;
    ann_free_network(net);
    tensor_free(x);
    tensor_free(y);
    return out;
}

#ifndef _WIN32
// Online training of a network with fixed starting weights. Epoch training
// draws weights and shuffles from the C library's shared rand(), so networks
// trained at the same time could not match a sequential run; online steps on
// fixed weights use no random numbers.
typedef struct {
    PNetwork net;
    PTensor x, y;
} train_args;

static void set_fixed_weights(PNetwork net)
{
    for (int l = 1; l < net->layer_count; l++)
        for (int p = 0; p < net->layers[l].param_count; p++) {
            PTensor t = net->layers[l].params[p].value;
            for (int i = 0; i < t->rows * t->cols; i++)
                t->values[i] = 0.05f * (real)((i * 13 + l * 7 + p) % 17 - 8);
        }
    net->weights_set = 1;
}

static void *train_thread(void *arg)
{
    train_args *a = (train_args *)arg;
    int in = a->net->layers[0].node_count;
    ann_train_begin(a->net);
    for (int step = 0; step < 20; step++) {
        int row = (step * 32) % ROWS;
        ann_train_step(a->net, &a->x->values[row * in], &a->y->values[row * 4], 32);
    }
    ann_train_end(a->net);
    return NULL;
}

static int same_params(PNetwork a, PNetwork b)
{
    for (int l = 1; l < a->layer_count; l++)
        for (int p = 0; p < a->layers[l].param_count; p++) {
            PTensor ta = a->layers[l].params[p].value, tb = b->layers[l].params[p].value;
            if (memcmp(ta->values, tb->values, (size_t)ta->rows * ta->cols * sizeof(real)) != 0)
                return 0;
        }
    return 1;
}
#endif

void test_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    MODULE("CPU Threading");

    SUITE("Thread Count");
    {
        int initial = ann_get_num_threads();
        TESTEX("Default thread count is at least 1", initial >= 1);
        ann_set_num_threads(3);
        TESTEX("ann_set_num_threads(3) takes effect", ann_get_num_threads() == 3);
        ann_set_num_threads(1);
        TESTEX("ann_set_num_threads(1) runs serially", ann_get_num_threads() == 1);
        ann_set_num_threads(0);
        TESTEX("ann_set_num_threads(0) restores the default", ann_get_num_threads() == initial);
    }

    SUITE("Backend Info");
    {
        Backend_info info;
        TESTEX("ann_get_backend_info rejects NULL", ann_get_backend_info(NULL) == ERR_NULL_PTR);
        ann_set_num_threads(3);
        TESTEX("reports the thread count and number size", ann_get_backend_info(&info) == ERR_OK &&
               info.threads == 3 && info.real_bytes == (int)sizeof(real));
#if defined(__OPTIMIZE__) || (defined(_MSC_VER) && !defined(_DEBUG))
        TESTEX("and that the build is optimized", info.optimized == 1);
#else
        TESTEX("and that the build is unoptimized", info.optimized == 0);
#endif
#if defined(USE_BLAS)
        TESTEX("and a BLAS library", info.blas && strcmp(info.blas, "none") != 0);
#else
        TESTEX("and no BLAS library", info.blas && strcmp(info.blas, "none") == 0);
#endif
#if defined(USE_METAL)
        TESTEX("and the Metal GPU backend, not started", strcmp(info.gpu, "Metal") == 0 && !info.gpu_active);
#elif defined(USE_VULKAN)
        TESTEX("and the Vulkan GPU backend, not started", strcmp(info.gpu, "Vulkan") == 0 && !info.gpu_active);
#elif defined(USE_CUDA)
        TESTEX("and the CUDA GPU backend, not started", strcmp(info.gpu, "CUDA") == 0 && !info.gpu_active);
#else
        TESTEX("and no GPU backend", strcmp(info.gpu, "none") == 0 && !info.gpu_active);
#endif

        // training prints the same as its Backend line
        g_printed[0] = '\0';
        PNetwork net = make_net(0);
        net->print_func = capture;
        ann_print_props(net);
        TESTEX("ann_print_props prints the CPU backend with its threads",
               strstr(g_printed, "Backend:") && strstr(g_printed, "CPU, 3 threads, "));
        TESTEX("and notes an unoptimized build", (strstr(g_printed, "unoptimized build") != NULL) == !info.optimized);
        ann_free_network(net);
        ann_set_num_threads(0);
    }

    SUITE("Results Do Not Depend on the Thread Count");
    {
        const char *names[2] = {"Dense", "CNN"};
        for (int cnn = 0; cnn < 2; cnn++) {
            size_t n1, n4, n8, again_n;
            ann_set_num_threads(1);
            real *one = train_params(cnn, &n1);
            // a BLAS library may not repeat its own results (Accelerate
            // doesn't), and then no thread count can match another
            real *again = train_params(cnn, &again_n);
            int repeatable = again_n == n1 && memcmp(one, again, n1 * sizeof(real)) == 0;
            free(again);
            if (!repeatable) {
                printf("\t%s: the BLAS library's results vary from run to run; not compared\n", names[cnn]);
                free(one);
                continue;
            }
            ann_set_num_threads(4);
            real *four = train_params(cnn, &n4);
            ann_set_num_threads(8);
            real *eight = train_params(cnn, &n8);
            int same = n1 == n4 && n1 == n8 && memcmp(one, four, n1 * sizeof(real)) == 0 &&
                       memcmp(one, eight, n1 * sizeof(real)) == 0;
            printf("\t%s: ", names[cnn]);
            if (cnn) { TESTEX("1, 4 and 8 threads train identical CNN weights", same); }
            else { TESTEX("1, 4 and 8 threads train identical dense weights", same); }
            free(one);
            free(four);
            free(eight);
        }
        ann_set_num_threads(0);
    }

#ifndef _WIN32
    SUITE("Concurrent Training");
    {
        // Two networks trained at the same time from two application threads
        // match the same networks trained one after the other
        train_args a[2], b[2];
        for (int i = 0; i < 2; i++) {
            a[i].net = make_net(i);
            b[i].net = make_net(i);
            set_fixed_weights(a[i].net);
            set_fixed_weights(b[i].net);
            make_data(&a[i].x, &a[i].y, a[i].net->layers[0].node_count);
            make_data(&b[i].x, &b[i].y, b[i].net->layers[0].node_count);
        }

        pthread_t threads[2];
        for (int i = 0; i < 2; i++)
            pthread_create(&threads[i], NULL, train_thread, &a[i]);
        for (int i = 0; i < 2; i++)
            pthread_join(threads[i], NULL);
        for (int i = 0; i < 2; i++)
            train_thread(&b[i]);

        TESTEX("Dense network trained concurrently matches a sequential run", same_params(a[0].net, b[0].net));
        TESTEX("CNN trained concurrently matches a sequential run", same_params(a[1].net, b[1].net));

        for (int i = 0; i < 2; i++) {
            ann_free_network(a[i].net);
            ann_free_network(b[i].net);
            tensor_free(a[i].x); tensor_free(a[i].y);
            tensor_free(b[i].x); tensor_free(b[i].y);
        }
    }
#endif
}
