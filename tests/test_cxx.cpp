//-----------------------------------------------------------
// libann from C++: the public headers declare C linkage, so a C++ program
// (or the CUDA backend, which nvcc compiles as C++) links against the C
// library. Without the extern "C" blocks this fails to link with undefined
// references to C++-mangled names.
//
// A shared libann exports only its ANN_API functions, so with ANN_TEST_SHARED
// (set by CMake for BUILD_SHARED) the internal helpers ann_gpu_backend.h
// declares are compiled but not called; the static build checks they link.
//-----------------------------------------------------------

#include <cstdio>
#include "ann.h"
#include "ann_gpu_backend.h"
#include "ann_hypertune.h"
#include "json.h"

int main()
{
    int failed = 0;

    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 4, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 3, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

#ifndef ANN_TEST_SHARED
    // the helpers ann.c shares with the GPU backends (internal: not exported by a shared libann)
    failed |= ann_lowest_trained_layer(net) != 1;
    ann_set_layer_trainable(net, 1, 0);
    failed |= ann_lowest_trained_layer(net) != 2;
    failed |= ann_cross_entropy_term(ACTIVATION_SOFTMAX, 1.0f, 1.0f) > 1e-6f;
#endif

    real x[4] = { 0.1f, 0.2f, 0.3f, 0.4f }, y[2];
    failed |= ann_predict(net, x, y) != ERR_OK;
    failed |= y[0] + y[1] < 0.999f || y[0] + y[1] > 1.001f;
    ann_free_network(net);

    PTensor t = tensor_zeros(2, 3);
    failed |= !t || t->rows != 2;
    tensor_free(t);

    JsonValue json;
    failed |= json_parse("{\"a\": 1}", &json) != 0;
    json_free(&json);

    std::printf("libann from C++: %s\n", failed ? "FAILED" : "ok");
    return failed;
}
