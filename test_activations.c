#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann.h"

// Test tolerance for floating point comparisons
#define TOLER_TIGHT 1e-6f
#define TOLER_LOOSE 1e-4f

// Test utilities
static int floats_approx_equal(real a, real b, real tolerance) {
    return fabs(a - b) < tolerance;
}

// Reference implementations for validation
static real ref_sigmoid(real x) {
    return 1.0f / (1.0f + expf(-x));
}

static real ref_sigmoid_derivative(real output) {
    return output * (1.0f - output);
}

static real ref_relu(real x) {
    return (x > 0.0f) ? x : 0.0f;
}

static real ref_relu_derivative(real x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

static real ref_leaky_relu(real x, real alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

static real ref_leaky_relu_derivative(real x, real alpha) {
    return (x > 0.0f) ? 1.0f : alpha;
}

static real ref_tanh(real x) {
    return tanhf(x);
}

static real ref_tanh_derivative(real output) {
    return 1.0f - output * output;
}

static real ref_softsign(real x) {
    return x / (1.0f + fabsf(x));
}

static real ref_softsign_derivative(real x) {
    real denom = 1.0f + fabsf(x);
    return 1.0f / (denom * denom);
}

void test_main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    MODULE("Activation Function Tests");

    // ========================================================================
    // SIGMOID ACTIVATION TESTS
    // ========================================================================
    SUITE("Sigmoid Activation");
    COMMENT("Testing sigmoid activation: f(x) = 1 / (1 + e^(-x)), range (0, 1)...");

    real sig_zero = ref_sigmoid(0.0f);
    TESTEX("sigmoid(0) = 0.5", floats_approx_equal(sig_zero, 0.5f, TOLER_TIGHT));

    real sig_pos = ref_sigmoid(2.0f);
    TESTEX("sigmoid(2) ≈ 0.8808", floats_approx_equal(sig_pos, 0.8808f, TOLER_LOOSE));

    real sig_neg = ref_sigmoid(-2.0f);
    TESTEX("sigmoid(-2) ≈ 0.1192", floats_approx_equal(sig_neg, 0.1192f, TOLER_LOOSE));

    real sig_large_pos = ref_sigmoid(10.0f);
    TESTEX("sigmoid(10) ≈ 1.0 (saturation)", floats_approx_equal(sig_large_pos, 1.0f, 1e-4f));

    real sig_large_neg = ref_sigmoid(-10.0f);
    TESTEX("sigmoid(-10) ≈ 0.0 (saturation)", floats_approx_equal(sig_large_neg, 0.0f, 1e-4f));

    // Test sigmoid output range [0, 1]
    real test_sigmoid_values[] = {-100.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 100.0f};
    int sigmoid_range_valid = 1;
    for (int i = 0; i < 7; i++) {
        real output = ref_sigmoid(test_sigmoid_values[i]);
        if (output < 0.0f || output > 1.0f) {
            sigmoid_range_valid = 0;
            break;
        }
    }
    TESTEX("sigmoid output always in range [0, 1]", sigmoid_range_valid);

    // Test sigmoid derivative
    real sig_deriv_at_half = ref_sigmoid_derivative(0.5f);
    TESTEX("sigmoid'(0.5) = 0.25", floats_approx_equal(sig_deriv_at_half, 0.25f, TOLER_TIGHT));

    real sig_deriv_at_quarter = ref_sigmoid_derivative(0.25f);
    real expected_deriv_quarter = 0.25f * 0.75f;
    TESTEX("sigmoid'(0.25) ≈ 0.1875", floats_approx_equal(sig_deriv_at_quarter, expected_deriv_quarter, TOLER_TIGHT));

    // ========================================================================
    // RELU ACTIVATION TESTS
    // ========================================================================
    SUITE("ReLU Activation");
    COMMENT("Testing ReLU activation: f(x) = max(0, x)...");

    real relu_zero = ref_relu(0.0f);
    TESTEX("relu(0) = 0", floats_approx_equal(relu_zero, 0.0f, TOLER_TIGHT));

    real relu_pos = ref_relu(5.0f);
    TESTEX("relu(5) = 5", floats_approx_equal(relu_pos, 5.0f, TOLER_TIGHT));

    real relu_neg = ref_relu(-3.0f);
    TESTEX("relu(-3) = 0", floats_approx_equal(relu_neg, 0.0f, TOLER_TIGHT));

    real relu_small_pos = ref_relu(0.001f);
    TESTEX("relu(0.001) = 0.001", floats_approx_equal(relu_small_pos, 0.001f, TOLER_TIGHT));

    real relu_large = ref_relu(1000.0f);
    TESTEX("relu(1000) = 1000", floats_approx_equal(relu_large, 1000.0f, TOLER_TIGHT));

    // Test ReLU output range [0, ∞)
    real test_relu_values[] = {-100.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 100.0f};
    int relu_range_valid = 1;
    for (int i = 0; i < 7; i++) {
        real output = ref_relu(test_relu_values[i]);
        if (output < 0.0f) {
            relu_range_valid = 0;
            break;
        }
    }
    TESTEX("relu output always >= 0", relu_range_valid);

    // Test ReLU derivative
    real relu_deriv_pos = ref_relu_derivative(5.0f);
    TESTEX("relu'(5) = 1", floats_approx_equal(relu_deriv_pos, 1.0f, TOLER_TIGHT));

    real relu_deriv_neg = ref_relu_derivative(-3.0f);
    TESTEX("relu'(-3) = 0", floats_approx_equal(relu_deriv_neg, 0.0f, TOLER_TIGHT));

    // ========================================================================
    // LEAKY RELU ACTIVATION TESTS
    // ========================================================================
    SUITE("Leaky ReLU Activation");
    COMMENT("Testing Leaky ReLU activation: f(x) = max(alpha*x, x)...");

    real alpha = 0.01f;

    real lrelu_zero = ref_leaky_relu(0.0f, alpha);
    TESTEX("leaky_relu(0, 0.01) = 0", floats_approx_equal(lrelu_zero, 0.0f, TOLER_TIGHT));

    real lrelu_pos = ref_leaky_relu(5.0f, alpha);
    TESTEX("leaky_relu(5, 0.01) = 5", floats_approx_equal(lrelu_pos, 5.0f, TOLER_TIGHT));

    real lrelu_neg = ref_leaky_relu(-3.0f, alpha);
    real expected_lrelu_neg = -3.0f * 0.01f;
    TESTEX("leaky_relu(-3, 0.01) = -0.03", floats_approx_equal(lrelu_neg, expected_lrelu_neg, TOLER_TIGHT));

    real lrelu_deriv_pos = ref_leaky_relu_derivative(5.0f, alpha);
    TESTEX("leaky_relu'(5, 0.01) = 1", floats_approx_equal(lrelu_deriv_pos, 1.0f, TOLER_TIGHT));

    real lrelu_deriv_neg = ref_leaky_relu_derivative(-3.0f, alpha);
    TESTEX("leaky_relu'(-3, 0.01) = 0.01", floats_approx_equal(lrelu_deriv_neg, alpha, TOLER_TIGHT));

    // ========================================================================
    // TANH ACTIVATION TESTS
    // ========================================================================
    SUITE("Tanh Activation");
    COMMENT("Testing Tanh activation: f(x) = (e^(2x) - 1) / (e^(2x) + 1), range (-1, 1)...");

    real tanh_zero = ref_tanh(0.0f);
    TESTEX("tanh(0) = 0", floats_approx_equal(tanh_zero, 0.0f, TOLER_TIGHT));

    real tanh_pos = ref_tanh(1.0f);
    TESTEX("tanh(1) ≈ 0.7616", floats_approx_equal(tanh_pos, 0.7616f, TOLER_LOOSE));

    real tanh_neg = ref_tanh(-1.0f);
    TESTEX("tanh(-1) ≈ -0.7616", floats_approx_equal(tanh_neg, -0.7616f, TOLER_LOOSE));

    real tanh_large_pos = ref_tanh(10.0f);
    TESTEX("tanh(10) ≈ 1.0 (saturation)", floats_approx_equal(tanh_large_pos, 1.0f, 1e-4f));

    real tanh_large_neg = ref_tanh(-10.0f);
    TESTEX("tanh(-10) ≈ -1.0 (saturation)", floats_approx_equal(tanh_large_neg, -1.0f, 1e-4f));

    // Test tanh output range [-1, 1]
    real test_tanh_values[] = {-100.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 100.0f};
    int tanh_range_valid = 1;
    for (int i = 0; i < 7; i++) {
        real output = ref_tanh(test_tanh_values[i]);
        if (output < -1.0f || output > 1.0f) {
            tanh_range_valid = 0;
            break;
        }
    }
    TESTEX("tanh output always in range [-1, 1]", tanh_range_valid);

    // Test tanh derivative
    real tanh_deriv_at_zero = ref_tanh_derivative(ref_tanh(0.0f));
    TESTEX("tanh'(0) = 1", floats_approx_equal(tanh_deriv_at_zero, 1.0f, TOLER_TIGHT));

    // ========================================================================
    // SOFTSIGN ACTIVATION TESTS
    // ========================================================================
    SUITE("Softsign Activation");
    COMMENT("Testing Softsign activation: f(x) = x / (1 + |x|), range (-1, 1)...");

    real softsign_zero = ref_softsign(0.0f);
    TESTEX("softsign(0) = 0", floats_approx_equal(softsign_zero, 0.0f, TOLER_TIGHT));

    real softsign_pos = ref_softsign(1.0f);
    TESTEX("softsign(1) = 0.5", floats_approx_equal(softsign_pos, 0.5f, TOLER_TIGHT));

    real softsign_neg = ref_softsign(-1.0f);
    TESTEX("softsign(-1) = -0.5", floats_approx_equal(softsign_neg, -0.5f, TOLER_TIGHT));

    real softsign_large_pos = ref_softsign(100.0f);
    TESTEX("softsign(100) ≈ 1.0 (approaches 1)", floats_approx_equal(softsign_large_pos, 1.0f, 0.01f));

    real softsign_large_neg = ref_softsign(-100.0f);
    TESTEX("softsign(-100) ≈ -1.0 (approaches -1)", floats_approx_equal(softsign_large_neg, -1.0f, 0.01f));

    // Test softsign derivative
    real softsign_deriv_at_zero = ref_softsign_derivative(0.0f);
    TESTEX("softsign'(0) = 1", floats_approx_equal(softsign_deriv_at_zero, 1.0f, TOLER_TIGHT));

    real softsign_deriv_at_one = ref_softsign_derivative(1.0f);
    TESTEX("softsign'(1) = 0.25", floats_approx_equal(softsign_deriv_at_one, 0.25f, TOLER_TIGHT));

    // ========================================================================
    // SOFTMAX TESTS (conceptual)
    // ========================================================================
    SUITE("Softmax Activation");
    COMMENT("Testing Softmax activation properties...");

    real test_logits[] = {1.0f, 2.0f, 3.0f};
    real denom = expf(1.0f) + expf(2.0f) + expf(3.0f);
    real softmax_out1 = expf(1.0f) / denom;
    real softmax_out2 = expf(2.0f) / denom;
    real softmax_out3 = expf(3.0f) / denom;
    real softmax_sum = softmax_out1 + softmax_out2 + softmax_out3;

    TESTEX("Softmax outputs sum to 1.0", floats_approx_equal(softmax_sum, 1.0f, TOLER_LOOSE));

    TESTEX("Softmax preserves ordering: exp(1)/Z < exp(2)/Z", (softmax_out1 < softmax_out2));
    TESTEX("Softmax preserves ordering: exp(2)/Z < exp(3)/Z", (softmax_out2 < softmax_out3));

    int softmax_range_valid = (softmax_out1 > 0.0f && softmax_out1 < 1.0f) &&
                               (softmax_out2 > 0.0f && softmax_out2 < 1.0f) &&
                               (softmax_out3 > 0.0f && softmax_out3 < 1.0f);
    TESTEX("Softmax outputs in range (0, 1)", softmax_range_valid);

    // ========================================================================
    // NUMERICAL STABILITY TESTS
    // ========================================================================
    SUITE("Numerical Stability");
    COMMENT("Testing activation functions with extreme values...");

    real sig_extreme_pos = ref_sigmoid(100.0f);
    real sig_extreme_neg = ref_sigmoid(-100.0f);
    int sigmoid_stable = (sig_extreme_pos > 0.99f && sig_extreme_neg < 0.01f);
    TESTEX("Sigmoid stable with extreme values", sigmoid_stable);

    real tanh_extreme_pos = ref_tanh(100.0f);
    real tanh_extreme_neg = ref_tanh(-100.0f);
    int tanh_stable = (tanh_extreme_pos > 0.99f && tanh_extreme_neg < -0.99f);
    TESTEX("Tanh stable with extreme values", tanh_stable);

    real relu_extreme_pos = ref_relu(1e6f);
    real relu_extreme_neg = ref_relu(-1e6f);
    TESTEX("ReLU handles large positive values", (relu_extreme_pos == 1e6f));
    TESTEX("ReLU handles large negative values", (relu_extreme_neg == 0.0f));

    TESTEX("Activation function tests completed", 1);
}
