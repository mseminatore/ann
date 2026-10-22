// Shared definitions for the Vulkan compute shaders (tensor_vulkan.c).
// Buffers are float32, row-major, as libann's tensors are.

// Activation_type values from ann.h (tensor_vulkan.c checks they match)
#define ACT_NULL       0
#define ACT_SIGMOID    1
#define ACT_RELU       2
#define ACT_LEAKY_RELU 3
#define ACT_TANH       4
#define ACT_SOFTSIGN   5
#define ACT_SOFTMAX    6
#define ACT_GELU       7

// Some drivers compute tanh as (e^2x - 1) / (e^2x + 1), which is inf / inf for
// large x; tanh(15) is already 1.0 in float
float safe_tanh(float x)
{
    return tanh(clamp(x, -15.0, 15.0));
}

// An activation applied as part of another kernel; softmax (a row operation)
// and ACT_NULL pass x through. Same math as the CPU activations in ann.c.
float activate(float x, uint act)
{
    switch (act)
    {
    case ACT_SIGMOID:    return 1.0 / (1.0 + exp(-x));
    case ACT_RELU:       return max(0.0, x);
    case ACT_LEAKY_RELU: return x > 0.0 ? x : 0.01 * x;
    case ACT_TANH:       return safe_tanh(x);
    case ACT_SOFTSIGN:   return x / (1.0 + abs(x));
    case ACT_GELU:       return 0.5 * x * (1.0 + safe_tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    default:             return x;
    }
}

// The activation's derivative from x, which is the activation's output, or for
// GELU the value before activation (z). ReLU and leaky ReLU test a > 0, which
// is z > 0. ACT_NULL and softmax (whose Jacobian is a row operation) give 1.
float derivative(float x, uint act)
{
    switch (act)
    {
    case ACT_SIGMOID:    return x * (1.0 - x);
    case ACT_RELU:       return x > 0.0 ? 1.0 : 0.0;
    case ACT_LEAKY_RELU: return x > 0.0 ? 1.0 : 0.01;
    case ACT_TANH:       return 1.0 - x * x;
    case ACT_SOFTSIGN:   { float s = 1.0 - abs(x); return s * s; }
    case ACT_GELU:
    {
        float t = safe_tanh(0.7978845608 * (x + 0.044715 * x * x * x));
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    }
    default:             return 1.0;
    }
}

// A conv, pool or upsample layer's geometry: input C x H x W, output F x OH x OW,
// kernel (or pool window, or upsample scale) K, stride S, padding P.
// Activations are [batch][channels][height][width]. Matches WindowDims in tensor_vulkan.c.
struct WindowDims
{
    uint C, H, W, F, OH, OW, K, S, P, batch;
};

// Grid-stride loop over n items: the host caps the workgroup count at the
// device limit (65535 on many GPUs), so each invocation may handle several
#define GRID_LOOP(i, n) \
    for (uint i = gl_GlobalInvocationID.x; i < (n); i += gl_NumWorkGroups.x * gl_WorkGroupSize.x)
