/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

//================================================================================================
// Vulkan GPU backend: inference and training on any Vulkan 1.1 GPU (AMD, Intel, NVIDIA) on
// Windows and Linux, for every layer kind (dense, conv, max pool, upsample, flatten, reshape,
// global average pool, batch norm).
// See docs/GPU_VULKAN.md.
//
// - The Vulkan loader is opened at run time (vulkan-1.dll / libvulkan.so.1), so a build with
//   USE_VULKAN runs on machines without a Vulkan driver; ann_gpu_init() then reports no GPU.
// - Kernels are GLSL compute shaders in shaders/vk, compiled to SPIR-V by glslc at build time
//   and embedded by cmake/embed_spirv.cmake (shaders_vk_spv.h). GEMM is gemm.comp; there is
//   no vendor BLAS.
// - Every tensor's gpu_buf is a VkBuf. Buffers live in device-local memory; when that memory
//   is also host-visible and large (integrated GPUs, discrete GPUs with resizable BAR) they
//   are mapped and the host copies in and out directly, as with Metal's shared buffers,
//   otherwise copies go through a host-visible staging buffer, as with CUDA.
// - Each operation (a training step, a forward pass, an upload) is recorded into one command
//   buffer, in one of TRAIN_SLOTS slots. Training steps are submitted without waiting, so the
//   next batch is assembled while the GPU runs this one; other operations wait for the steps
//   in flight, then for themselves. Dispatches are separated by full memory barriers.
// - Descriptor sets come from pools reset before each command buffer (push descriptors are
//   not available on every driver).
//================================================================================================

#ifdef USE_VULKAN

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann.h"
#include "ann_gpu_backend.h"
#include "shaders_vk_spv.h"

// The shaders work in float; a double build of libann cannot use this backend
typedef char vk_real_must_be_float[sizeof(real) == sizeof(float) ? 1 : -1];

// shaders/vk/common.glsl numbers the activations as ann.h does
typedef char vk_activation_values_match[(ACTIVATION_NULL == 0 && ACTIVATION_SIGMOID == 1 &&
    ACTIVATION_RELU == 2 && ACTIVATION_LEAKY_RELU == 3 && ACTIVATION_TANH == 4 &&
    ACTIVATION_SOFTSIGN == 5 && ACTIVATION_SOFTMAX == 6 && ACTIVATION_GELU == 7) ? 1 : -1];

//-----------------------------------------------------------
// Vulkan entry points, loaded at run time
//-----------------------------------------------------------
#define VK_GLOBAL_FUNCS(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties)

#define VK_INSTANCE_FUNCS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr)

#define VK_DEVICE_FUNCS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines) \
    X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) \
    X(vkResetCommandBuffer) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier) \
    X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCreateFence) X(vkResetFences) X(vkWaitForFences) \
    X(vkQueueSubmit) X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) \
    X(vkGetQueryPoolResults)

#define VK_DECLARE(f) static PFN_##f f;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
VK_GLOBAL_FUNCS(VK_DECLARE)
VK_INSTANCE_FUNCS(VK_DECLARE)
VK_DEVICE_FUNCS(VK_DECLARE)
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;

//-----------------------------------------------------------
// Kernels: one pipeline each (gemm.comp gives three, by transpose)
//-----------------------------------------------------------
typedef enum {
    K_GEMM_NN, K_GEMM_NT, K_GEMM_TN,
    K_BIAS_ACT, K_SOFTMAX, K_ACT_DERIV, K_SOFTMAX_JACOBIAN, K_OUTPUT_DELTA_LOSS,
    K_BIAS_GRAD_SUM, K_OPTIMIZER, K_DROPOUT,
    K_CONV_IM2COL, K_CONV_OUTPUT, K_CONV_GRAD_ROWS, K_CONV_COL2IM,
    K_MAXPOOL_FORWARD, K_MAXPOOL_BACKWARD, K_UPSAMPLE, K_GLOBAL_AVGPOOL, K_BATCHNORM, K_SUM_PARTIALS,
    K_CONV_DIRECT_FORWARD, K_CONV_DIRECT_INPUT_GRAD, K_CONV_DIRECT_WEIGHT_GRAD,
    K_COUNT
} Kernel;

#define SPV(name) spv_##name, sizeof(spv_##name)
static const struct {
    const uint32_t *code;
    size_t bytes;
    uint32_t trans_a, trans_b;  // gemm.comp's specialization constants
} g_kernels[K_COUNT] = {
    { SPV(gemm), 0, 0 },
    { SPV(gemm), 0, 1 },
    { SPV(gemm), 1, 0 },
    { SPV(bias_act), 0, 0 },
    { SPV(softmax), 0, 0 },
    { SPV(act_deriv), 0, 0 },
    { SPV(softmax_jacobian), 0, 0 },
    { SPV(output_delta_loss), 0, 0 },
    { SPV(bias_grad_sum), 0, 0 },
    { SPV(optimizer), 0, 0 },
    { SPV(dropout), 0, 0 },
    { SPV(conv_im2col), 0, 0 },
    { SPV(conv_output), 0, 0 },
    { SPV(conv_grad_rows), 0, 0 },
    { SPV(conv_col2im), 0, 0 },
    { SPV(maxpool_forward), 0, 0 },
    { SPV(maxpool_backward), 0, 0 },
    { SPV(upsample), 0, 0 },
    { SPV(global_avgpool), 0, 0 },
    { SPV(batchnorm), 0, 0 },
    { SPV(sum_partials), 0, 0 },
    { SPV(conv_direct_forward), 0, 0 },
    { SPV(conv_direct_input_grad), 0, 0 },
    { SPV(conv_direct_weight_grad), 0, 0 },
};

// Kernel names for the ANN_VK_PROFILE report; the last is the copies and fills
static const char *const g_kernel_names[K_COUNT + 1] = {
    "gemm (A * B)", "gemm (A * B^T)", "gemm (A^T * B)",
    "bias_act", "softmax", "act_deriv", "softmax_jacobian", "output_delta_loss",
    "bias_grad_sum", "optimizer", "dropout",
    "conv_im2col", "conv_output", "conv_grad_rows", "conv_col2im",
    "maxpool_forward", "maxpool_backward", "upsample", "global_avgpool", "batchnorm", "sum_partials",
    "conv_direct_forward", "conv_direct_input_grad", "conv_direct_weight_grad",
    "copy / fill",
};
#define K_TRANSFER K_COUNT

// Timestamps a command buffer records for ANN_VK_PROFILE; commands beyond this go unmeasured
#define VK_PROFILE_QUERIES  4096

// Optimizer codes in shaders/vk/optimizer.comp
enum { VK_OPT_SGD, VK_OPT_MOMENTUM, VK_OPT_ADAGRAD, VK_OPT_RMSPROP, VK_OPT_ADAM };

#define VK_MAX_BINDINGS     8       // storage buffers a kernel binds
#define VK_PUSH_BYTES       64      // push constant space a kernel uses
#define VK_POOL_SETS        256     // descriptor sets per pool
#define VK_MAX_POOLS        256     // pools a command buffer may use
// gemm.comp's tile shapes (rows x columns of the result), each 256 invocations of 4 x 4 results:
// square, and thin ones for results with few rows or columns (a conv's filters, a small batch)
#define GEMM_SHAPES 5
static const uint32_t g_gemm_shapes[GEMM_SHAPES][2] = {
    { 64, 64 }, { 128, 32 }, { 32, 128 }, { 256, 16 }, { 16, 256 },
};

// What ANN_VK_PROFILE reports separately: each Kernel, K_TRANSFER (copies and fills), and each
// GEMM by layout and tile shape, from PROFILE_GEMM on
#define PROFILE_GEMM        (K_COUNT + 1)
// A thin tile shape replaces the square one when it pads the result this many times less
#define GEMM_THIN_GAIN      3
// sum_partials.comp gives each element one invocation, not a workgroup, up to this many slices
#define SUM_PER_INVOCATION  32
#define PROFILE_KINDS       (PROFILE_GEMM + 3 * GEMM_SHAPES)
#define EVAL_BATCH_BYTES    ((size_t)256 << 20)
#define EVAL_MAX_ROWS       (1 << 20)

//-----------------------------------------------------------
// Work buffers kept per layer between a batch's forward and backward passes (training and
// inference never overlap, so they share them):
//   SCRATCH_COLS  - conv im2col columns [batch*OH*OW x C*K*K]; max pool argmax [batch x nodes]
//   SCRATCH_ROWS  - conv GEMM output rows, then dZ rows [batch*OH*OW x F]
//   SCRATCH_DCOLS - conv column gradients [batch*OH*OW x C*K*K]
//   SCRATCH_PARTIAL - direct conv weight gradient partial sums [chunks x F*(C*K*K + 1)]
//-----------------------------------------------------------
enum { SCRATCH_COLS, SCRATCH_ROWS, SCRATCH_DCOLS, SCRATCH_PARTIAL, SCRATCH_SLOTS };

// Split-K GEMMs aim for about this many workgroups, with at least SPLIT_MIN_DEPTH of K each,
// and their partial sums fit in g_vk.split (SPLIT_FLOATS floats). On the Radeon HD 7700, 64 to
// 256 workgroups ran within 2% of each other; a depth of 32 beat 64 and 128 by 4-30% on dense
// networks, and 16 made too many slices.
#define SPLIT_GROUPS        128
#define SPLIT_MIN_DEPTH     32
#define SPLIT_FLOATS        ((uint64_t)1 << 20)

//-----------------------------------------------------------
// A tensor's GPU buffer (t->gpu_buf), or a backend work buffer
//-----------------------------------------------------------
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;          // bytes
    void *map;                  // host address when the host copies in and out directly
} VkBuf;

// Training steps that can be in flight at once (see vk_train_batch)
#define TRAIN_SLOTS 2

//-----------------------------------------------------------
// A slot: a command buffer with what one operation needs of its own. vk_train_batch() submits a
// step without waiting, so the CPU assembles the next batch while the GPU runs this one; each
// slot keeps its own copies of what the CPU writes for a step (inputs, targets, dropout masks,
// through its staging buffer) and reads back (per-row losses). Weights, gradients, activations
// and layer work buffers are shared: steps run in submission order, each command buffer
// starting with a barrier on the work before it.
//-----------------------------------------------------------
typedef struct {
    VkCommandBuffer cmd;
    VkFence fence;
    int pending;                // submitted and not yet waited for
    VkQueryPool queries;        // ANN_VK_PROFILE timestamps: one at the start, one after each command
    uint32_t query_count;
    uint8_t query_kind[VK_PROFILE_QUERIES];     // the profile kind (PROFILE_KINDS) each timestamp ends
    VkDescriptorPool pools[VK_MAX_POOLS];
    int pool_count, pool_cur;   // pools created, and the one sets come from now
    VkBuf *staging;             // host-visible; uploads and downloads pass through it
    VkDeviceSize staging_used;

    // A training step's own buffers and results
    VkBuf *inputs, *targets, *row_loss;
    VkBuf **masks;              // dropout masks by layer
    int mask_layers;            // layers masks has room for
    const float *loss_src;      // the step's per-row losses, readable once it completes; NULL
                                // when the slot's operation is not a training step
    int batch_size;
    int divisor;                // batch loss = sum of row losses / divisor
} VkSlot;

typedef struct {
    int initialized;
    int init_failed;
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    int device_index;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    VkPhysicalDeviceMemoryProperties mem;
    uint32_t max_groups[3];     // maxComputeWorkGroupCount x, y, z
    int map_device;             // device buffers are host-visible and mapped (no staging)

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipe_layout;
    VkPipeline pipelines[K_COUNT];
    VkPipeline gemm_pipes[3][GEMM_SHAPES];  // gemm.comp by (NN, NT, TN) and tile shape

    VkCommandPool cmd_pool;
    VkSlot slots[TRAIN_SLOTS];
    int next_slot;              // the slot the next operation uses (the older pending one)
    VkSlot *rec;                // the slot being recorded; see cmd_begin()
    int profile;                // ANN_VK_PROFILE: time each command on the GPU
    double ns_per_tick;         // timestamp period
    double profile_ns[PROFILE_KINDS];       // GPU time and commands by profile kind
    unsigned long long profile_calls[PROFILE_KINDS];
    int record_failed;          // a command could not be recorded; the submit fails
    int barrier_pending;        // a command since the last barrier wrote memory

    VkBuf *infer[2];            // inference ping-pong buffers
    VkBuf **scratch;            // per-layer work buffers, scratch[layer * SCRATCH_SLOTS + slot]
    VkBuf *split;               // split-K GEMM partial sums, SPLIT_FLOATS floats
    int scratch_layers;         // layers scratch has room for
    int training_buffers_ready; // optimizer state was uploaded since upload_network()
    real loss_sum;              // losses of the steps finished since vk_train_finish()
    int train_failed;           // a step since vk_train_finish() did not complete
} VulkanContext;

static VulkanContext g_vk = {0};

//================================================================================================
// Loader and device setup
//================================================================================================

// Open the Vulkan loader and find vkGetInstanceProcAddr; 0 if there is no Vulkan runtime
static int load_loader(void)
{
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib)
        return 0;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(lib, "vkGetInstanceProcAddr");
#else
#  if defined(__APPLE__)
    const char *names[] = { "libvulkan.1.dylib", "libvulkan.dylib", NULL };
#  else
    const char *names[] = { "libvulkan.so.1", "libvulkan.so", NULL };
#  endif
    void *lib = NULL;
    for (int i = 0; names[i] && !lib; i++)
        lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!lib)
        return 0;
    void *sym = dlsym(lib, "vkGetInstanceProcAddr");
    memcpy(&vkGetInstanceProcAddr, &sym, sizeof(sym));     // object to function pointer, as POSIX allows
#endif
    return vkGetInstanceProcAddr != NULL;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT types,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                     void *user)
{
    (void)types; (void)user;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
    return VK_FALSE;
}

// Create the instance, with the validation layer when ANN_VK_DEBUG is set
static int create_instance(void)
{
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "libann";
    app.pEngineName = "libann";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;

    const char *layer = "VK_LAYER_KHRONOS_validation";
    const char *exts[2] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME };
    // with the layer, also check synchronization: barriers between dispatches and between
    // the training steps in flight
    VkValidationFeatureEnableEXT sync = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT features = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
    features.enabledValidationFeatureCount = 1;
    features.pEnabledValidationFeatures = &sync;
    int debug = 0;
    const char *env = getenv("ANN_VK_DEBUG");
    if (env && *env && strcmp(env, "0") != 0)
    {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, NULL);
        VkLayerProperties *props = (VkLayerProperties *)calloc(count ? count : 1, sizeof(*props));
        if (props && vkEnumerateInstanceLayerProperties(&count, props) == VK_SUCCESS)
            for (uint32_t i = 0; i < count; i++)
                if (strcmp(props[i].layerName, layer) == 0)
                    debug = 1;
        free(props);
        if (debug)
        {
            ci.pNext = &features;
            ci.enabledLayerCount = 1;
            ci.ppEnabledLayerNames = &layer;
            ci.enabledExtensionCount = 2;
            ci.ppEnabledExtensionNames = exts;
        }
        else
            fprintf(stderr, "[Vulkan] ANN_VK_DEBUG: validation layer not installed\n");
    }

    VkResult res = vkCreateInstance(&ci, NULL, &g_vk.instance);
    if (res != VK_SUCCESS)
    {
        fprintf(stderr, "[Vulkan] vkCreateInstance failed (%d)\n", (int)res);
        return 0;
    }

#define VK_LOAD_INSTANCE(f) f = (PFN_##f)vkGetInstanceProcAddr(g_vk.instance, #f); if (!f) return 0;
    VK_INSTANCE_FUNCS(VK_LOAD_INSTANCE)
#undef VK_LOAD_INSTANCE

    if (debug)
    {
        vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g_vk.instance, "vkCreateDebugUtilsMessengerEXT");
        if (vkCreateDebugUtilsMessengerEXT)
        {
            VkDebugUtilsMessengerCreateInfoEXT mi = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
            mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mi.pfnUserCallback = debug_callback;
            VkDebugUtilsMessengerEXT messenger;
            vkCreateDebugUtilsMessengerEXT(g_vk.instance, &mi, NULL, &messenger);
        }
    }
    return 1;
}

// A queue family with compute, or -1
static int compute_queue_family(VkPhysicalDevice pd)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, NULL);
    VkQueueFamilyProperties props[16];
    if (count > 16)
        count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props);
    for (uint32_t i = 0; i < count; i++)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return (int)i;
    return -1;
}

// Rank a device: discrete GPUs first, then integrated, virtual, and CPU (e.g. lavapipe) ones
static int device_rank(VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
    default:                                     return 0;
    }
}

// Choose the device: ANN_VK_DEVICE=<index> if set, else the best ranked Vulkan 1.1 device
static int pick_device(int *queue_family)
{
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(g_vk.instance, &count, NULL) != VK_SUCCESS || count == 0)
    {
        fprintf(stderr, "[Vulkan] No Vulkan devices found\n");
        return 0;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice *)calloc(count, sizeof(*devices));
    if (!devices)
        return 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &count, devices);

    const char *env = getenv("ANN_VK_DEVICE");
    int want = env && *env ? atoi(env) : -1;
    int best = -1, best_rank = -1;
    for (uint32_t i = 0; i < count; i++)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.apiVersion < VK_API_VERSION_1_1 || compute_queue_family(devices[i]) < 0 ||
            props.limits.maxPerStageDescriptorStorageBuffers < VK_MAX_BINDINGS ||
            props.limits.maxPushConstantsSize < VK_PUSH_BYTES)
            continue;
        int rank = device_rank(props.deviceType);
        if (want >= 0 ? (int)i == want : rank > best_rank)
        {
            best = (int)i;
            best_rank = rank;
        }
    }
    if (best < 0)
    {
        if (want >= 0)
            fprintf(stderr, "[Vulkan] ANN_VK_DEVICE=%d is not a usable device (%u found)\n", want, count);
        else
            fprintf(stderr, "[Vulkan] No Vulkan 1.1 device with compute found\n");
        free(devices);
        return 0;
    }

    g_vk.physical = devices[best];
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_vk.physical, &props);
    g_vk.max_groups[0] = props.limits.maxComputeWorkGroupCount[0];
    g_vk.max_groups[1] = props.limits.maxComputeWorkGroupCount[1];
    g_vk.max_groups[2] = props.limits.maxComputeWorkGroupCount[2];
    g_vk.ns_per_tick = props.limits.timestampPeriod;
    *queue_family = compute_queue_family(g_vk.physical);
    g_vk.device_index = best;
    memcpy(g_vk.device_name, props.deviceName, sizeof(g_vk.device_name));
    return 1;
}

// Map device buffers when device-local memory is host-visible and about as large as the
// device's memory (integrated GPUs, resizable BAR), not a small window into it.
// ANN_VK_STAGING=1 forces copies through the staging buffer instead; ANN_VK_STAGING=0 maps
// device buffers whenever any device-local memory is host-visible (for testing that path).
static void choose_memory_model(void)
{
    vkGetPhysicalDeviceMemoryProperties(g_vk.physical, &g_vk.mem);

    VkDeviceSize largest = 0;
    for (uint32_t h = 0; h < g_vk.mem.memoryHeapCount; h++)
        if ((g_vk.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && g_vk.mem.memoryHeaps[h].size > largest)
            largest = g_vk.mem.memoryHeaps[h].size;

    const char *env = getenv("ANN_VK_STAGING");
    int force = env && *env ? (strcmp(env, "0") == 0 ? 0 : 1) : -1;
    if (force == 0)
        largest = 0;

    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    g_vk.map_device = 0;
    for (uint32_t t = 0; t < g_vk.mem.memoryTypeCount; t++)
    {
        const VkMemoryType *type = &g_vk.mem.memoryTypes[t];
        if ((type->propertyFlags & want) == want && g_vk.mem.memoryHeaps[type->heapIndex].size >= largest / 2)
            g_vk.map_device = 1;
    }
    if (force == 1)
        g_vk.map_device = 0;
}

static int create_device(int queue_family)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = (uint32_t)queue_family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;

    VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;

    VkResult res = vkCreateDevice(g_vk.physical, &di, NULL, &g_vk.device);
    if (res != VK_SUCCESS)
    {
        fprintf(stderr, "[Vulkan] vkCreateDevice failed (%d)\n", (int)res);
        return 0;
    }

#define VK_LOAD_DEVICE(f) f = (PFN_##f)vkGetDeviceProcAddr(g_vk.device, #f); if (!f) return 0;
    VK_DEVICE_FUNCS(VK_LOAD_DEVICE)
#undef VK_LOAD_DEVICE

    vkGetDeviceQueue(g_vk.device, (uint32_t)queue_family, 0, &g_vk.queue);

    VkCommandPoolCreateInfo pi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = (uint32_t)queue_family;
    if (vkCreateCommandPool(g_vk.device, &pi, NULL, &g_vk.cmd_pool) != VK_SUCCESS)
        return 0;

    for (int s = 0; s < TRAIN_SLOTS; s++)
    {
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = g_vk.cmd_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(g_vk.device, &ai, &g_vk.slots[s].cmd) != VK_SUCCESS)
            return 0;

        VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(g_vk.device, &fi, NULL, &g_vk.slots[s].fence) != VK_SUCCESS)
            return 0;
    }
    return 1;
}

// Every kernel shares one layout: VK_MAX_BINDINGS storage buffers and VK_PUSH_BYTES of push constants
static int create_pipelines(void)
{
    VkDescriptorSetLayoutBinding bindings[VK_MAX_BINDINGS];
    for (uint32_t i = 0; i < VK_MAX_BINDINGS; i++)
    {
        memset(&bindings[i], 0, sizeof(bindings[i]));
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = VK_MAX_BINDINGS;
    li.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(g_vk.device, &li, NULL, &g_vk.set_layout) != VK_SUCCESS)
        return 0;

    VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, VK_PUSH_BYTES };
    VkPipelineLayoutCreateInfo pi = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &g_vk.set_layout;
    pi.pushConstantRangeCount = 1;
    pi.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(g_vk.device, &pi, NULL, &g_vk.pipe_layout) != VK_SUCCESS)
        return 0;

    for (int k = 0; k < K_COUNT; k++)
    {
        VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        mi.codeSize = g_kernels[k].bytes;
        mi.pCode = g_kernels[k].code;
        VkShaderModule module;
        if (vkCreateShaderModule(g_vk.device, &mi, NULL, &module) != VK_SUCCESS)
            return 0;

        // gemm.comp: a pipeline per tile shape, with its transposes, tile and workgroup size
        // as specialization constants 0-5
        int shapes = k <= K_GEMM_TN ? GEMM_SHAPES : 1;
        VkResult res = VK_SUCCESS;
        for (int sh = 0; sh < shapes && res == VK_SUCCESS; sh++)
        {
            uint32_t tm = g_gemm_shapes[sh][0], tn = g_gemm_shapes[sh][1];
            uint32_t spec[6] = { g_kernels[k].trans_a, g_kernels[k].trans_b, tm, tn, tn / 4, tm / 4 };
            VkSpecializationMapEntry entries[6];
            for (uint32_t e = 0; e < 6; e++)
            {
                entries[e].constantID = e;
                entries[e].offset = e * (uint32_t)sizeof(uint32_t);
                entries[e].size = sizeof(uint32_t);
            }
            VkSpecializationInfo si = { 6, entries, sizeof(spec), spec };

            VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = module;
            ci.stage.pName = "main";
            ci.stage.pSpecializationInfo = k <= K_GEMM_TN ? &si : NULL;
            ci.layout = g_vk.pipe_layout;
            VkPipeline pipe;
            res = vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
            if (res != VK_SUCCESS)
                break;
            if (k <= K_GEMM_TN)
                g_vk.gemm_pipes[k][sh] = pipe;
            if (sh == 0)
                g_vk.pipelines[k] = pipe;
        }
        vkDestroyShaderModule(g_vk.device, module, NULL);
        if (res != VK_SUCCESS)
        {
            fprintf(stderr, "[Vulkan] Pipeline %d failed (%d)\n", k, (int)res);
            return 0;
        }
    }
    return 1;
}

static void start_profile(int queue_family);

static int vk_init(void)
{
    if (g_vk.initialized)
        return 1;
    if (g_vk.init_failed)
        return 0;
    g_vk.init_failed = 1;

    if (!load_loader())
    {
        fprintf(stderr, "[Vulkan] Vulkan runtime not found\n");
        return 0;
    }
#define VK_LOAD_GLOBAL(f) f = (PFN_##f)vkGetInstanceProcAddr(NULL, #f); if (!f) return 0;
    VK_GLOBAL_FUNCS(VK_LOAD_GLOBAL)
#undef VK_LOAD_GLOBAL

    int queue_family = -1;
    if (!create_instance())
        return 0;
    if (!pick_device(&queue_family))
    {
        vkDestroyInstance(g_vk.instance, NULL);
        return 0;
    }
    choose_memory_model();
    printf("[Vulkan] Device %d: %s (%s)\n", g_vk.device_index, g_vk.device_name,
           g_vk.map_device ? "mapped device memory" : "copies through staging memory");
    if (!create_device(queue_family) || !create_pipelines())
    {
        fprintf(stderr, "[Vulkan] Device setup failed\n");
        return 0;
    }
    start_profile(queue_family);

    g_vk.init_failed = 0;
    g_vk.initialized = 1;
    return 1;
}

//================================================================================================
// Buffers
//================================================================================================

// The first memory type in type_bits with every flag in want and none in avoid, or -1
static int find_memory_type_without(uint32_t type_bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
    for (uint32_t t = 0; t < g_vk.mem.memoryTypeCount; t++)
    {
        VkMemoryPropertyFlags flags = g_vk.mem.memoryTypes[t].propertyFlags;
        if ((type_bits & (1u << t)) && (flags & want) == want && !(flags & avoid))
            return (int)t;
    }
    return -1;
}

static int find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want)
{
    return find_memory_type_without(type_bits, want, 0);
}

static void buf_destroy(VkBuf *b)
{
    if (!b)
        return;
    vkDestroyBuffer(g_vk.device, b->buffer, NULL);
    vkFreeMemory(g_vk.device, b->memory, NULL);
    free(b);
}

// A buffer of at least bytes: host-visible staging memory when host is set, else device memory
static VkBuf *buf_create(size_t bytes, int host)
{
    VkBuf *b = (VkBuf *)calloc(1, sizeof(*b));
    if (!b)
        return NULL;

    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = bytes < 4 ? 4 : (bytes + 3) & ~(size_t)3;     // vkCmdFillBuffer works in words
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vk.device, &bi, NULL, &b->buffer) != VK_SUCCESS)
    {
        free(b);
        return NULL;
    }
    b->size = bi.size;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_vk.device, b->buffer, &req);
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Memory types to try in turn. Device buffers prefer mapped device memory when
    // map_device is set; when that runs out (it may be a window smaller than the device's
    // memory) a buffer goes in device memory the host cannot see, and its copies go through
    // the staging buffer
    int types[4], n_types = 0;
    if (host)
    {
        types[n_types++] = find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        types[n_types++] = find_memory_type(req.memoryTypeBits, visible);
    }
    else
    {
        if (g_vk.map_device)
            types[n_types++] = find_memory_type(req.memoryTypeBits, visible | local);
        types[n_types++] = find_memory_type_without(req.memoryTypeBits, local, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        types[n_types++] = find_memory_type(req.memoryTypeBits, local);
        types[n_types++] = find_memory_type(req.memoryTypeBits, 0);
    }

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    int type = -1;
    for (int i = 0; i < n_types && type < 0; i++)
    {
        if (types[i] < 0)
            continue;
        ai.memoryTypeIndex = (uint32_t)types[i];
        if (vkAllocateMemory(g_vk.device, &ai, NULL, &b->memory) == VK_SUCCESS)
            type = types[i];
    }
    if (type < 0)
    {
        fprintf(stderr, "[Vulkan] Failed to allocate %zu bytes of %s memory\n", bytes, host ? "staging" : "device");
        vkDestroyBuffer(g_vk.device, b->buffer, NULL);
        free(b);
        return NULL;
    }
    vkBindBufferMemory(g_vk.device, b->buffer, b->memory, 0);

    // The host copies in and out of staging buffers, and of device buffers when they are mapped
    int map = (g_vk.mem.memoryTypes[type].propertyFlags & visible) == visible && (host || g_vk.map_device);
    if (map && vkMapMemory(g_vk.device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map) != VK_SUCCESS)
        b->map = NULL;
    if (host && !b->map)
    {
        buf_destroy(b);
        return NULL;
    }
    return b;
}

//-----------------------------------------------------------
// ANN_VK_PROFILE=1: the GPU time of every command, added up by kernel and printed at exit.
// Each command's time runs from the previous command's end to its own, so it includes the
// barrier before it.
//-----------------------------------------------------------
static int profile_order(const void *a, const void *b)
{
    double ta = g_vk.profile_ns[*(const int *)a], tb = g_vk.profile_ns[*(const int *)b];
    return ta < tb ? 1 : (ta > tb ? -1 : 0);
}

static void print_profile(void)
{
    int order[PROFILE_KINDS];
    double total = 0.0;
    for (int k = 0; k < PROFILE_KINDS; k++)
    {
        order[k] = k;
        total += g_vk.profile_ns[k];
    }
    qsort(order, PROFILE_KINDS, sizeof(order[0]), profile_order);
    printf("\n[Vulkan] GPU time by kernel (ANN_VK_PROFILE):\n");
    printf("  %-26s %10s %12s %10s %7s\n", "kernel", "commands", "total ms", "avg us", "share");
    for (int i = 0; i < PROFILE_KINDS; i++)
    {
        int k = order[i];
        if (!g_vk.profile_calls[k])
            continue;
        char name[40];
        if (k < PROFILE_GEMM)
            snprintf(name, sizeof(name), "%s", g_kernel_names[k]);
        else
        {
            int g = k - PROFILE_GEMM, shape = g % GEMM_SHAPES;
            snprintf(name, sizeof(name), "%s %ux%u", g_kernel_names[g / GEMM_SHAPES],
                     g_gemm_shapes[shape][0], g_gemm_shapes[shape][1]);
        }
        printf("  %-26s %10llu %12.2f %10.2f %6.1f%%\n", name, g_vk.profile_calls[k],
               g_vk.profile_ns[k] / 1e6, g_vk.profile_ns[k] / 1e3 / (double)g_vk.profile_calls[k],
               total > 0.0 ? 100.0 * g_vk.profile_ns[k] / total : 0.0);
    }
    printf("  %-26s %10s %12.2f\n", "total", "", total / 1e6);
}

// Turn profiling on when ANN_VK_PROFILE is set and the queue records timestamps
static void start_profile(int queue_family)
{
    const char *env = getenv("ANN_VK_PROFILE");
    if (!env || !*env || strcmp(env, "0") == 0)
        return;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.physical, &count, NULL);
    VkQueueFamilyProperties props[16];
    if (count > 16)
        count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.physical, &count, props);
    if (props[queue_family].timestampValidBits == 0)
    {
        fprintf(stderr, "[Vulkan] ANN_VK_PROFILE: this queue has no timestamps\n");
        return;
    }

    for (int s = 0; s < TRAIN_SLOTS; s++)
    {
        VkQueryPoolCreateInfo qi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = VK_PROFILE_QUERIES;
        if (vkCreateQueryPool(g_vk.device, &qi, NULL, &g_vk.slots[s].queries) != VK_SUCCESS)
            return;
    }
    g_vk.profile = 1;
    atexit(print_profile);
}

// Add a completed command buffer's command times to the totals
static void collect_profile(VkSlot *slot)
{
    static uint64_t stamps[VK_PROFILE_QUERIES];
    uint32_t n = slot->query_count;
    slot->query_count = 0;
    if (vkGetQueryPoolResults(g_vk.device, slot->queries, 0, n, sizeof(stamps), stamps, sizeof(stamps[0]),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return;
    for (uint32_t i = 1; i < n; i++)
    {
        int k = slot->query_kind[i];
        g_vk.profile_ns[k] += (double)(stamps[i] - stamps[i - 1]) * g_vk.ns_per_tick;
        g_vk.profile_calls[k]++;
    }
}

//-----------------------------------------------------------
// Wait for a slot's operation; a training step then adds its batch loss (averaged as on the
// CPU) to g_vk.loss_sum, or flags its failure. Returns 0 if the operation failed.
//-----------------------------------------------------------
static int finish_slot(int s)
{
    VkSlot *slot = &g_vk.slots[s];
    if (!slot->pending)
        return 1;
    VkResult res = vkWaitForFences(g_vk.device, 1, &slot->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_vk.device, 1, &slot->fence);
    slot->pending = 0;
    if (res != VK_SUCCESS)
    {
        fprintf(stderr, "[Vulkan] GPU work failed (%d)\n", (int)res);
        if (slot->loss_src)
            g_vk.train_failed = 1;
        slot->loss_src = NULL;
        return 0;
    }
    if (slot->query_count > 1)
        collect_profile(slot);
    if (slot->loss_src)
    {
        real total = 0.0f;
        for (int b = 0; b < slot->batch_size; b++)
            total += slot->loss_src[b];
        g_vk.loss_sum += total / (real)slot->divisor;
        slot->loss_src = NULL;
    }
    return 1;
}

//-----------------------------------------------------------
// Wait for every training step in flight, oldest first. Anything that reads or replaces GPU
// state the steps use (weights, optimizer state, shared work buffers) calls this first.
//-----------------------------------------------------------
static void finish_all_slots(void)
{
    for (int i = 0; i < TRAIN_SLOTS; i++)
        finish_slot((g_vk.next_slot + i) % TRAIN_SLOTS);
}

// Free a buffer, after the steps in flight, which may use it, complete
static void buf_release(VkBuf *b)
{
    if (!b)
        return;
    finish_all_slots();
    buf_destroy(b);
}

// A work buffer of at least bytes, reallocated when too small
static VkBuf *ensure_buf(VkBuf **b, size_t bytes)
{
    if (*b && (*b)->size >= bytes)
        return *b;
    buf_release(*b);
    *b = buf_create(bytes, 0);
    return *b;
}

static size_t tensor_bytes(const Tensor *t)
{
    return (size_t)t->rows * t->cols * sizeof(float);
}

// The tensor's GPU buffer, (re)allocated to hold its values
static VkBuf *tensor_buf(PTensor t)
{
    if (!t)
        return NULL;
    VkBuf *b = (VkBuf *)t->gpu_buf;
    if (!ensure_buf(&b, tensor_bytes(t)))
    {
        t->gpu_buf = NULL;
        return NULL;
    }
    t->gpu_buf = b;
    return b;
}

static void vk_release_buffer(void *gpu_buf)
{
    buf_release((VkBuf *)gpu_buf);
}

// Room for bytes of uploads and downloads in slot s's next command buffer; call before
// cmd_begin(s), with the slot finished
static int staging_reserve(int s, size_t bytes)
{
    VkSlot *slot = &g_vk.slots[s];
    if (slot->staging && slot->staging->size >= bytes)
        return 1;
    buf_destroy(slot->staging);
    slot->staging = buf_create(bytes < ((size_t)1 << 20) ? ((size_t)1 << 20) : bytes, 1);
    return slot->staging != NULL;
}

// Staging bytes per copy, rounded up for the offset alignment staging_take() keeps
static size_t staged(size_t bytes)
{
    return (bytes + 15) & ~(size_t)15;
}

//================================================================================================
// Command recording
//================================================================================================

// Start recording an operation into slot s, which must be finished (finish_slot); it runs
// when cmd_submit() or cmd_submit_wait() is called
static int cmd_begin(int s)
{
    VkSlot *slot = &g_vk.slots[s];
    for (int i = 0; i < slot->pool_count; i++)
        vkResetDescriptorPool(g_vk.device, slot->pools[i], 0);
    slot->pool_cur = 0;
    slot->staging_used = 0;
    slot->loss_src = NULL;
    g_vk.rec = slot;
    g_vk.record_failed = 0;
    // the first command waits for the operations submitted before this one (submission
    // order is part of a barrier's scope), which share weights and work buffers with it
    g_vk.barrier_pending = 1;

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS)
        return 0;
    slot->query_count = 0;
    if (g_vk.profile)
    {
        vkCmdResetQueryPool(slot->cmd, slot->queries, 0, VK_PROFILE_QUERIES);
        vkCmdWriteTimestamp(slot->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, slot->queries, 0);
        slot->query_count = 1;
    }
    return 1;
}

// ANN_VK_PROFILE: stamp the time the command just recorded (a Kernel, or K_TRANSFER) ends
static void profile_mark(int kind)
{
    VkSlot *slot = g_vk.rec;
    if (!g_vk.profile || slot->query_count >= VK_PROFILE_QUERIES)
        return;
    vkCmdWriteTimestamp(slot->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot->queries, slot->query_count);
    slot->query_kind[slot->query_count++] = (uint8_t)kind;
}

// Make earlier commands' writes visible to the next command
static void cmd_barrier(void)
{
    if (!g_vk.barrier_pending)
        return;
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(g_vk.rec->cmd, stages, stages, 0, 1, &mb, 0, NULL, 0, NULL);
    g_vk.barrier_pending = 0;
}

// Submit the recorded operation without waiting; 1 if it was submitted
static int cmd_submit(void)
{
    VkSlot *slot = g_vk.rec;
    g_vk.rec = NULL;

    // results the host reads (downloads, mapped buffers) must be visible to it
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(slot->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(slot->cmd) != VK_SUCCESS || g_vk.record_failed)
        return 0;

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot->cmd;
    VkResult res = vkQueueSubmit(g_vk.queue, 1, &si, slot->fence);
    if (res != VK_SUCCESS)
    {
        fprintf(stderr, "[Vulkan] Submit failed (%d)\n", (int)res);
        return 0;
    }
    slot->pending = 1;
    return 1;
}

// Submit the recorded operation and wait for it; 1 if it completed
static int cmd_submit_wait(void)
{
    int s = (int)(g_vk.rec - g_vk.slots);
    return cmd_submit() && finish_slot(s);
}

// The slot an operation other than a training step uses, once the steps in flight complete:
// it reads or replaces state they use
static int sync_slot(void)
{
    finish_all_slots();
    return g_vk.next_slot;
}

// Take bytes of the recording slot's staging buffer; returns the offset
static VkDeviceSize staging_take(size_t bytes)
{
    VkSlot *slot = g_vk.rec;
    VkDeviceSize off = slot->staging_used;
    if (!slot->staging || off + bytes > slot->staging->size)
    {
        g_vk.record_failed = 1;
        return 0;
    }
    slot->staging_used += staged(bytes);
    return off;
}

// Record a copy of host memory into dst (or copy it now when dst is mapped, so dst must not
// be in use by a step in flight)
static void cmd_upload(VkBuf *dst, const void *src, size_t bytes)
{
    if (bytes == 0)
        return;
    if (dst->map)
    {
        memcpy(dst->map, src, bytes);
        return;
    }
    VkDeviceSize off = staging_take(bytes);
    if (g_vk.record_failed)
        return;
    memcpy((char *)g_vk.rec->staging->map + off, src, bytes);
    cmd_barrier();
    VkBufferCopy region = { off, 0, bytes };
    vkCmdCopyBuffer(g_vk.rec->cmd, g_vk.rec->staging->buffer, dst->buffer, 1, &region);
    profile_mark(K_TRANSFER);
    g_vk.barrier_pending = 1;
}

// Record a copy of src's first bytes to where the host reads them once the operation
// completes; returns that address
static const void *cmd_download(VkBuf *src, size_t bytes)
{
    if (src->map)
        return src->map;
    VkDeviceSize off = staging_take(bytes);
    if (g_vk.record_failed)
        return NULL;
    cmd_barrier();
    VkBufferCopy region = { 0, off, bytes };
    vkCmdCopyBuffer(g_vk.rec->cmd, src->buffer, g_vk.rec->staging->buffer, 1, &region);
    profile_mark(K_TRANSFER);
    g_vk.barrier_pending = 1;
    return (const char *)g_vk.rec->staging->map + off;
}

static void cmd_zero(VkBuf *b)
{
    cmd_barrier();
    vkCmdFillBuffer(g_vk.rec->cmd, b->buffer, 0, VK_WHOLE_SIZE, 0);
    profile_mark(K_TRANSFER);
    g_vk.barrier_pending = 1;
}

// Record a copy of src's first bytes to dst
static void cmd_copy(VkBuf *src, VkBuf *dst, size_t bytes)
{
    if (bytes == 0)
        return;
    cmd_barrier();
    VkBufferCopy region = { 0, 0, bytes };
    vkCmdCopyBuffer(g_vk.rec->cmd, src->buffer, dst->buffer, 1, &region);
    profile_mark(K_TRANSFER);
    g_vk.barrier_pending = 1;
}

// A descriptor set for one dispatch, from the recording slot's pools (reset by cmd_begin())
static VkDescriptorSet alloc_set(void)
{
    VkSlot *slot = g_vk.rec;
    while (slot->pool_cur < VK_MAX_POOLS)
    {
        if (slot->pool_cur == slot->pool_count)
        {
            VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_POOL_SETS * VK_MAX_BINDINGS };
            VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pi.maxSets = VK_POOL_SETS;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &size;
            if (vkCreateDescriptorPool(g_vk.device, &pi, NULL, &slot->pools[slot->pool_count]) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            slot->pool_count++;
        }
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = slot->pools[slot->pool_cur];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_vk.set_layout;
        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) == VK_SUCCESS)
            return set;
        slot->pool_cur++;
    }
    return VK_NULL_HANDLE;
}

// Workgroups for n items at local invocations each; the shaders loop over any excess
static uint32_t groups_for(size_t n, uint32_t local)
{
    size_t groups = (n + local - 1) / local;
    if (groups < 1)
        groups = 1;
    return groups > g_vk.max_groups[0] ? g_vk.max_groups[0] : (uint32_t)groups;
}

//-----------------------------------------------------------
// Record a kernel: bufs bind to 0..n_bufs-1 (the rest repeat bufs[0]), params are its
// push constants
//-----------------------------------------------------------
static void cmd_dispatch_pipe(VkPipeline pipe, int profile_kind, VkBuf *const *bufs, int n_bufs, const void *params,
                              uint32_t params_size, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
    if (g_vk.record_failed)
        return;
    VkDescriptorSet set = alloc_set();
    if (!set || groups_x > g_vk.max_groups[0] || groups_y > g_vk.max_groups[1] || groups_z > g_vk.max_groups[2])
    {
        g_vk.record_failed = 1;
        return;
    }

    VkDescriptorBufferInfo info[VK_MAX_BINDINGS];
    VkWriteDescriptorSet writes[VK_MAX_BINDINGS];
    for (int i = 0; i < VK_MAX_BINDINGS; i++)
    {
        VkBuf *b = bufs[i < n_bufs ? i : 0];
        info[i].buffer = b->buffer;
        info[i].offset = 0;
        info[i].range = VK_WHOLE_SIZE;
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(g_vk.device, VK_MAX_BINDINGS, writes, 0, NULL);

    cmd_barrier();
    vkCmdBindPipeline(g_vk.rec->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(g_vk.rec->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(g_vk.rec->cmd, g_vk.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    vkCmdDispatch(g_vk.rec->cmd, groups_x, groups_y, groups_z);
    profile_mark(profile_kind);
    g_vk.barrier_pending = 1;
}

// The same with kernel k's pipeline
static void cmd_dispatch_3d(Kernel k, VkBuf *const *bufs, int n_bufs, const void *params, uint32_t params_size,
                            uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
    cmd_dispatch_pipe(g_vk.pipelines[k], (int)k, bufs, n_bufs, params, params_size, groups_x, groups_y, groups_z);
}

static void cmd_dispatch(Kernel k, VkBuf *const *bufs, int n_bufs, const void *params, uint32_t params_size,
                         uint32_t groups_x, uint32_t groups_y)
{
    cmd_dispatch_3d(k, bufs, n_bufs, params, params_size, groups_x, groups_y, 1);
}

//================================================================================================
// Kernels
//================================================================================================

// Add up n-element partial sums from slices into c (the first n_first) and c2 (the rest),
// each becoming the sum + beta * its old value (see sum_partials.comp): one invocation per
// element when there are few slices, else a workgroup per element
static void k_sum_partials(VkBuf *partial, VkBuf *c, VkBuf *c2, uint32_t n, uint32_t slices, uint32_t n_first,
                           float beta)
{
    uint32_t per_invocation = slices <= SUM_PER_INVOCATION;
    struct { uint32_t n, slices, n_first; float beta; uint32_t per_invocation; } p = {
        n, slices, n_first, beta, per_invocation
    };
    VkBuf *bufs[3] = { partial, c, c2 };
    cmd_dispatch(K_SUM_PARTIALS, bufs, 3, &p, sizeof(p), groups_for(n, per_invocation ? 64 : 1), 1);
}

// The tile shape (gemm.comp's TM x TN) for a GEMM with an M x N result: square, unless a
// thin shape pads the result at least GEMM_THIN_GAIN times less. On the Radeon HD 7700 the
// thin shapes ran slower per useful result, so only a matrix that fills a sliver of the square
// tiles (a 16-filter conv's forward pass: 16 of 64 columns) gained from them.
static int gemm_shape(uint32_t M, uint32_t N)
{
    // ANN_VK_GEMM_SHAPE=<0..4> forces one shape of g_gemm_shapes, for tuning on other GPUs
    static int forced = -2;
    if (forced == -2)
    {
        const char *env = getenv("ANN_VK_GEMM_SHAPE");
        forced = env && *env ? atoi(env) : -1;
        if (forced >= GEMM_SHAPES)
            forced = -1;
    }
    if (forced >= 0)
        return forced;
    int best = 0;
    uint64_t best_work = UINT64_MAX;
    uint32_t best_tiles = 0;
    for (int s = 1; s < GEMM_SHAPES; s++)
    {
        uint32_t tm = g_gemm_shapes[s][0], tn = g_gemm_shapes[s][1];
        uint32_t tiles = ((M + tm - 1) / tm) * ((N + tn - 1) / tn);
        uint64_t work = (uint64_t)tiles * tm * tn;
        if (work < best_work || (work == best_work && tiles > best_tiles))
        {
            best = s;
            best_work = work;
            best_tiles = tiles;
        }
    }
    uint64_t square = (uint64_t)((M + 63) / 64) * ((N + 63) / 64) * 64 * 64;
    return square >= GEMM_THIN_GAIN * best_work ? best : 0;
}

//-----------------------------------------------------------
// Row-major C[M x N] = alpha * op(A) * op(B) + beta * C (see gemm.comp), in the tile shape
// that wastes the least work. A GEMM with few output tiles and a deep K (a forward pass at a
// small batch, a conv weight gradient summing over every output position of the batch) would
// leave most of the GPU idle, so K is split over about SPLIT_GROUPS workgroups, each summing
// its slice into its own block of g_vk.split, and sum_partials.comp adds the blocks into C.
//-----------------------------------------------------------
static void k_gemm(int trans_a, int trans_b, uint32_t M, uint32_t N, uint32_t K,
                   float alpha, VkBuf *A, VkBuf *B, float beta, VkBuf *C)
{
    int shape = gemm_shape(M, N);
    uint32_t tm = g_gemm_shapes[shape][0], tn = g_gemm_shapes[shape][1];
    uint32_t col_tiles = (N + tn - 1) / tn, row_tiles = (M + tm - 1) / tm;
    uint32_t tiles = col_tiles * row_tiles;

    uint32_t slices = tiles >= SPLIT_GROUPS ? 1 : SPLIT_GROUPS / tiles;
    if (slices > K / SPLIT_MIN_DEPTH)
        slices = K / SPLIT_MIN_DEPTH;
    if (slices > SPLIT_FLOATS / ((uint64_t)M * N))
        slices = (uint32_t)(SPLIT_FLOATS / ((uint64_t)M * N));
    if (slices > g_vk.max_groups[2])
        slices = g_vk.max_groups[2];
    if (slices < 1)
        slices = 1;
    // slices a whole number of gemm.comp's 16-deep steps
    uint32_t k_slice = (K + slices - 1) / slices;
    k_slice = (k_slice + 15) / 16 * 16;
    slices = (K + k_slice - 1) / k_slice;

    // gemm.comp loops over row tiles beyond the device's workgroup limit
    if (row_tiles > g_vk.max_groups[1])
        row_tiles = g_vk.max_groups[1];
    int kind = trans_a ? 2 : (trans_b ? 1 : 0);     // K_GEMM_NN, K_GEMM_NT or K_GEMM_TN
    VkPipeline pipe = g_vk.gemm_pipes[kind][shape];
    int profile_kind = PROFILE_GEMM + kind * GEMM_SHAPES + shape;

    struct { uint32_t M, N, K; float alpha, beta; uint32_t k_slice; } p = { M, N, K, alpha, beta, k_slice };
    if (slices == 1)
    {
        VkBuf *bufs[3] = { A, B, C };
        cmd_dispatch_pipe(pipe, profile_kind, bufs, 3, &p, sizeof(p), col_tiles, row_tiles, 1);
        return;
    }
    p.beta = 0.0f;
    VkBuf *bufs[3] = { A, B, g_vk.split };
    cmd_dispatch_pipe(pipe, profile_kind, bufs, 3, &p, sizeof(p), col_tiles, row_tiles, slices);

    k_sum_partials(g_vk.split, C, C, M * N, slices, M * N, beta);
}

// dl_dz *= the activation's derivative for a layer, reading A, or Z for GELU, as the CPU
// does; softmax (MSE through a softmax output) applies its Jacobian
static void k_activation_derivative(PLayer l, int rows)
{
    VkBuf *dl_dz = (VkBuf *)l->t_batch_dl_dz->gpu_buf;
    VkBuf *A = (VkBuf *)l->t_batch_values->gpu_buf;

    if (l->activation == ACTIVATION_SOFTMAX)
    {
        struct { uint32_t rows, cols; } p = { (uint32_t)rows, (uint32_t)l->node_count };
        VkBuf *bufs[2] = { dl_dz, A };
        cmd_dispatch(K_SOFTMAX_JACOBIAN, bufs, 2, &p, sizeof(p), groups_for((size_t)rows, 64), 1);
        return;
    }
    if (l->activation == ACTIVATION_NULL)
        return;

    size_t count = (size_t)rows * l->node_count;
    struct { uint32_t n, act; } p = { (uint32_t)count, (uint32_t)l->activation };
    VkBuf *bufs[2] = { dl_dz, l->activation == ACTIVATION_GELU ? (VkBuf *)l->t_batch_z->gpu_buf : A };
    cmd_dispatch(K_ACT_DERIV, bufs, 2, &p, sizeof(p), groups_for(count, 256), 1);
}

// Y = activate(Y + b), or activate(Y) when b is NULL, first saving the values before
// activation in Z when Z is given; softmax passes through (k_softmax applies it)
static void k_bias_act(VkBuf *Y, VkBuf *b, VkBuf *Z, size_t count, uint32_t cols, Activation_type act)
{
    struct { uint32_t n, cols, act, save_z, has_bias; } p = {
        (uint32_t)count, cols, (uint32_t)act, Z != NULL, b != NULL
    };
    VkBuf *bufs[3] = { Y, b ? b : Y, Z ? Z : Y };
    cmd_dispatch(K_BIAS_ACT, bufs, 3, &p, sizeof(p), groups_for(count, 256), 1);
}

//-----------------------------------------------------------
// A conv, pool or upsample layer's geometry, laid out like WindowDims in
// shaders/vk/common.glsl
//-----------------------------------------------------------
typedef struct {
    uint32_t C, H, W, F, OH, OW, K, S, P, batch;
} WindowDims;

static WindowDims window_dims(PNetwork pnet, int layer, int batch_size)
{
    PLayer l = &pnet->layers[layer];
    int size = 0, stride = 1, padding = 0;
    ann_get_layer_settings(pnet, layer, &size, &stride, &padding);

    WindowDims d;
    d.C = (uint32_t)l->in_shape.c; d.H = (uint32_t)l->in_shape.h; d.W = (uint32_t)l->in_shape.w;
    d.F = (uint32_t)l->shape.c;    d.OH = (uint32_t)l->shape.h;   d.OW = (uint32_t)l->shape.w;
    d.K = (uint32_t)size; d.S = (uint32_t)stride; d.P = (uint32_t)padding;
    d.batch = (uint32_t)batch_size;
    return d;
}

// The layer's work buffer for slot (see SCRATCH_COLS), allocated by ensure_layer_scratch()
static VkBuf *scratch_of(int layer, int slot)
{
    return g_vk.scratch[layer * SCRATCH_SLOTS + slot];
}

// The layer's work buffer for slot, of at least bytes; call before cmd_begin()
static VkBuf *layer_scratch(int layer, int slot, size_t bytes)
{
    if (layer >= g_vk.scratch_layers)
    {
        int layers = layer + 1;
        VkBuf **grown = (VkBuf **)realloc(g_vk.scratch, (size_t)layers * SCRATCH_SLOTS * sizeof(*grown));
        if (!grown)
            return NULL;
        memset(grown + (size_t)g_vk.scratch_layers * SCRATCH_SLOTS, 0,
               (size_t)(layers - g_vk.scratch_layers) * SCRATCH_SLOTS * sizeof(*grown));
        g_vk.scratch = grown;
        g_vk.scratch_layers = layers;
    }
    return ensure_buf(&g_vk.scratch[layer * SCRATCH_SLOTS + slot], bytes);
}

static void free_scratch(void)
{
    buf_destroy(g_vk.split);
    g_vk.split = NULL;
    for (int i = 0; i < g_vk.scratch_layers * SCRATCH_SLOTS; i++)
        buf_destroy(g_vk.scratch[i]);
    free(g_vk.scratch);
    g_vk.scratch = NULL;
    g_vk.scratch_layers = 0;
}

//-----------------------------------------------------------
// Conv2D layers with at most this many filters convolve directly, with no im2col columns:
// their GEMMs would have F outputs per row, a sliver of gemm.comp's 64-wide tiles, and the
// columns are C*K*K times the input's size to write and read back. On an AMD Radeon HD 7700,
// fashion_cnn's 8-filter layer cut its GPU time 14% (batch 32) to 26% (batch 256) this way;
// at 16 filters the direct input gradient, which reads F values per window position, lost to
// the GEMM and col2im. (Metal, whose MPS GEMM handles thin matrices well, stops at 4.)
// ANN_VK_CONV_DIRECT=<filters> overrides it, for tuning on other GPUs; at most 16.
//-----------------------------------------------------------
#define CONV_DIRECT_FILTERS 8
// Rows each im2col invocation takes: they share its one-time index math. On the Radeon HD
// 7700, 16 beat 8 and 4, and beat keeping at least 256 workgroups for small layers.
#define IM2COL_ROWS         16
// The direct weight gradient aims for about this many workgroups
#define CONV_DIRECT_GROUPS  256

static int conv_direct(const WindowDims *d)
{
    static int limit = -1;
    if (limit < 0)
    {
        const char *env = getenv("ANN_VK_CONV_DIRECT");
        limit = env && *env ? atoi(env) : CONV_DIRECT_FILTERS;
    }
    // conv_direct_weight_grad.comp keeps a sum per filter, for up to 16
    return (int)d->F <= (limit < 16 ? limit : 16);
}

// Lanes (columns) per workgroup for a direct conv's weight gradient
// (conv_direct_weight_grad.comp): the columns, the bias one included, rounded up to a power
// of two, at most 128 so a workgroup has at least two rows at a time
static uint32_t direct_lanes(const WindowDims *d)
{
    uint32_t columns = d->C * d->K * d->K + 1, px = 1;
    while (px < columns && px < 128)
        px *= 2;
    return px;
}

// Chunks of rows for a direct conv's weight gradient, which has a workgroup per block of
// lanes and chunk
static uint32_t direct_chunks(const WindowDims *d, uint32_t rows)
{
    uint32_t px = direct_lanes(d), ys = 256 / px;
    uint32_t blocks = (d->C * d->K * d->K + 1 + px - 1) / px;
    uint32_t chunks = blocks >= CONV_DIRECT_GROUPS ? 1 : CONV_DIRECT_GROUPS / blocks;
    if (chunks > rows / ys)
        chunks = rows / ys;
    if (chunks > g_vk.max_groups[1])
        chunks = g_vk.max_groups[1];
    return chunks < 1 ? 1 : chunks;
}

//-----------------------------------------------------------
// Allocate the work buffers a batch of rows needs: split-K partial sums, conv columns and
// GEMM rows (and, for training, column gradients, which layer 1 never needs; a direct conv
// needs only gradient rows and weight gradient partial sums), and max pool argmax
//-----------------------------------------------------------
static int ensure_layer_scratch(PNetwork pnet, int rows, int training)
{
    if (!ensure_buf(&g_vk.split, SPLIT_FLOATS * sizeof(float)))
        return 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (l->kind == LAYER_KIND_CONV2D)
        {
            WindowDims d = window_dims(pnet, layer, rows);
            size_t out_rows = (size_t)rows * d.OH * d.OW, patch = (size_t)d.C * d.K * d.K;
            if (conv_direct(&d))
            {
                if (training &&
                    (!layer_scratch(layer, SCRATCH_ROWS, out_rows * d.F * sizeof(float)) ||
                     !layer_scratch(layer, SCRATCH_PARTIAL, (size_t)direct_chunks(&d, (uint32_t)out_rows) *
                                                            d.F * (patch + 1) * sizeof(float))))
                    return 0;
                continue;
            }
            if (!layer_scratch(layer, SCRATCH_COLS, out_rows * patch * sizeof(float)) ||
                !layer_scratch(layer, SCRATCH_ROWS, out_rows * d.F * sizeof(float)) ||
                (training && layer > 1 && !layer_scratch(layer, SCRATCH_DCOLS, out_rows * patch * sizeof(float))))
                return 0;
        }
        else if (l->kind == LAYER_KIND_MAXPOOL2D)
        {
            if (!layer_scratch(layer, SCRATCH_COLS, (size_t)rows * l->node_count * sizeof(uint32_t)))
                return 0;
        }
        else if (l->kind == LAYER_KIND_BATCHNORM && training)
        {
            // x_hat, and the per-channel statistics (see batchnorm.comp)
            if (!layer_scratch(layer, SCRATCH_COLS, (size_t)rows * l->node_count * sizeof(float)) ||
                !layer_scratch(layer, SCRATCH_PARTIAL, (size_t)4 * l->shape.c * sizeof(float)))
                return 0;
        }
    }
    return 1;
}

// GPU memory one sample needs in a forward pass: the ping-pong buffers, and conv columns
// and GEMM rows, and max pool argmax
static size_t forward_bytes_per_sample(const PNetwork pnet)
{
    size_t max_nodes = 0, floats = 0;
    for (int layer = 0; layer < pnet->layer_count; layer++)
    {
        const Layer *l = &pnet->layers[layer];
        if ((size_t)l->node_count > max_nodes)
            max_nodes = (size_t)l->node_count;
        if (l->kind == LAYER_KIND_CONV2D && l->shape.c > CONV_DIRECT_FILTERS)
            floats += (size_t)l->shape.h * l->shape.w * (l->params[0].value->cols + l->shape.c);
        else if (l->kind == LAYER_KIND_MAXPOOL2D)
            floats += (size_t)l->node_count;
    }
    return (floats + 2 * max_nodes) * sizeof(float);
}

//-----------------------------------------------------------
// Conv2D forward for a batch, as on the CPU but with one GEMM for the whole batch:
// rows [batch*OH*OW x F] = im2col(X) * W^T, then Y = activate(rows + b) rearranged to
// [batch][F][OH*OW] (saving the values before activation in Z when given); or directly
// (conv_direct)
//-----------------------------------------------------------
static void encode_conv_forward(PNetwork pnet, int layer, int batch_size, VkBuf *X, VkBuf *Y, VkBuf *Z)
{
    PLayer l = &pnet->layers[layer];
    WindowDims d = window_dims(pnet, layer, batch_size);
    uint32_t out_rows = (uint32_t)batch_size * d.OH * d.OW, patch = d.C * d.K * d.K;
    struct { WindowDims d; uint32_t act, save_z; } p = { d, (uint32_t)l->activation, Z != NULL };

    if (conv_direct(&d))
    {
        VkBuf *bufs[5] = { X, (VkBuf *)l->params[0].value->gpu_buf, (VkBuf *)l->params[1].value->gpu_buf,
                           Y, Z ? Z : Y };
        cmd_dispatch(K_CONV_DIRECT_FORWARD, bufs, 5, &p, sizeof(p),
                     groups_for((size_t)l->node_count * batch_size, 256), 1);
        return;
    }

    VkBuf *cols = scratch_of(layer, SCRATCH_COLS), *rows = scratch_of(layer, SCRATCH_ROWS);
    VkBuf *im2col_bufs[2] = { X, cols };
    // 16 x 16 workgroups over (columns, rows); each invocation takes about
    // IM2COL_ROWS rows (see conv_im2col.comp)
    uint32_t col_groups = (patch + 15) / 16;
    uint32_t row_groups = (out_rows + 16 * IM2COL_ROWS - 1) / (16 * IM2COL_ROWS);
    if (row_groups > g_vk.max_groups[1])
        row_groups = g_vk.max_groups[1];
    cmd_dispatch(K_CONV_IM2COL, im2col_bufs, 2, &d, sizeof(d), col_groups, row_groups);
    k_gemm(0, 1, out_rows, d.F, patch, 1.0f, cols, (VkBuf *)l->params[0].value->gpu_buf, 0.0f, rows);

    VkBuf *out_bufs[4] = { rows, (VkBuf *)l->params[1].value->gpu_buf, Y, Z ? Z : Y };
    cmd_dispatch(K_CONV_OUTPUT, out_bufs, 4, &p, sizeof(p), groups_for((size_t)l->node_count * batch_size, 256), 1);
}

//-----------------------------------------------------------
// Conv2D backward for a batch: dZ_rows = dZ * f'(z) for the activation act
// (ACTIVATION_NULL when dZ already has it), dW += dZ_rows^T * cols, db += column sums of
// dZ_rows, and, when dX is given, dX = col2im(dZ_rows * W). cols is kept from the forward pass.
// A direct conv has none: its weight gradient reads the layer's input X, with the bias as one
// more column, and dX is computed directly.
//-----------------------------------------------------------
static void encode_conv_backward(PNetwork pnet, int layer, int batch_size, VkBuf *X, VkBuf *dZ, VkBuf *dX,
                                 Activation_type act)
{
    PLayer l = &pnet->layers[layer];
    WindowDims d = window_dims(pnet, layer, batch_size);
    uint32_t out_rows = (uint32_t)batch_size * d.OH * d.OW, patch = d.C * d.K * d.K;
    VkBuf *W = (VkBuf *)l->params[0].value->gpu_buf;
    VkBuf *dW = (VkBuf *)l->params[0].grad->gpu_buf, *db = (VkBuf *)l->params[1].grad->gpu_buf;
    VkBuf *dz_rows = scratch_of(layer, SCRATCH_ROWS);

    // the derivative reads A, or Z for GELU (the only one the forward pass saves)
    struct { WindowDims d; uint32_t act; } p = { d, (uint32_t)act };
    VkBuf *row_bufs[3] = { dZ, dz_rows, (VkBuf *)(act == ACTIVATION_GELU ? l->t_batch_z : l->t_batch_values)->gpu_buf };
    cmd_dispatch(K_CONV_GRAD_ROWS, row_bufs, 3, &p, sizeof(p), groups_for((size_t)l->node_count * batch_size, 256), 1);

    if (conv_direct(&d))
    {
        // dW and db: a workgroup per block of columns (the bias is column patch) and row
        // chunk, whose partial sums are then added up when there are chunks
        uint32_t px = direct_lanes(&d), chunks = direct_chunks(&d, out_rows);
        uint32_t chunk_rows = (out_rows + chunks - 1) / chunks;
        chunks = (out_rows + chunk_rows - 1) / chunk_rows;
        VkBuf *partial = chunks > 1 ? scratch_of(layer, SCRATCH_PARTIAL) : dW;
        struct { WindowDims d; uint32_t chunk_rows, px; } wp = { d, chunk_rows, px };
        VkBuf *grad_bufs[4] = { dz_rows, X, partial, db };
        cmd_dispatch(K_CONV_DIRECT_WEIGHT_GRAD, grad_bufs, 4, &wp, sizeof(wp), (patch + 1 + px - 1) / px, chunks);
        if (chunks > 1)
        {
            k_sum_partials(partial, dW, db, d.F * (patch + 1), chunks, d.F * patch, 1.0f);
        }

        if (dX)
        {
            VkBuf *dx_bufs[3] = { dz_rows, W, dX };
            cmd_dispatch(K_CONV_DIRECT_INPUT_GRAD, dx_bufs, 3, &d, sizeof(d),
                         groups_for((size_t)batch_size * d.C * d.H * d.W, 256), 1);
        }
        return;
    }

    VkBuf *cols = scratch_of(layer, SCRATCH_COLS);

    // dW += dz_rows^T * cols  ([F x out_rows] * [out_rows x patch]): few outputs, each a sum
    // over every output position of the batch, so split over them
    k_gemm(1, 0, d.F, patch, out_rows, 1.0f, dz_rows, cols, 1.0f, dW);

    // db += column sums of dz_rows
    {
        struct { uint32_t rows, cols; } bp = { out_rows, d.F };
        VkBuf *bufs[2] = { dz_rows, db };
        cmd_dispatch(K_BIAS_GRAD_SUM, bufs, 2, &bp, sizeof(bp), groups_for(d.F, 1), 1);
    }

    if (!dX)
        return;
    // dcols = dz_rows * W  ([out_rows x F] * [F x patch]), then gathered back to the input
    VkBuf *dcols = scratch_of(layer, SCRATCH_DCOLS);
    k_gemm(0, 0, out_rows, patch, d.F, 1.0f, dz_rows, W, 0.0f, dcols);
    VkBuf *col2im_bufs[2] = { dcols, dX };
    cmd_dispatch(K_CONV_COL2IM, col2im_bufs, 2, &d, sizeof(d), groups_for((size_t)batch_size * d.C * d.H * d.W, 256), 1);
}

// MaxPool2D forward; each output's argmax goes to the layer's scratch
static void encode_maxpool_forward(PNetwork pnet, int layer, int batch_size, VkBuf *X, VkBuf *Y)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    VkBuf *bufs[3] = { X, Y, scratch_of(layer, SCRATCH_COLS) };
    cmd_dispatch(K_MAXPOOL_FORWARD, bufs, 3, &d, sizeof(d), groups_for((size_t)batch_size * d.C * d.OH * d.OW, 256), 1);
}

static void encode_maxpool_backward(PNetwork pnet, int layer, int batch_size, VkBuf *dZ, VkBuf *dX)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    // when the windows tile the input, each output writes its window; else each input
    // gathers from the windows covering it
    struct { WindowDims d; uint32_t tiled; } p = { d, d.K == d.S };
    VkBuf *bufs[3] = { dZ, dX, scratch_of(layer, SCRATCH_COLS) };
    size_t count = (size_t)batch_size * d.C * (p.tiled ? d.OH * d.OW : d.H * d.W);
    cmd_dispatch(K_MAXPOOL_BACKWARD, bufs, 3, &p, sizeof(p), groups_for(count, 256), 1);
}

// Upsample2D forward (src = X, dst = Y) or backward (src = dZ, dst = dX)
static void encode_upsample(PNetwork pnet, int layer, int batch_size, VkBuf *src, VkBuf *dst, int backward)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    struct { WindowDims d; uint32_t backward; } p = { d, (uint32_t)backward };
    VkBuf *bufs[2] = { src, dst };
    size_t count = (size_t)batch_size * d.C * (backward ? d.H * d.W : d.OH * d.OW);
    cmd_dispatch(K_UPSAMPLE, bufs, 2, &p, sizeof(p), groups_for(count, 256), 1);
}

//-----------------------------------------------------------
// BatchNorm (batchnorm.comp): params are gamma, beta, running mean, running variance.
// Training keeps x_hat in SCRATCH_COLS and the statistics in SCRATCH_PARTIAL.
//-----------------------------------------------------------
typedef struct {
    uint32_t C, area, batch, mode;
    float momentum, eps;
} BnParams;

static VkBuf *bn_param(PLayer l, int p)
{
    return (VkBuf *)l->params[p].value->gpu_buf;
}

static void k_batchnorm(PNetwork pnet, int layer, int rows, uint32_t mode, VkBuf **bufs)
{
    PLayer l = &pnet->layers[layer];
    BnParams p = { (uint32_t)l->shape.c, (uint32_t)(l->shape.h * l->shape.w), (uint32_t)rows, mode,
                   0.1f, (float)ANN_BATCHNORM_EPSILON };
    // the statistics and gradient modes run a workgroup per channel
    size_t groups = mode == 0 || mode == 3 ? (size_t)p.C * 256 : (size_t)rows * l->node_count;
    cmd_dispatch(K_BATCHNORM, bufs, 8, &p, sizeof(p), groups_for(groups, 256), 1);
}

static void encode_batchnorm_forward(PNetwork pnet, int layer, int rows, VkBuf *X, VkBuf *Y, int training)
{
    PLayer l = &pnet->layers[layer];
    if (!training)
    {
        VkBuf *bufs[8] = { X, Y, X, X, bn_param(l, 0), bn_param(l, 1), bn_param(l, 2), bn_param(l, 3) };
        k_batchnorm(pnet, layer, rows, 2, bufs);
        return;
    }
    VkBuf *x_hat = scratch_of(layer, SCRATCH_COLS), *stats = scratch_of(layer, SCRATCH_PARTIAL);
    VkBuf *sbufs[8] = { X, Y, x_hat, stats, bn_param(l, 0), bn_param(l, 1), bn_param(l, 2), bn_param(l, 3) };
    k_batchnorm(pnet, layer, rows, 0, sbufs);
    k_batchnorm(pnet, layer, rows, 1, sbufs);
}

// gamma and beta gradients, then dX unless it is NULL (layer 1)
static void encode_batchnorm_backward(PNetwork pnet, int layer, int rows, VkBuf *dZ, VkBuf *dX)
{
    PLayer l = &pnet->layers[layer];
    VkBuf *x_hat = scratch_of(layer, SCRATCH_COLS), *stats = scratch_of(layer, SCRATCH_PARTIAL);
    VkBuf *gbufs[8] = { dZ, dZ, x_hat, stats, (VkBuf *)l->params[0].grad->gpu_buf,
                        (VkBuf *)l->params[1].grad->gpu_buf, dZ, dZ };
    k_batchnorm(pnet, layer, rows, 3, gbufs);
    if (dX)
    {
        VkBuf *bufs[8] = { dZ, dX, x_hat, stats, bn_param(l, 0), dZ, dZ, dZ };
        k_batchnorm(pnet, layer, rows, 4, bufs);
    }
}

// GlobalAvgPool2D forward (src = X, dst = Y) or backward (src = dZ, dst = dX)
static void encode_global_avgpool(PNetwork pnet, int layer, int batch_size, VkBuf *src, VkBuf *dst, int backward)
{
    WindowDims d = window_dims(pnet, layer, batch_size);
    struct { WindowDims d; uint32_t backward; } p = { d, (uint32_t)backward };
    VkBuf *bufs[2] = { src, dst };
    size_t count = (size_t)batch_size * d.C * (backward ? d.H * d.W : 1);
    cmd_dispatch(K_GLOBAL_AVGPOOL, bufs, 2, &p, sizeof(p), groups_for(count, 256), 1);
}

//-----------------------------------------------------------
// Encode a layer's forward pass for a batch: the kind's linear part (dense:
// Y = X * W^T + b), then (when Z is given) save the values before activation (GELU only
// needs them), apply the activation, and apply dropout when a mask is given
//-----------------------------------------------------------
static void encode_layer_forward(PNetwork pnet, int layer, int rows, VkBuf *X, VkBuf *Y, VkBuf *Z, VkBuf *mask,
                                 int training)
{
    PLayer dst = &pnet->layers[layer];
    uint32_t out = (uint32_t)dst->node_count;
    size_t count = (size_t)rows * out;

    // Only GELU's derivative needs the values before activation (ReLU's and
    // leaky ReLU's read the sign from the output), so Z is saved only for it
    if (dst->activation != ACTIVATION_GELU)
        Z = NULL;

    // Dense and conv layers apply their activation (other than softmax) as they add the bias
    int activated = 0;
    switch (dst->kind)
    {
    case LAYER_KIND_DENSE:
        k_gemm(0, 1, (uint32_t)rows, out, (uint32_t)pnet->layers[layer - 1].node_count, 1.0f, X,
               (VkBuf *)dst->t_weights->gpu_buf, 0.0f, Y);
        k_bias_act(Y, (VkBuf *)dst->t_bias->gpu_buf, Z, count, out, dst->activation);
        activated = 1;
        break;
    case LAYER_KIND_CONV2D:
        encode_conv_forward(pnet, layer, rows, X, Y, Z);
        activated = 1;
        break;
    case LAYER_KIND_MAXPOOL2D:
        encode_maxpool_forward(pnet, layer, rows, X, Y);
        break;
    case LAYER_KIND_UPSAMPLE2D:
        encode_upsample(pnet, layer, rows, X, Y, 0);
        break;
    case LAYER_KIND_GLOBAL_AVGPOOL2D:
        encode_global_avgpool(pnet, layer, rows, X, Y, 0);
        break;
    case LAYER_KIND_BATCHNORM:
        encode_batchnorm_forward(pnet, layer, rows, X, Y, training);
        break;
    case LAYER_KIND_FLATTEN:
    case LAYER_KIND_RESHAPE:
        cmd_copy(X, Y, count * sizeof(float));
        break;
    default:
        g_vk.record_failed = 1;
        return;
    }

    if (dst->activation == ACTIVATION_SOFTMAX)
    {
        struct { uint32_t rows, cols; } sp = { (uint32_t)rows, out };
        cmd_dispatch(K_SOFTMAX, &Y, 1, &sp, sizeof(sp), groups_for((size_t)rows, 64), 1);
    }
    else if (!activated && dst->activation != ACTIVATION_NULL)
        k_bias_act(Y, NULL, Z, count, out, dst->activation);

    if (mask)
    {
        struct { uint32_t n, backward; } dp = { (uint32_t)count, 0 };
        VkBuf *dbufs[3] = { Y, mask, Y };
        cmd_dispatch(K_DROPOUT, dbufs, 3, &dp, sizeof(dp), groups_for(count, 256), 1);
    }
}

// Conv2D layers apply their activation's derivative in encode_conv_backward (softmax
// keeps its Jacobian kernel)
static int conv_fuses_derivative(PLayer l)
{
    return l->kind == LAYER_KIND_CONV2D && l->activation != ACTIVATION_SOFTMAX;
}

//================================================================================================
// Networks and inference
//================================================================================================

// 1 if the network's parameters were uploaded (the check ann.c makes)
static int network_on_gpu(const PNetwork pnet)
{
    for (int layer = 1; layer < pnet->layer_count; layer++)
        if (pnet->layers[layer].param_count > 0)
            return pnet->layers[layer].params[0].value->gpu_buf != NULL;
    return 0;
}

static void vk_free_network(PNetwork pnet)
{
    if (!pnet)
        return;
    finish_all_slots();
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
        {
            PTensor v = l->params[p].value;
            if (v && v->gpu_buf)
            {
                buf_destroy((VkBuf *)v->gpu_buf);
                v->gpu_buf = NULL;
            }
        }
    }
    // the training buffers are released by tensor_free() when the network is freed
    free_scratch();
    for (int s = 0; s < TRAIN_SLOTS; s++)
    {
        VkSlot *slot = &g_vk.slots[s];
        buf_destroy(slot->inputs);
        buf_destroy(slot->targets);
        buf_destroy(slot->row_loss);
        slot->inputs = slot->targets = slot->row_loss = NULL;
        for (int i = 0; i < slot->mask_layers; i++)
            buf_destroy(slot->masks[i]);
        free(slot->masks);
        slot->masks = NULL;
        slot->mask_layers = 0;
    }
    g_vk.training_buffers_ready = 0;
}

static int vk_upload_network(PNetwork pnet)
{
    if (!pnet || !g_vk.initialized)
        return ERR_NULL_PTR;

    // Every layer kind in ann.h is implemented; reject any added later
    for (int layer = 0; layer < pnet->layer_count; layer++)
    {
        Layer_kind kind = pnet->layers[layer].kind;
        if (kind != LAYER_KIND_DENSE && kind != LAYER_KIND_CONV2D && kind != LAYER_KIND_MAXPOOL2D &&
            kind != LAYER_KIND_FLATTEN && kind != LAYER_KIND_UPSAMPLE2D && kind != LAYER_KIND_RESHAPE &&
            kind != LAYER_KIND_GLOBAL_AVGPOOL2D && kind != LAYER_KIND_BATCHNORM)
            return ERR_INVALID;
    }

    // Rebuild the optimizer state on the next batch, from the CPU's
    g_vk.training_buffers_ready = 0;

    size_t total = 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
        {
            if (!tensor_buf(l->params[p].value))
            {
                vk_free_network(pnet);
                return ERR_FAIL;
            }
            total += staged(tensor_bytes(l->params[p].value));
        }
    }

    int s = sync_slot();
    int ok = staging_reserve(s, total) && cmd_begin(s);
    if (ok)
    {
        for (int layer = 1; layer < pnet->layer_count; layer++)
        {
            PLayer l = &pnet->layers[layer];
            for (int p = 0; p < l->param_count; p++)
                cmd_upload((VkBuf *)l->params[p].value->gpu_buf, l->params[p].value->values,
                           tensor_bytes(l->params[p].value));
        }
        ok = cmd_submit_wait();
    }
    if (!ok)
    {
        // never leave a partial copy that dispatch would take for the network
        vk_free_network(pnet);
        return ERR_FAIL;
    }
    return ERR_OK;
}

//-----------------------------------------------------------
// Copy each tensor's GPU buffer to its values, in one command buffer
//-----------------------------------------------------------
static int download_tensors(PTensor *list, int n)
{
    size_t total = 0;
    for (int i = 0; i < n; i++)
        total += staged(tensor_bytes(list[i]));
    const void **src = (const void **)calloc((size_t)(n ? n : 1), sizeof(*src));
    int s = sync_slot();
    int ok = src && staging_reserve(s, total) && cmd_begin(s);
    if (ok)
    {
        for (int i = 0; i < n; i++)
            src[i] = cmd_download((VkBuf *)list[i]->gpu_buf, tensor_bytes(list[i]));
        ok = cmd_submit_wait();
    }
    for (int i = 0; ok && i < n; i++)
        memcpy(list[i]->values, src[i], tensor_bytes(list[i]));
    free(src);
    return ok;
}

//-----------------------------------------------------------
// Copy every parameter and its optimizer state to the CPU, so ann_predict(),
// ann_save_network() and CPU (online) training see the latest values
//-----------------------------------------------------------
static void vk_sync_weights(PNetwork pnet)
{
    if (!pnet || !g_vk.initialized)
        return;

    PTensor *list = (PTensor *)calloc((size_t)pnet->layer_count * ANN_LAYER_MAX_PARAMS * 3, sizeof(*list));
    if (!list)
        return;
    int n = 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
        {
            PTensor state[3] = { l->params[p].value, l->params[p].m, l->params[p].v };
            for (int i = 0; i < 3; i++)
                if (state[i] && state[i]->gpu_buf)
                    list[n++] = state[i];
        }
    }
    if (!download_tensors(list, n))
        fprintf(stderr, "[Vulkan] Copying the weights from the GPU failed\n");
    free(list);
}

//-----------------------------------------------------------
// Forward pass of rows samples through the ping-pong buffers
//-----------------------------------------------------------
static int forward_infer(PNetwork pnet, const real *inputs, real *outputs, int rows)
{
    int in_nodes = pnet->layers[0].node_count;
    int out_nodes = pnet->layers[pnet->layer_count - 1].node_count;
    int max_nodes = 0;
    for (int i = 0; i < pnet->layer_count; i++)
        if (pnet->layers[i].node_count > max_nodes)
            max_nodes = pnet->layers[i].node_count;

    // inference shares the layers' work buffers with training
    int s = sync_slot();
    size_t bytes = (size_t)rows * max_nodes * sizeof(float);
    if (!ensure_buf(&g_vk.infer[0], bytes) || !ensure_buf(&g_vk.infer[1], bytes) ||
        !ensure_layer_scratch(pnet, rows, 0))
        return 0;
    size_t in_bytes = (size_t)rows * in_nodes * sizeof(float);
    size_t out_bytes = (size_t)rows * out_nodes * sizeof(float);
    if (!staging_reserve(s, staged(in_bytes) + staged(out_bytes)) || !cmd_begin(s))
        return 0;

    cmd_upload(g_vk.infer[0], inputs, in_bytes);
    int cur = 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        encode_layer_forward(pnet, layer, rows, g_vk.infer[cur], g_vk.infer[1 - cur], NULL, NULL, 0);
        cur = 1 - cur;
    }
    const void *result = cmd_download(g_vk.infer[cur], out_bytes);
    if (!cmd_submit_wait())
        return 0;
    memcpy(outputs, result, out_bytes);
    return 1;
}

static int vk_eval_single(PNetwork pnet)
{
    if (!pnet || !g_vk.initialized || !network_on_gpu(pnet))
        return 0;
    return forward_infer(pnet, pnet->layers[0].t_values->values,
                         pnet->layers[pnet->layer_count - 1].t_values->values, 1);
}

static int vk_eval_batch(const PNetwork pnet, const real *inputs, real *outputs, int batch_size)
{
    if (!pnet || !inputs || !outputs || batch_size <= 0)
        return ERR_NULL_PTR;
    if (!g_vk.initialized || !network_on_gpu(pnet))
        return ERR_FAIL;

    int in_nodes = pnet->layers[0].node_count;
    int out_nodes = pnet->layers[pnet->layer_count - 1].node_count;

    // a large batch is evaluated in pieces that fit the work buffers
    size_t chunk_rows = EVAL_BATCH_BYTES / forward_bytes_per_sample(pnet);
    if (chunk_rows > EVAL_MAX_ROWS)
        chunk_rows = EVAL_MAX_ROWS;
    int chunk = chunk_rows < 1 ? 1 : (chunk_rows < (size_t)batch_size ? (int)chunk_rows : batch_size);

    for (int start = 0; start < batch_size; start += chunk)
    {
        int rows = batch_size - start < chunk ? batch_size - start : chunk;
        if (!forward_infer((PNetwork)pnet, inputs + (size_t)start * in_nodes, outputs + (size_t)start * out_nodes, rows))
            return ERR_FAIL;
    }
    return ERR_OK;
}

//================================================================================================
// Training
//
// Each batch is one command buffer: copy in the inputs, targets and dropout masks, zero the
// gradients, forward pass, output gradient and loss, backward pass, then clipping, the
// optimizer update and regularization, and copy out the per-row losses.
//================================================================================================

//-----------------------------------------------------------
// GPU buffers for the batch activations, pre-activations and gradients, and each
// parameter's gradient. The first batch after upload_network() also copies the CPU's
// optimizer state, so training continues where it left off.
//-----------------------------------------------------------
static int ensure_training_buffers(PNetwork pnet)
{
    size_t state_bytes = 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        if (!tensor_buf(l->t_batch_values) || !tensor_buf(l->t_batch_z) || !tensor_buf(l->t_batch_dl_dz))
            return 0;
        for (int p = 0; p < l->param_count; p++)
        {
            Layer_param *param = &l->params[p];
            if (!tensor_buf(param->grad))
                return 0;
            if (g_vk.training_buffers_ready)
                continue;
            if ((param->m && !tensor_buf(param->m)) || (param->v && !tensor_buf(param->v)))
                return 0;
            state_bytes += (param->m ? staged(tensor_bytes(param->m)) : 0) +
                           (param->v ? staged(tensor_bytes(param->v)) : 0);
        }
    }
    if (g_vk.training_buffers_ready)
        return 1;

    int s = sync_slot();
    if (!staging_reserve(s, state_bytes) || !cmd_begin(s))
        return 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
        {
            PTensor state[2] = { l->params[p].m, l->params[p].v };
            for (int i = 0; i < 2; i++)
                if (state[i])
                    cmd_upload((VkBuf *)state[i]->gpu_buf, state[i]->values, tensor_bytes(state[i]));
        }
    }
    if (!cmd_submit_wait())
        return 0;
    g_vk.training_buffers_ready = 1;
    return 1;
}

// The slot's GPU buffer for a layer's dropout mask, of at least bytes; call before cmd_begin()
static VkBuf *slot_mask(VkSlot *slot, int layer, size_t bytes)
{
    if (layer >= slot->mask_layers)
    {
        VkBuf **grown = (VkBuf **)realloc(slot->masks, (size_t)(layer + 1) * sizeof(*grown));
        if (!grown)
            return NULL;
        memset(grown + slot->mask_layers, 0, (size_t)(layer + 1 - slot->mask_layers) * sizeof(*grown));
        slot->masks = grown;
        slot->mask_layers = layer + 1;
    }
    return ensure_buf(&slot->masks[layer], bytes);
}

// The batch's dropout mask for a hidden layer, which ann.c drew on the host, or NULL when
// the layer does not use dropout
static PTensor dropout_mask(PNetwork pnet, int layer, int batch_size)
{
    PLayer l = &pnet->layers[layer];
    PTensor m = l->t_dropout_mask;
    if (!pnet->is_training || l->dropout_rate <= 0.0f || !m || m->rows != batch_size || m->cols != l->node_count)
        return NULL;
    return m;
}

//-----------------------------------------------------------
// Encode the backward pass. On entry the output layer's t_batch_dl_dz holds -dL/dz (-dL/da
// for a Conv2D output, which applies its own derivative). For each layer, from the output
// down to layer 1:
//   1. hidden layers: undo dropout, then the activation derivative (Conv2D applies it in its
//      backward pass)
//   2. the kind's backward pass: accumulate parameter gradients (dense: dW += dl_dz^T * A_prev,
//      bias_grad += column sums of dl_dz) and write dl_dz_prev (not for layer 1, whose input
//      needs none)
//-----------------------------------------------------------
static void encode_backward(PNetwork pnet, int batch_size, VkSlot *slot)
{
    for (int li = pnet->layer_count - 1; li >= 1; li--)
    {
        PLayer layer = &pnet->layers[li];
        PLayer prev = &pnet->layers[li - 1];
        uint32_t nodes = (uint32_t)layer->node_count;
        uint32_t prev_nodes = (uint32_t)prev->node_count;
        size_t count = (size_t)batch_size * nodes;

        VkBuf *dl_dz = (VkBuf *)layer->t_batch_dl_dz->gpu_buf;
        VkBuf *A_prev = li == 1 ? slot->inputs : (VkBuf *)prev->t_batch_values->gpu_buf;
        VkBuf *dX = li > 1 ? (VkBuf *)prev->t_batch_dl_dz->gpu_buf : NULL;

        if (li < pnet->layer_count - 1)
        {
            if (dropout_mask(pnet, li, batch_size))
            {
                struct { uint32_t n, backward; } p = { (uint32_t)count, 1 };
                VkBuf *bufs[3] = { (VkBuf *)layer->t_batch_values->gpu_buf, slot->masks[li], dl_dz };
                cmd_dispatch(K_DROPOUT, bufs, 3, &p, sizeof(p), groups_for(count, 256), 1);
            }
            if (!conv_fuses_derivative(layer))
                k_activation_derivative(layer, batch_size);
        }

        switch (layer->kind)
        {
        case LAYER_KIND_DENSE:
            // dW += dl_dz^T * A_prev   ([nodes x batch] * [batch x prev_nodes])
            k_gemm(1, 0, nodes, prev_nodes, (uint32_t)batch_size, 1.0f, dl_dz, A_prev, 1.0f,
                   (VkBuf *)layer->t_gradients->gpu_buf);

            // bias_grad += column sums of dl_dz
            {
                struct { uint32_t rows, cols; } p = { (uint32_t)batch_size, nodes };
                VkBuf *bufs[2] = { dl_dz, (VkBuf *)layer->t_bias_grad->gpu_buf };
                cmd_dispatch(K_BIAS_GRAD_SUM, bufs, 2, &p, sizeof(p), groups_for(nodes, 1), 1);
            }

            // dl_dz_prev = dl_dz * W   ([batch x nodes] * [nodes x prev_nodes])
            if (dX)
                k_gemm(0, 0, (uint32_t)batch_size, prev_nodes, nodes, 1.0f, dl_dz,
                       (VkBuf *)layer->t_weights->gpu_buf, 0.0f, dX);
            break;
        case LAYER_KIND_CONV2D:
            // the output layer's delta already has its derivative, unless the loss is MSE
            encode_conv_backward(pnet, li, batch_size, A_prev, dl_dz, dX,
                                 li < pnet->layer_count - 1 || pnet->loss_type == LOSS_MSE ? layer->activation
                                                                                            : ACTIVATION_NULL);
            break;
        case LAYER_KIND_MAXPOOL2D:
            if (dX)
                encode_maxpool_backward(pnet, li, batch_size, dl_dz, dX);
            break;
        case LAYER_KIND_UPSAMPLE2D:
            if (dX)
                encode_upsample(pnet, li, batch_size, dl_dz, dX, 1);
            break;
        case LAYER_KIND_GLOBAL_AVGPOOL2D:
            if (dX)
                encode_global_avgpool(pnet, li, batch_size, dl_dz, dX, 1);
            break;
        case LAYER_KIND_BATCHNORM:
            encode_batchnorm_backward(pnet, li, batch_size, dl_dz, dX);
            break;
        case LAYER_KIND_FLATTEN:
        case LAYER_KIND_RESHAPE:
            if (dX)
                cmd_copy(dl_dz, dX, count * sizeof(float));
            break;
        default:
            g_vk.record_failed = 1;
            return;
        }
    }
}

//-----------------------------------------------------------
// Encode gradient clipping, the optimizer update and L1/L2 regularization for every
// parameter, as the CPU optimizers do. pnet->train_iteration must already count this batch.
//-----------------------------------------------------------
static void encode_optimizer(PNetwork pnet)
{
    struct {
        uint32_t n0, n1, opt;
        float lr, beta, beta1, beta2, eps, bc1, bc2, max_grad;
        float decay0, l1_0, decay1, l1_1;
    } p;
    p.lr = (float)pnet->learning_rate;
    p.beta = 0.9f;
    p.beta1 = 0.9f;
    p.beta2 = 0.999f;
    p.eps = 1e-8f;
    p.bc1 = p.bc2 = 0.0f;
    p.max_grad = pnet->max_gradient > 0.0f ? (float)pnet->max_gradient : 0.0f;
    switch (pnet->optimizer)
    {
    case OPT_SGD:      p.opt = VK_OPT_SGD; break;
    case OPT_MOMENTUM: p.opt = VK_OPT_MOMENTUM; break;
    case OPT_ADAGRAD:  p.opt = VK_OPT_ADAGRAD; break;
    case OPT_RMSPROP:  p.opt = VK_OPT_RMSPROP; break;
    case OPT_ADAM:
    {
        float t = (float)pnet->train_iteration;
        p.opt = VK_OPT_ADAM;
        p.bc1 = 1.0f / (1.0f - powf(p.beta1, t));
        p.bc2 = 1.0f / (1.0f - powf(p.beta2, t));
        break;
    }
    default:
        g_vk.record_failed = 1;
        return;
    }

    // L2 multiplies by decay and L1 moves weights l1 toward zero, for parameters that take
    // regularization (weights, not biases)
    float decay = pnet->l2_lambda > 0.0f ? 1.0f - (float)(pnet->learning_rate * pnet->l2_lambda) : 1.0f;
    float l1 = pnet->l1_lambda > 0.0f ? (float)(pnet->learning_rate * pnet->l1_lambda) : 0.0f;

    // One dispatch per pair of parameters (a dense or conv layer's weights and bias), which
    // clips, updates and regularizes both
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int i = 0; i < l->param_count; i += 2)
        {
            VkBuf *bufs[8];
            uint32_t n[2] = { 0, 0 };
            float reg[2][2] = { { 1.0f, 0.0f }, { 1.0f, 0.0f } };
            for (int k = 0; k < 2; k++)
            {
                // a missing second parameter repeats the first, with no elements
                Layer_param *param = &l->params[i + k < l->param_count ? i + k : i];
                VkBuf *w = (VkBuf *)param->value->gpu_buf;
                // optimizers without m or v never touch the binding; w stands in
                bufs[4 * k + 0] = w;
                bufs[4 * k + 1] = (VkBuf *)param->grad->gpu_buf;
                bufs[4 * k + 2] = param->m ? (VkBuf *)param->m->gpu_buf : w;
                bufs[4 * k + 3] = param->v ? (VkBuf *)param->v->gpu_buf : w;
                // running statistics (batch norm) are updated by their own kernel
                if (i + k >= l->param_count || param->state)
                    continue;
                n[k] = (uint32_t)(param->value->rows * param->value->cols);
                if (param->regularize)
                {
                    reg[k][0] = decay;
                    reg[k][1] = l1;
                }
            }
            if (n[0] + n[1] == 0)
                continue;
            p.n0 = n[0];
            p.n1 = n[1];
            p.decay0 = reg[0][0];
            p.l1_0 = reg[0][1];
            p.decay1 = reg[1][0];
            p.l1_1 = reg[1][1];
            cmd_dispatch(K_OPTIMIZER, bufs, 8, &p, sizeof(p), groups_for((size_t)n[0] + n[1], 256), 1);
        }
    }
}

//-----------------------------------------------------------
// One mini-batch: forward pass, backward pass and optimizer update. The batch inputs are in
// layers[0].t_batch_values and the dropout masks are drawn. Copies them into a free slot,
// records the step and submits it without waiting: the CPU assembles the next batch while
// the GPU runs this one, and vk_train_finish() collects the losses. Returns 1 when the GPU
// took the batch, 0 to fall back to the CPU.
//-----------------------------------------------------------
static int vk_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size)
{
    if (!pnet || !g_vk.initialized || !network_on_gpu(pnet) || !pnet->layers[0].t_batch_values)
        return 0;

    // The slot's previous step must be done before its buffers are reused
    int s = g_vk.next_slot;
    VkSlot *slot = &g_vk.slots[s];
    finish_slot(s);
    if (g_vk.train_failed)
        return 0;

    int out_idx = pnet->layer_count - 1;
    PLayer out = &pnet->layers[out_idx];
    int input_nodes = pnet->layers[0].node_count;
    int output_nodes = out->node_count;
    size_t input_bytes = (size_t)batch_size * input_nodes * sizeof(float);
    size_t target_bytes = (size_t)batch_size * output_nodes * sizeof(float);
    size_t loss_bytes = (size_t)batch_size * sizeof(float);

    // Buffers that grow (the first batch, or a larger one) wait for the steps in flight
    if (!ensure_training_buffers(pnet) || !ensure_layer_scratch(pnet, batch_size, 1) ||
        !ensure_buf(&slot->inputs, input_bytes) || !ensure_buf(&slot->targets, target_bytes) ||
        !ensure_buf(&slot->row_loss, loss_bytes))
        return 0;

    size_t stage_bytes = staged(input_bytes) + staged(target_bytes) + staged(loss_bytes);
    for (int layer = 1; layer < out_idx; layer++)
    {
        PTensor mask = dropout_mask(pnet, layer, batch_size);
        if (mask)
        {
            if (!slot_mask(slot, layer, tensor_bytes(mask)))
                return 0;
            stage_bytes += staged(tensor_bytes(mask));
        }
    }
    if (!staging_reserve(s, stage_bytes) || !cmd_begin(s))
        return 0;

    // This step's inputs, targets and dropout masks, in the slot's buffers
    cmd_upload(slot->inputs, pnet->layers[0].t_batch_values->values, input_bytes);
    cmd_upload(slot->targets, batch_targets->values, target_bytes);
    for (int layer = 1; layer < out_idx; layer++)
    {
        PTensor mask = dropout_mask(pnet, layer, batch_size);
        if (mask)
            cmd_upload(slot->masks[layer], mask->values, tensor_bytes(mask));
    }

    // Zero the gradients
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int p = 0; p < l->param_count; p++)
            cmd_zero((VkBuf *)l->params[p].grad->gpu_buf);
    }

    // Forward pass, saving pre-activations; dropout on hidden layers
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer dst = &pnet->layers[layer];
        int masked = layer < out_idx && dropout_mask(pnet, layer, batch_size);
        VkBuf *X = layer == 1 ? slot->inputs : (VkBuf *)pnet->layers[layer - 1].t_batch_values->gpu_buf;
        encode_layer_forward(pnet, layer, batch_size, X, (VkBuf *)dst->t_batch_values->gpu_buf,
                             (VkBuf *)dst->t_batch_z->gpu_buf, masked ? slot->masks[layer] : NULL, 1);
    }

    // Output delta = t - y and each row's loss; MSE then applies the output activation's
    // derivative (cross-entropy with softmax or sigmoid needs none; a Conv2D output applies
    // it in its backward pass)
    {
        struct { uint32_t rows, cols, mode; } p = {
            (uint32_t)batch_size, (uint32_t)output_nodes,
            pnet->loss_type == LOSS_MSE ? 0u : (out->activation == ACTIVATION_SIGMOID ? 2u : 1u)
        };
        VkBuf *bufs[4] = { (VkBuf *)out->t_batch_values->gpu_buf, slot->targets,
                           (VkBuf *)out->t_batch_dl_dz->gpu_buf, slot->row_loss };
        cmd_dispatch(K_OUTPUT_DELTA_LOSS, bufs, 4, &p, sizeof(p), groups_for((size_t)batch_size, 64), 1);
        if (pnet->loss_type == LOSS_MSE && !conv_fuses_derivative(out))
            k_activation_derivative(out, batch_size);
    }

    encode_backward(pnet, batch_size, slot);

    // Adam's bias correction counts this batch
    pnet->train_iteration++;
    encode_optimizer(pnet);

    const float *row_loss = (const float *)cmd_download(slot->row_loss, loss_bytes);
    if (!cmd_submit())
    {
        pnet->train_iteration--;
        return 0;
    }
    slot->loss_src = row_loss;
    slot->batch_size = batch_size;
    slot->divisor = pnet->loss_type == LOSS_MSE ? batch_size * output_nodes : batch_size;
    g_vk.next_slot = (s + 1) % TRAIN_SLOTS;
    return 1;
}

//-----------------------------------------------------------
// Wait for the steps vk_train_batch() submitted; *loss_sum gets the sum of their batch
// losses. Returns 0 if any failed.
//-----------------------------------------------------------
static int vk_train_finish(PNetwork pnet, real *loss_sum)
{
    (void)pnet;
    finish_all_slots();
    if (loss_sum)
        *loss_sum = g_vk.loss_sum;
    int ok = !g_vk.train_failed;
    g_vk.loss_sum = 0.0f;
    g_vk.train_failed = 0;
    return ok;
}

GpuBackend vulkan_backend = {
    "Vulkan",                   /* name           */
    vk_init,                    /* init           */
    vk_upload_network,          /* upload_network */
    vk_free_network,            /* free_network   */
    vk_sync_weights,            /* sync_weights   */
    vk_release_buffer,          /* release_buffer */
    vk_eval_single,             /* eval_single    */
    vk_eval_batch,              /* eval_batch     */
    vk_train_batch,             /* train_batch    */
    vk_train_finish,            /* train_finish   */
};

#endif /* USE_VULKAN */
