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
// Vulkan GPU backend: inference and training for dense networks on any Vulkan 1.1 GPU (AMD,
// Intel, NVIDIA) on Windows and Linux. See docs/GPU_VULKAN.md.
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
//   buffer, submitted, and waited for. Dispatches are separated by full memory barriers.
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
    X(vkQueueSubmit)

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
    K_BIAS_GRAD_SUM, K_GRAD_CLIP, K_OPTIMIZER, K_REGULARIZE, K_DROPOUT,
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
    { SPV(grad_clip), 0, 0 },
    { SPV(optimizer), 0, 0 },
    { SPV(regularize), 0, 0 },
    { SPV(dropout), 0, 0 },
};

// Optimizer codes in shaders/vk/optimizer.comp
enum { VK_OPT_SGD, VK_OPT_MOMENTUM, VK_OPT_ADAGRAD, VK_OPT_RMSPROP, VK_OPT_ADAM };

#define VK_MAX_BINDINGS     4       // storage buffers a kernel binds
#define VK_PUSH_BYTES       64      // push constant space a kernel uses
#define VK_POOL_SETS        256     // descriptor sets per pool
#define VK_MAX_POOLS        256     // pools a command buffer may use
#define GEMM_TILE           64      // gemm.comp's output tile
#define EVAL_BATCH_BYTES    ((size_t)256 << 20)
#define EVAL_MAX_ROWS       (1 << 20)

//-----------------------------------------------------------
// A tensor's GPU buffer (t->gpu_buf), or a backend work buffer
//-----------------------------------------------------------
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;          // bytes
    void *map;                  // host address when the host copies in and out directly
} VkBuf;

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
    uint32_t max_groups[2];     // maxComputeWorkGroupCount x, y
    int map_device;             // device buffers are host-visible and mapped (no staging)

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipe_layout;
    VkPipeline pipelines[K_COUNT];
    VkDescriptorPool pools[VK_MAX_POOLS];
    int pool_count, pool_cur;   // pools created, and the one sets come from now

    // The command buffer being recorded; see cmd_begin()
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    int record_failed;          // a command could not be recorded; the submit fails
    int barrier_pending;        // a command since the last barrier wrote memory

    VkBuf *staging;             // host-visible; uploads and downloads pass through it
    VkDeviceSize staging_used;

    VkBuf *inputs, *targets, *row_loss;     // training step work buffers
    VkBuf *infer[2];                        // inference ping-pong buffers
    int training_buffers_ready; // optimizer state was uploaded since upload_network()
    real loss_sum;              // losses of the steps since vk_train_finish()
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
    const char *ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
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
            ci.enabledLayerCount = 1;
            ci.ppEnabledLayerNames = &layer;
            ci.enabledExtensionCount = 1;
            ci.ppEnabledExtensionNames = &ext;
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
        if (props.apiVersion < VK_API_VERSION_1_1 || compute_queue_family(devices[i]) < 0)
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

    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_vk.cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &ai, &g_vk.cmd) != VK_SUCCESS)
        return 0;

    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    return vkCreateFence(g_vk.device, &fi, NULL, &g_vk.fence) == VK_SUCCESS;
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

        uint32_t spec[2] = { g_kernels[k].trans_a, g_kernels[k].trans_b };
        VkSpecializationMapEntry entries[2] = { { 0, 0, sizeof(uint32_t) }, { 1, sizeof(uint32_t), sizeof(uint32_t) } };
        VkSpecializationInfo si = { 2, entries, sizeof(spec), spec };

        VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = k <= K_GEMM_TN ? &si : NULL;
        ci.layout = g_vk.pipe_layout;
        VkResult res = vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g_vk.pipelines[k]);
        vkDestroyShaderModule(g_vk.device, module, NULL);
        if (res != VK_SUCCESS)
        {
            fprintf(stderr, "[Vulkan] Pipeline %d failed (%d)\n", k, (int)res);
            return 0;
        }
    }
    return 1;
}

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

    g_vk.init_failed = 0;
    g_vk.initialized = 1;
    return 1;
}

//================================================================================================
// Buffers
//================================================================================================

static int find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want)
{
    for (uint32_t t = 0; t < g_vk.mem.memoryTypeCount; t++)
        if ((type_bits & (1u << t)) && (g_vk.mem.memoryTypes[t].propertyFlags & want) == want)
            return (int)t;
    return -1;
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
    int type = -1;
    if (host)
    {
        type = find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (type < 0)
            type = find_memory_type(req.memoryTypeBits, visible);
    }
    else
    {
        if (g_vk.map_device)
            type = find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0)
            type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0)
            type = find_memory_type(req.memoryTypeBits, 0);
    }

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    if (type < 0 || vkAllocateMemory(g_vk.device, &ai, NULL, &b->memory) != VK_SUCCESS)
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

// A work buffer of at least bytes, reallocated when too small
static VkBuf *ensure_buf(VkBuf **b, size_t bytes)
{
    if (*b && (*b)->size >= bytes)
        return *b;
    buf_destroy(*b);
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
    buf_destroy((VkBuf *)gpu_buf);
}

// Room for bytes of uploads and downloads in the next command buffer; call before cmd_begin()
static int staging_reserve(size_t bytes)
{
    if (g_vk.staging && g_vk.staging->size >= bytes)
        return 1;
    buf_destroy(g_vk.staging);
    g_vk.staging = buf_create(bytes < ((size_t)1 << 20) ? ((size_t)1 << 20) : bytes, 1);
    return g_vk.staging != NULL;
}

// Staging bytes per copy, rounded up for the offset alignment staging_take() keeps
static size_t staged(size_t bytes)
{
    return (bytes + 15) & ~(size_t)15;
}

//================================================================================================
// Command recording
//================================================================================================

// Start recording an operation; it runs when cmd_submit_wait() is called
static int cmd_begin(void)
{
    for (int i = 0; i < g_vk.pool_count; i++)
        vkResetDescriptorPool(g_vk.device, g_vk.pools[i], 0);
    g_vk.pool_cur = 0;
    g_vk.staging_used = 0;
    g_vk.record_failed = 0;
    g_vk.barrier_pending = 0;

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(g_vk.cmd, &bi) == VK_SUCCESS;
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
    vkCmdPipelineBarrier(g_vk.cmd, stages, stages, 0, 1, &mb, 0, NULL, 0, NULL);
    g_vk.barrier_pending = 0;
}

// Submit the recorded operation and wait for it; 1 if it completed
static int cmd_submit_wait(void)
{
    // results the host reads (downloads, mapped buffers) must be visible to it
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(g_vk.cmd) != VK_SUCCESS || g_vk.record_failed)
        return 0;

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_vk.cmd;
    VkResult res = vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence);
    if (res == VK_SUCCESS)
        res = vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_vk.device, 1, &g_vk.fence);
    if (res != VK_SUCCESS)
        fprintf(stderr, "[Vulkan] Submit failed (%d)\n", (int)res);
    return res == VK_SUCCESS;
}

// Take bytes of the staging buffer for this command buffer; returns the offset
static VkDeviceSize staging_take(size_t bytes)
{
    VkDeviceSize off = g_vk.staging_used;
    if (!g_vk.staging || off + bytes > g_vk.staging->size)
    {
        g_vk.record_failed = 1;
        return 0;
    }
    g_vk.staging_used += staged(bytes);
    return off;
}

// Record a copy of host memory into dst (or copy it now when dst is mapped)
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
    memcpy((char *)g_vk.staging->map + off, src, bytes);
    cmd_barrier();
    VkBufferCopy region = { off, 0, bytes };
    vkCmdCopyBuffer(g_vk.cmd, g_vk.staging->buffer, dst->buffer, 1, &region);
    g_vk.barrier_pending = 1;
}

// Record a copy of src's first bytes to where the host reads them once cmd_submit_wait()
// returns; returns that address
static const void *cmd_download(VkBuf *src, size_t bytes)
{
    if (src->map)
        return src->map;
    VkDeviceSize off = staging_take(bytes);
    if (g_vk.record_failed)
        return NULL;
    cmd_barrier();
    VkBufferCopy region = { 0, off, bytes };
    vkCmdCopyBuffer(g_vk.cmd, src->buffer, g_vk.staging->buffer, 1, &region);
    g_vk.barrier_pending = 1;
    return (const char *)g_vk.staging->map + off;
}

static void cmd_zero(VkBuf *b)
{
    cmd_barrier();
    vkCmdFillBuffer(g_vk.cmd, b->buffer, 0, VK_WHOLE_SIZE, 0);
    g_vk.barrier_pending = 1;
}

// A descriptor set for one dispatch, from the pools reset by cmd_begin()
static VkDescriptorSet alloc_set(void)
{
    while (g_vk.pool_cur < VK_MAX_POOLS)
    {
        if (g_vk.pool_cur == g_vk.pool_count)
        {
            VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_POOL_SETS * VK_MAX_BINDINGS };
            VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pi.maxSets = VK_POOL_SETS;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &size;
            if (vkCreateDescriptorPool(g_vk.device, &pi, NULL, &g_vk.pools[g_vk.pool_count]) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            g_vk.pool_count++;
        }
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = g_vk.pools[g_vk.pool_cur];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_vk.set_layout;
        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) == VK_SUCCESS)
            return set;
        g_vk.pool_cur++;
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
static void cmd_dispatch(Kernel k, VkBuf *const *bufs, int n_bufs, const void *params, uint32_t params_size,
                         uint32_t groups_x, uint32_t groups_y)
{
    if (g_vk.record_failed)
        return;
    VkDescriptorSet set = alloc_set();
    if (!set || groups_x > g_vk.max_groups[0] || groups_y > g_vk.max_groups[1])
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
    vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipelines[k]);
    vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(g_vk.cmd, g_vk.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    vkCmdDispatch(g_vk.cmd, groups_x, groups_y, 1);
    g_vk.barrier_pending = 1;
}

//================================================================================================
// Kernels
//================================================================================================

// Row-major C[M x N] = alpha * op(A) * op(B) + beta * C (see gemm.comp)
static void k_gemm(int trans_a, int trans_b, uint32_t M, uint32_t N, uint32_t K,
                   float alpha, VkBuf *A, VkBuf *B, float beta, VkBuf *C)
{
    struct { uint32_t M, N, K; float alpha, beta; } p = { M, N, K, alpha, beta };
    VkBuf *bufs[3] = { A, B, C };
    Kernel k = trans_a ? K_GEMM_TN : (trans_b ? K_GEMM_NT : K_GEMM_NN);
    cmd_dispatch(k, bufs, 3, &p, sizeof(p), (N + GEMM_TILE - 1) / GEMM_TILE, (M + GEMM_TILE - 1) / GEMM_TILE);
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

//-----------------------------------------------------------
// A dense layer's forward pass: Y = activate(X * W^T + b), saving the values before
// activation in Z for GELU when Z is given, then dropout when mask is given
//-----------------------------------------------------------
static void encode_dense_forward(PNetwork pnet, int layer, int rows, VkBuf *X, VkBuf *Y, VkBuf *Z, VkBuf *mask)
{
    PLayer dst = &pnet->layers[layer];
    uint32_t out = (uint32_t)dst->node_count;
    uint32_t in = (uint32_t)pnet->layers[layer - 1].node_count;
    size_t count = (size_t)rows * out;

    k_gemm(0, 1, (uint32_t)rows, out, in, 1.0f, X, (VkBuf *)dst->t_weights->gpu_buf, 0.0f, Y);

    int save_z = Z && dst->activation == ACTIVATION_GELU;
    struct { uint32_t n, cols, act, save_z; } p = { (uint32_t)count, out, (uint32_t)dst->activation, (uint32_t)save_z };
    VkBuf *bufs[3] = { Y, (VkBuf *)dst->t_bias->gpu_buf, save_z ? Z : Y };
    cmd_dispatch(K_BIAS_ACT, bufs, 3, &p, sizeof(p), groups_for(count, 256), 1);

    if (dst->activation == ACTIVATION_SOFTMAX)
    {
        struct { uint32_t rows, cols; } sp = { (uint32_t)rows, out };
        cmd_dispatch(K_SOFTMAX, &Y, 1, &sp, sizeof(sp), groups_for((size_t)rows, 64), 1);
    }

    if (mask)
    {
        struct { uint32_t n, backward; } dp = { (uint32_t)count, 0 };
        VkBuf *dbufs[3] = { Y, mask, Y };
        cmd_dispatch(K_DROPOUT, dbufs, 3, &dp, sizeof(dp), groups_for(count, 256), 1);
    }
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
    g_vk.training_buffers_ready = 0;
}

static int vk_upload_network(PNetwork pnet)
{
    if (!pnet || !g_vk.initialized)
        return ERR_NULL_PTR;

    // Dense layers only so far; other networks stay on the CPU
    for (int layer = 0; layer < pnet->layer_count; layer++)
        if (pnet->layers[layer].kind != LAYER_KIND_DENSE)
            return ERR_INVALID;

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

    int ok = staging_reserve(total) && cmd_begin();
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
    int ok = src && staging_reserve(total) && cmd_begin();
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

    size_t bytes = (size_t)rows * max_nodes * sizeof(float);
    if (!ensure_buf(&g_vk.infer[0], bytes) || !ensure_buf(&g_vk.infer[1], bytes))
        return 0;
    size_t in_bytes = (size_t)rows * in_nodes * sizeof(float);
    size_t out_bytes = (size_t)rows * out_nodes * sizeof(float);
    if (!staging_reserve(staged(in_bytes) + staged(out_bytes)) || !cmd_begin())
        return 0;

    cmd_upload(g_vk.infer[0], inputs, in_bytes);
    int cur = 0;
    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        encode_dense_forward(pnet, layer, rows, g_vk.infer[cur], g_vk.infer[1 - cur], NULL, NULL);
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
    int max_nodes = 0;
    for (int i = 0; i < pnet->layer_count; i++)
        if (pnet->layers[i].node_count > max_nodes)
            max_nodes = pnet->layers[i].node_count;

    // a large batch is evaluated in pieces that fit the work buffers
    size_t chunk_rows = EVAL_BATCH_BYTES / (2 * (size_t)max_nodes * sizeof(float));
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

    if (!staging_reserve(state_bytes) || !cmd_begin())
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
// Encode the backward pass. On entry the output layer's t_batch_dl_dz holds -dL/dz. For each
// layer, from the output down to layer 1:
//   1. hidden layers: undo dropout, then the activation derivative
//   2. dW += dl_dz^T * A_prev, bias_grad += column sums of dl_dz, and dl_dz_prev = dl_dz * W
//      (not for layer 1, whose input needs none)
//-----------------------------------------------------------
static void encode_backward(PNetwork pnet, int batch_size)
{
    for (int li = pnet->layer_count - 1; li >= 1; li--)
    {
        PLayer layer = &pnet->layers[li];
        PLayer prev = &pnet->layers[li - 1];
        uint32_t nodes = (uint32_t)layer->node_count;
        uint32_t prev_nodes = (uint32_t)prev->node_count;
        size_t count = (size_t)batch_size * nodes;

        VkBuf *dl_dz = (VkBuf *)layer->t_batch_dl_dz->gpu_buf;
        VkBuf *A_prev = li == 1 ? g_vk.inputs : (VkBuf *)prev->t_batch_values->gpu_buf;

        if (li < pnet->layer_count - 1)
        {
            PTensor mask = dropout_mask(pnet, li, batch_size);
            if (mask)
            {
                struct { uint32_t n, backward; } p = { (uint32_t)count, 1 };
                VkBuf *bufs[3] = { (VkBuf *)layer->t_batch_values->gpu_buf, (VkBuf *)mask->gpu_buf, dl_dz };
                cmd_dispatch(K_DROPOUT, bufs, 3, &p, sizeof(p), groups_for(count, 256), 1);
            }
            k_activation_derivative(layer, batch_size);
        }

        // dW += dl_dz^T * A_prev   ([nodes x batch] * [batch x prev_nodes])
        k_gemm(1, 0, nodes, prev_nodes, (uint32_t)batch_size, 1.0f, dl_dz, A_prev, 1.0f,
               (VkBuf *)layer->t_gradients->gpu_buf);

        // bias_grad += column sums of dl_dz
        {
            struct { uint32_t rows, cols; } p = { (uint32_t)batch_size, nodes };
            VkBuf *bufs[2] = { dl_dz, (VkBuf *)layer->t_bias_grad->gpu_buf };
            cmd_dispatch(K_BIAS_GRAD_SUM, bufs, 2, &p, sizeof(p), groups_for(nodes, 64), 1);
        }

        // dl_dz_prev = dl_dz * W   ([batch x nodes] * [nodes x prev_nodes])
        if (li > 1)
            k_gemm(0, 0, (uint32_t)batch_size, prev_nodes, nodes, 1.0f, dl_dz,
                   (VkBuf *)layer->t_weights->gpu_buf, 0.0f, (VkBuf *)prev->t_batch_dl_dz->gpu_buf);
    }
}

//-----------------------------------------------------------
// Encode gradient clipping, the optimizer update and L1/L2 regularization for every
// parameter, as the CPU optimizers do. pnet->train_iteration must already count this batch.
//-----------------------------------------------------------
static void encode_optimizer(PNetwork pnet)
{
    struct {
        uint32_t n, opt;
        float lr, beta, beta1, beta2, eps, bc1, bc2;
    } p;
    p.lr = (float)pnet->learning_rate;
    p.beta = 0.9f;
    p.beta1 = 0.9f;
    p.beta2 = 0.999f;
    p.eps = 1e-8f;
    p.bc1 = p.bc2 = 0.0f;
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

    for (int layer = 1; layer < pnet->layer_count; layer++)
    {
        PLayer l = &pnet->layers[layer];
        for (int i = 0; i < l->param_count; i++)
        {
            Layer_param *param = &l->params[i];
            size_t count = (size_t)param->value->rows * param->value->cols;
            uint32_t groups = groups_for(count, 256);
            VkBuf *w = (VkBuf *)param->value->gpu_buf;
            VkBuf *g = (VkBuf *)param->grad->gpu_buf;

            if (pnet->max_gradient > 0.0f)
            {
                struct { uint32_t n; float max_grad; } cp = { (uint32_t)count, (float)pnet->max_gradient };
                cmd_dispatch(K_GRAD_CLIP, &g, 1, &cp, sizeof(cp), groups, 1);
            }

            // optimizers without m or v never touch the binding; w stands in
            VkBuf *bufs[4] = { w, g, param->m ? (VkBuf *)param->m->gpu_buf : w, param->v ? (VkBuf *)param->v->gpu_buf : w };
            p.n = (uint32_t)count;
            cmd_dispatch(K_OPTIMIZER, bufs, 4, &p, sizeof(p), groups, 1);

            if (param->regularize && (pnet->l2_lambda > 0.0f || pnet->l1_lambda > 0.0f))
            {
                struct { uint32_t n; float decay, l1; } rp = {
                    (uint32_t)count,
                    pnet->l2_lambda > 0.0f ? 1.0f - (float)(pnet->learning_rate * pnet->l2_lambda) : 1.0f,
                    pnet->l1_lambda > 0.0f ? (float)(pnet->learning_rate * pnet->l1_lambda) : 0.0f
                };
                cmd_dispatch(K_REGULARIZE, &w, 1, &rp, sizeof(rp), groups, 1);
            }
        }
    }
}

//-----------------------------------------------------------
// One mini-batch: forward pass, backward pass and optimizer update. The batch inputs are in
// layers[0].t_batch_values and the dropout masks are drawn. Returns 1 when the GPU ran the
// step, 0 to fall back to the CPU.
//-----------------------------------------------------------
static int vk_train_batch(PNetwork pnet, PTensor batch_targets, int batch_size)
{
    if (!pnet || !g_vk.initialized || !network_on_gpu(pnet) || !pnet->layers[0].t_batch_values)
        return 0;

    int out_idx = pnet->layer_count - 1;
    PLayer out = &pnet->layers[out_idx];
    int input_nodes = pnet->layers[0].node_count;
    int output_nodes = out->node_count;
    size_t input_bytes = (size_t)batch_size * input_nodes * sizeof(float);
    size_t target_bytes = (size_t)batch_size * output_nodes * sizeof(float);
    size_t loss_bytes = (size_t)batch_size * sizeof(float);

    if (!ensure_training_buffers(pnet) ||
        !ensure_buf(&g_vk.inputs, input_bytes) || !ensure_buf(&g_vk.targets, target_bytes) ||
        !ensure_buf(&g_vk.row_loss, loss_bytes))
        return 0;

    size_t stage_bytes = staged(input_bytes) + staged(target_bytes) + staged(loss_bytes);
    for (int layer = 1; layer < out_idx; layer++)
    {
        PTensor mask = dropout_mask(pnet, layer, batch_size);
        if (mask)
        {
            if (!tensor_buf(mask))
                return 0;
            stage_bytes += staged(tensor_bytes(mask));
        }
    }
    if (!staging_reserve(stage_bytes) || !cmd_begin())
        return 0;

    cmd_upload(g_vk.inputs, pnet->layers[0].t_batch_values->values, input_bytes);
    cmd_upload(g_vk.targets, batch_targets->values, target_bytes);
    for (int layer = 1; layer < out_idx; layer++)
    {
        PTensor mask = dropout_mask(pnet, layer, batch_size);
        if (mask)
            cmd_upload((VkBuf *)mask->gpu_buf, mask->values, tensor_bytes(mask));
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
        PTensor mask = layer < out_idx ? dropout_mask(pnet, layer, batch_size) : NULL;
        VkBuf *X = layer == 1 ? g_vk.inputs : (VkBuf *)pnet->layers[layer - 1].t_batch_values->gpu_buf;
        encode_dense_forward(pnet, layer, batch_size, X, (VkBuf *)dst->t_batch_values->gpu_buf,
                             (VkBuf *)dst->t_batch_z->gpu_buf, mask ? (VkBuf *)mask->gpu_buf : NULL);
    }

    // Output delta = t - y and each row's loss; MSE then applies the output activation's
    // derivative (cross-entropy with softmax or sigmoid needs none)
    {
        struct { uint32_t rows, cols, mode; } p = {
            (uint32_t)batch_size, (uint32_t)output_nodes,
            pnet->loss_type == LOSS_MSE ? 0u : (out->activation == ACTIVATION_SIGMOID ? 2u : 1u)
        };
        VkBuf *bufs[4] = { (VkBuf *)out->t_batch_values->gpu_buf, g_vk.targets,
                           (VkBuf *)out->t_batch_dl_dz->gpu_buf, g_vk.row_loss };
        cmd_dispatch(K_OUTPUT_DELTA_LOSS, bufs, 4, &p, sizeof(p), groups_for((size_t)batch_size, 64), 1);
        if (pnet->loss_type == LOSS_MSE)
            k_activation_derivative(out, batch_size);
    }

    encode_backward(pnet, batch_size);

    // Adam's bias correction counts this batch
    pnet->train_iteration++;
    encode_optimizer(pnet);

    const float *row_loss = (const float *)cmd_download(g_vk.row_loss, loss_bytes);
    if (!cmd_submit_wait())
    {
        pnet->train_iteration--;
        return 0;
    }

    real total = 0.0f;
    for (int b = 0; b < batch_size; b++)
        total += row_loss[b];
    g_vk.loss_sum += total / (real)(pnet->loss_type == LOSS_MSE ? batch_size * output_nodes : batch_size);
    return 1;
}

// Every step already completed in vk_train_batch(); report their losses
static int vk_train_finish(PNetwork pnet, real *loss_sum)
{
    (void)pnet;
    if (loss_sum)
        *loss_sum = g_vk.loss_sum;
    g_vk.loss_sum = 0.0f;
    return 1;
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
