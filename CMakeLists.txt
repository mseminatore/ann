cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)

# options
option(BUILD_SHARED "Build shared library (DLL/so/dylib)" OFF)

# default to an optimized Release build if the user didn't specify one.
# single-config generators (Unix Makefiles) ignore --config, so this is
# required to get -O3 instead of an unoptimized -O0 build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message("No build type specified, defaulting to Release (-O3)")
endif()

# testing
enable_testing()
add_test(NAME logic COMMAND logic "${CMAKE_SOURCE_DIR}/data/and.csv")
add_test(NAME digit5x7 COMMAND digit5x7 "${CMAKE_SOURCE_DIR}/data/num5x7.csv")
add_test(NAME save_test COMMAND save_test "${CMAKE_SOURCE_DIR}/data/mnist-fashion.nna" "${CMAKE_SOURCE_DIR}/data/fashion-mnist_test.csv")
add_test(NAME save_test_binary COMMAND save_test_binary "${CMAKE_SOURCE_DIR}/data/mnist-fashion.nnb" "${CMAKE_SOURCE_DIR}/data/fashion-mnist_test.csv")
add_test(NAME test_tensor COMMAND test_tensor)
add_test(NAME test_network COMMAND test_network)
add_test(NAME test_activations COMMAND test_activations)
add_test(NAME test_onnx_export COMMAND test_onnx_export)
add_test(NAME test_loss_functions COMMAND test_loss_functions)
add_test(NAME test_save_load COMMAND test_save_load)
add_test(NAME test_optimizers COMMAND test_optimizers)
add_test(NAME test_hypertune COMMAND test_hypertune)
add_test(NAME test_forward_pass COMMAND test_forward_pass)
add_test(NAME test_gradients COMMAND test_gradients)
add_test(NAME test_comprehensive_callbacks COMMAND test_comprehensive_callbacks)
add_test(NAME test_training_options COMMAND test_training_options)
add_test(NAME test_conv COMMAND test_conv)
add_test(NAME test_batchnorm COMMAND test_batchnorm)
add_test(NAME test_threads COMMAND test_threads)
add_test(NAME test_noise COMMAND test_noise)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# compiler warnings for C/CXX sources (nvcc has its own flag syntax)
if (NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>)
else()
    # fopen/fscanf/strcat are standard C; MSVC's *_s replacements are not portable
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# set compiler and arch specific optimizations
# NOTE: these are host-CPU SIMD flags and must only be applied to C/CXX sources.
# nvcc rejects bare flags like /arch:AVX ("A single input file is required..."), so they are
# scoped to C/CXX via a generator expression and never reach the CUDA (nvcc) front-end.
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX>)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-mavx2>
                        $<$<COMPILE_LANGUAGE:C,CXX>:-mfma>)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-mcpu=native>)
endif()

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

# build with or without BLAS library
if ( USE_BLAS OR USE_CBLAS OR USE_MKL OR USE_ACCELERATE )
    message("BLAS library build requested...")
    add_compile_options(-DUSE_BLAS)
    if ( USE_ACCELERATE )
        if ( NOT APPLE )
            message(FATAL_ERROR "USE_ACCELERATE is only supported on macOS")
        endif()
        # libann includes <cblas.h>; point it at Accelerate's copy
        execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(VECLIB_HEADERS "${MACOS_SDK}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers")
        if (NOT EXISTS "${VECLIB_HEADERS}/cblas.h")
            message(FATAL_ERROR "Accelerate cblas.h not found at ${VECLIB_HEADERS}")
        endif()
        message("Apple Accelerate library used")
        include_directories("${VECLIB_HEADERS}")
        # Accelerate names cblas_saxpby catlas_saxpby (same signature); its legacy CBLAS
        # header is marked deprecated on macOS 13.3+
        add_compile_options(-DUSE_ACCELERATE -Dcblas_saxpby=catlas_saxpby -Wno-deprecated-declarations)
        set(BLAS_LIB "-framework Accelerate")
    elseif ( USE_MKL )
        if (NOT DEFINED ENV{ONEAPI_ROOT})
            message(FATAL_ERROR "USE_MKL requires ONEAPI_ROOT environment variable (set by Intel oneAPI installer)")
        endif()
        set(MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest")
        message("Intel MKL library used (${MKL_ROOT})")
        add_compile_options(-DUSE_MKL)
        include_directories("${MKL_ROOT}/include")
        link_directories("${MKL_ROOT}/lib")
        set(CMAKE_BUILD_RPATH "${MKL_ROOT}/lib")
        set(CMAKE_INSTALL_RPATH "${MKL_ROOT}/lib")
        set(BLAS_LIB mkl_intel_lp64 mkl_sequential mkl_core)
    elseif ( USE_CBLAS )
        add_compile_options(-DUSE_CBLAS)
        set(BLAS_LIB "cblas")
        message("CBLAS library used")
        # CBLAS requires C11 atomics (host C/CXX only — these flags break nvcc)
        if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
            add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/std:c11>
                                $<$<COMPILE_LANGUAGE:C,CXX>:/experimental:c11atomics>)
            include_directories("C:/opt/cblas/include")
            link_directories("C:/opt/cblas/lib")
        else()
            add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-std=c11>)
            include_directories("/opt/cblas/include")
            link_directories("/opt/cblas/lib")
        endif()
    else()
        set(BLAS_LIB "openblas")
        message("OpenBLAS library used")
        include_directories("/opt/OpenBLAS/include")
        link_directories("/opt/OpenBLAS/lib")
    endif()
else()
    set(BLAS_LIB "")
endif()

# build with or without Apple Metal GPU inference support
set(METAL_LIB "")
set(METAL_SRC "")
if ( USE_METAL )
    if ( NOT APPLE )
        message(FATAL_ERROR "USE_METAL is only supported on macOS/Apple platforms")
    endif()
    message("Apple Metal GPU inference build requested...")
    add_compile_options(-DUSE_METAL)
    set(METAL_SRC "tensor_metal.m")
    set(METAL_LIB "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
    # Compile tensor_metal.m as Objective-C with ARC; its __bridge casts rely on
    # ARC to retain and release the MTLBuffers held in tensor gpu_buf pointers
    set_source_files_properties(tensor_metal.m PROPERTIES
        COMPILE_FLAGS "-x objective-c -fobjc-arc"
        LANGUAGE C
    )
endif()

# build with or without NVIDIA CUDA GPU support
set(CUDA_LIBS "")
set(CUDA_SRC "")
if ( USE_CUDA )
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message("NVIDIA CUDA GPU build requested...")
    # Select target GPU architecture(s). Override with -DCMAKE_CUDA_ARCHITECTURES=<list>,
    # e.g. 61 for Pascal (Quadro P620), 75 for Turing, 86 for Ampere.
    # Default targets Turing+Ampere (75;86) which compiles under both CUDA 12.x and 13.x.
    # NOTE: CUDA 13.x dropped offline codegen for Maxwell/Pascal (sm_5x/sm_6x); to target an
    # older GPU build with a CUDA 12.x toolkit AND set the matching arch, e.g.
    #   cmake -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=61 ..
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA architectures")
    endif()
    message("CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    add_compile_options(-DUSE_CUDA)
    set(CUDA_SRC "tensor_cuda.cu" "shaders.cu")
    set(CUDA_LIBS CUDA::cudart CUDA::cublas)
endif()

# build with or without Vulkan GPU support (AMD, Intel and NVIDIA GPUs). Needs the Vulkan
# headers and glslc (Vulkan SDK, or libvulkan-dev + glslc on Linux) to build. The loader is
# opened at run time, so the result also runs, on the CPU, where Vulkan is not installed.
set(VULKAN_SRC "")
set(VULKAN_LIBS "")
if ( USE_VULKAN )
    find_package(Vulkan REQUIRED)
    if (NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "USE_VULKAN needs glslc (install the Vulkan SDK, or the glslc package)")
    endif()
    message("Vulkan GPU build requested...")
    add_compile_options(-DUSE_VULKAN)
    # compile each shader to SPIR-V, then embed them all in shaders_vk_spv.h
    set(VULKAN_SHADERS gemm bias_act softmax act_deriv softmax_jacobian output_delta_loss
                       bias_grad_sum optimizer dropout
                       conv_im2col conv_output conv_grad_rows conv_col2im
                       maxpool_forward maxpool_backward upsample global_avgpool batchnorm sum_partials
                       conv_direct_forward conv_direct_input_grad conv_direct_weight_grad)
    set(VULKAN_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/vk_spv)
    file(MAKE_DIRECTORY ${VULKAN_SPV_DIR})
    set(VULKAN_SPV_FILES "")
    foreach(shader ${VULKAN_SHADERS})
        add_custom_command(
            OUTPUT ${VULKAN_SPV_DIR}/${shader}.spv
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.1 -O
                    -I ${CMAKE_SOURCE_DIR}/shaders/vk
                    -o ${VULKAN_SPV_DIR}/${shader}.spv ${CMAKE_SOURCE_DIR}/shaders/vk/${shader}.comp
            DEPENDS shaders/vk/${shader}.comp shaders/vk/common.glsl
            COMMENT "Compiling Vulkan shader ${shader}.comp"
            VERBATIM)
        list(APPEND VULKAN_SPV_FILES ${VULKAN_SPV_DIR}/${shader}.spv)
    endforeach()
    # the names go comma-separated: a ';' list would end the command in a Unix shell
    string(REPLACE ";" "," VULKAN_SHADER_NAMES "${VULKAN_SHADERS}")
    add_custom_command(
        OUTPUT ${VULKAN_SPV_DIR}/shaders_vk_spv.h
        COMMAND ${CMAKE_COMMAND} -DSPV_DIR=${VULKAN_SPV_DIR} -DNAMES=${VULKAN_SHADER_NAMES}
                -DOUT=${VULKAN_SPV_DIR}/shaders_vk_spv.h -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
        DEPENDS ${VULKAN_SPV_FILES} cmake/embed_spirv.cmake
        COMMENT "Embedding Vulkan shaders"
        VERBATIM)
    set(VULKAN_SRC tensor_vulkan.c ${VULKAN_SPV_DIR}/shaders_vk_spv.h)
    # dlopen() on Linux/macOS
    set(VULKAN_LIBS ${CMAKE_DL_LIBS})
endif()

# add the core library (includes hypertuning)
set(ANN_LIB_SRC ann.c tensor.c ann_hypertune.c json.c ann_quantize.c ${METAL_SRC} ${CUDA_SRC} ${VULKAN_SRC})
if (BUILD_SHARED)
    add_library(libann SHARED ${ANN_LIB_SRC})
    target_compile_definitions(libann PRIVATE ANN_BUILDING_DLL)
    # export only ANN_API-marked functions (matches Windows dllexport behavior)
    set_target_properties(libann PROPERTIES C_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
else()
    add_library(libann STATIC ${ANN_LIB_SRC})
endif()
if ( USE_VULKAN )
    target_include_directories(libann PRIVATE ${Vulkan_INCLUDE_DIRS} ${VULKAN_SPV_DIR})
endif()
# ann_parallel_for's worker threads (pthreads; nothing extra on Windows)
find_package(Threads REQUIRED)
target_link_libraries(libann ${BLAS_LIB} ${MATH_LIB} ${METAL_LIB} ${CUDA_LIBS} ${VULKAN_LIBS} Threads::Threads)
set_target_properties(libann PROPERTIES PREFIX "")  # avoid liblibann on Unix

# install rules
install(TARGETS libann ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES ann.h ann_rt.h tensor.h ann_config.h ann_hypertune.h json.h ann_gpu_backend.h
        DESTINATION include/ann)

#
# examples
#
#add_executable(blas_perf examples/blas_perf.c)
#target_link_libraries(blas_perf libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist examples/mnist.c)
target_link_libraries(mnist libann ${BLAS_LIB} ${MATH_LIB})

add_executable(fashion_cnn examples/fashion_cnn.c)
target_link_libraries(fashion_cnn libann ${BLAS_LIB} ${MATH_LIB})

add_executable(fashion_dae examples/fashion_dae.c)
target_link_libraries(fashion_dae libann ${BLAS_LIB} ${MATH_LIB})

add_executable(export_model examples/export_model.c)
target_link_libraries(export_model libann ${BLAS_LIB} ${MATH_LIB})

# run_exported builds the headers export_model writes, without libann. Not in
# the default build since the headers exist only after export_model runs:
#   cmake -DEXPORT_HEADER_DIR=<dir with fashion_q.h, fashion_f.h> ..
#   cmake --build . --target run_exported
set(EXPORT_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "Directory holding the headers export_model wrote")
add_executable(run_exported EXCLUDE_FROM_ALL examples/run_exported.c)
target_include_directories(run_exported PRIVATE ${EXPORT_HEADER_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(run_exported ${MATH_LIB})

add_executable(mnist_hypertune_tpe examples/mnist_hypertune_tpe.c)
target_link_libraries(mnist_hypertune_tpe libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist_hypertune_grid examples/mnist_hypertune_grid.c)
target_link_libraries(mnist_hypertune_grid libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist_hypertune_random examples/mnist_hypertune_random.c)
target_link_libraries(mnist_hypertune_random libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist_hypertune_bayes examples/mnist_hypertune_bayes.c)
target_link_libraries(mnist_hypertune_bayes libann ${BLAS_LIB} ${MATH_LIB})

add_executable(digit5x7 examples/digit5x7.c)
target_link_libraries(digit5x7 libann ${BLAS_LIB} ${MATH_LIB})

add_executable(logic examples/logic.c)
target_link_libraries(logic libann ${BLAS_LIB} ${MATH_LIB})

add_executable(save_test examples/save_test.c)
target_link_libraries(save_test libann ${BLAS_LIB} ${MATH_LIB})

add_executable(save_test_binary examples/save_test_binary.c)
target_link_libraries(save_test_binary libann ${BLAS_LIB} ${MATH_LIB})

#
# tests
#
add_executable(test_tensor tests/test_tensor.c testy/test_main.c)
target_link_libraries(test_tensor libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_network tests/test_network.c testy/test_main.c)
target_link_libraries(test_network libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_activations tests/test_activations.c testy/test_main.c)
target_link_libraries(test_activations libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_onnx_export tests/test_onnx_export.c testy/test_main.c)
target_link_libraries(test_onnx_export libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_loss_functions tests/test_loss_functions.c testy/test_main.c)
target_link_libraries(test_loss_functions libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_save_load tests/test_save_load.c testy/test_main.c)
target_link_libraries(test_save_load libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_hypertune tests/test_hypertune.c testy/test_main.c)
target_link_libraries(test_hypertune libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_optimizers tests/test_optimizers.c testy/test_main.c)
target_link_libraries(test_optimizers libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_training_convergence tests/test_training_convergence.c testy/test_main.c)
target_link_libraries(test_training_convergence libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_training_convergence COMMAND test_training_convergence)

add_executable(test_online_training tests/test_online_training.c testy/test_main.c)
target_link_libraries(test_online_training libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_online_training COMMAND test_online_training)

add_executable(test_noise tests/test_noise.c testy/test_main.c)
target_link_libraries(test_noise libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_json tests/test_json.c testy/test_main.c)
target_link_libraries(test_json libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_json COMMAND test_json)

if ( USE_METAL )
    add_executable(test_metal tests/test_metal.c testy/test_main.c)
    target_link_libraries(test_metal libann ${BLAS_LIB} ${MATH_LIB} ${METAL_LIB})
    add_test(NAME test_metal COMMAND test_metal)
endif()

if ( USE_CUDA )
    add_executable(test_cuda tests/test_cuda.c testy/test_main.c)
    target_link_libraries(test_cuda libann ${BLAS_LIB} ${MATH_LIB} ${CUDA_LIBS})
    add_test(NAME test_cuda COMMAND test_cuda)
endif()

if ( USE_VULKAN )
    add_executable(test_vulkan tests/test_vulkan.c testy/test_main.c)
    target_link_libraries(test_vulkan libann ${BLAS_LIB} ${MATH_LIB})
    add_test(NAME test_vulkan COMMAND test_vulkan)
    # again with each of the two ways data reaches the GPU, whichever one this machine's
    # GPU takes by default: copies through a staging buffer (discrete GPUs without resizable
    # BAR) and mapped device memory (integrated GPUs, resizable BAR)
    add_test(NAME test_vulkan_staging COMMAND test_vulkan)
    set_tests_properties(test_vulkan_staging PROPERTIES ENVIRONMENT "ANN_VK_STAGING=1")
    add_test(NAME test_vulkan_mapped COMMAND test_vulkan)
    set_tests_properties(test_vulkan_mapped PROPERTIES ENVIRONMENT "ANN_VK_STAGING=0")
endif()

if ( USE_METAL OR USE_CUDA OR USE_VULKAN )
    add_executable(bench_gpu_training examples/bench_gpu_training.c)
    target_link_libraries(bench_gpu_training libann ${BLAS_LIB} ${MATH_LIB} ${METAL_LIB} ${CUDA_LIBS})
endif()

add_executable(test_forward_pass tests/test_forward_pass.c testy/test_main.c)
target_link_libraries(test_forward_pass libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_gradients tests/test_gradients.c testy/test_main.c)
target_link_libraries(test_gradients libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_comprehensive_callbacks tests/test_comprehensive_callbacks.c testy/test_main.c)
target_link_libraries(test_comprehensive_callbacks libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_training_options tests/test_training_options.c testy/test_main.c)
target_link_libraries(test_training_options libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_conv tests/test_conv.c testy/test_main.c)
target_link_libraries(test_conv libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_batchnorm tests/test_batchnorm.c testy/test_main.c)
target_link_libraries(test_batchnorm libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_threads tests/test_threads.c testy/test_main.c)
target_link_libraries(test_threads libann ${BLAS_LIB} ${MATH_LIB} Threads::Threads)

# test_unload loads a shared copy of libann, starts its thread pool and
# unloads it, as pyann's cffi does at interpreter exit
add_library(ann_unload SHARED ${ANN_LIB_SRC})
target_compile_definitions(ann_unload PRIVATE ANN_BUILDING_DLL)
set_target_properties(ann_unload PROPERTIES C_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
if ( USE_VULKAN )
    target_include_directories(ann_unload PRIVATE ${Vulkan_INCLUDE_DIRS} ${VULKAN_SPV_DIR})
endif()
target_link_libraries(ann_unload ${BLAS_LIB} ${MATH_LIB} ${METAL_LIB} ${CUDA_LIBS} ${VULKAN_LIBS} Threads::Threads)

add_executable(test_unload tests/test_unload.c testy/test_main.c)
add_dependencies(test_unload ann_unload)
target_compile_definitions(test_unload PRIVATE ANN_UNLOAD_DLL="$<TARGET_FILE:ann_unload>")
target_link_libraries(test_unload ${CMAKE_DL_LIBS} Threads::Threads)
add_test(NAME test_unload COMMAND test_unload)
set_tests_properties(test_unload PROPERTIES TIMEOUT 60)

add_executable(test_quantize tests/test_quantize.c testy/test_main.c)
target_link_libraries(test_quantize libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_quantize COMMAND test_quantize)

# C header export: gen_c_export writes model headers with libann, then
# test_c_export compiles them with only ann_rt.h (no libann) and checks they
# give libann's results; test_c_export_int8 does the same with ANN_RT_NO_FLOAT
add_executable(gen_c_export tests/gen_c_export.c)
target_link_libraries(gen_c_export libann ${BLAS_LIB} ${MATH_LIB})
set(C_EXPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/c_export)
set(C_EXPORT_FILES ${C_EXPORT_DIR}/ann_rt.h ${C_EXPORT_DIR}/expected.h
    ${C_EXPORT_DIR}/dense_q.h ${C_EXPORT_DIR}/dense_f.h ${C_EXPORT_DIR}/cnn_q.h ${C_EXPORT_DIR}/cnn_f.h
    ${C_EXPORT_DIR}/ae_q.h ${C_EXPORT_DIR}/ae_f.h ${C_EXPORT_DIR}/gap_q.h ${C_EXPORT_DIR}/gap_f.h
    ${C_EXPORT_DIR}/bn_q.h ${C_EXPORT_DIR}/bn_f.h)
add_custom_command(OUTPUT ${C_EXPORT_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${C_EXPORT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/ann_rt.h ${C_EXPORT_DIR}/ann_rt.h
    COMMAND gen_c_export ${C_EXPORT_DIR}
    DEPENDS gen_c_export ${CMAKE_SOURCE_DIR}/ann_rt.h
    COMMENT "Exporting test models as C headers")
# one target owns the headers: listing them in both tests' sources would run
# the export once per test, and in a parallel build the two runs overwrite
# each other's files
add_custom_target(c_export_headers DEPENDS ${C_EXPORT_FILES})
foreach(target test_c_export test_c_export_int8)
    add_executable(${target} tests/test_c_export.c)
    add_dependencies(${target} c_export_headers)
    target_include_directories(${target} PRIVATE ${C_EXPORT_DIR})
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -std=c99 -Wall -Wextra -pedantic -Werror)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endforeach()
target_link_libraries(test_c_export ${MATH_LIB})
target_compile_definitions(test_c_export_int8 PRIVATE ANN_RT_NO_FLOAT)
