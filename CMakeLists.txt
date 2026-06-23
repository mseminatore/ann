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
#add_test(NAME test_forward_pass COMMAND test_forward_pass)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

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
if ( USE_BLAS OR USE_CBLAS OR USE_MKL)
    message("BLAS library build requested...")
    add_compile_options(-DUSE_BLAS)
    if ( USE_MKL )
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
    # Compile tensor_metal.m as Objective-C
    set_source_files_properties(tensor_metal.m PROPERTIES
        COMPILE_FLAGS "-x objective-c"
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

# add the core library (includes hypertuning)
if (BUILD_SHARED)
    add_library(libann SHARED ann.c tensor.c ann_hypertune.c json.c ${METAL_SRC} ${CUDA_SRC})
    target_compile_definitions(libann PRIVATE ANN_BUILDING_DLL)
else()
    add_library(libann STATIC ann.c tensor.c ann_hypertune.c json.c ${METAL_SRC} ${CUDA_SRC})
endif()
target_link_libraries(libann ${METAL_LIB} ${CUDA_LIBS})
set_target_properties(libann PROPERTIES PREFIX "")  # avoid liblibann on Unix

# install rules
install(TARGETS libann ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES ann.h tensor.h ann_config.h ann_hypertune.h json.h ann_gpu_backend.h
        DESTINATION include/ann)

#
# examples
#
#add_executable(blas_perf examples/blas_perf.c)
#target_link_libraries(blas_perf libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist examples/mnist.c)
target_link_libraries(mnist libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist_hypertune examples/mnist_hypertune.c)
target_link_libraries(mnist_hypertune libann ${BLAS_LIB} ${MATH_LIB})

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

if ( USE_METAL OR USE_CUDA )
    add_executable(bench_gpu_training examples/bench_gpu_training.c)
    target_link_libraries(bench_gpu_training libann ${BLAS_LIB} ${MATH_LIB} ${METAL_LIB} ${CUDA_LIBS})
endif()

#add_executable(test_forward_pass tests/test_forward_pass.c testy/test_main.c)
#target_link_libraries(test_forward_pass ann ${BLAS_LIB} ${MATH_LIB})
