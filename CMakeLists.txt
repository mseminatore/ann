cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)

# options
option(BUILD_SHARED "Build shared library (DLL/so/dylib)" OFF)

# testing
enable_testing()
add_test(NAME logic COMMAND logic "${CMAKE_SOURCE_DIR}/and.csv")
add_test(NAME digit5x7 COMMAND digit5x7 "${CMAKE_SOURCE_DIR}/num5x7.csv")
add_test(NAME save_test COMMAND save_test "${CMAKE_SOURCE_DIR}/mnist-fashion.nna" "${CMAKE_SOURCE_DIR}/fashion-mnist_test.csv")
add_test(NAME save_test_binary COMMAND save_test_binary "${CMAKE_SOURCE_DIR}/mnist-fashion.nnb" "${CMAKE_SOURCE_DIR}/fashion-mnist_test.csv")
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
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options(/arch:AVX)
endif()

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

# build with or without BLAS library
if ( USE_BLAS OR USE_CBLAS)
    message("BLAS library build requested...")
    add_compile_options(-DUSE_BLAS)
    if ( USE_CBLAS )
        add_compile_options(-DUSE_CBLAS)
        set(BLAS_LIB "cblas")
        message("CBLAS library used")
        # CBLAS requires C11 atomics
        if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
            add_compile_options(/std:c11 /experimental:c11atomics)
            include_directories("C:/opt/cblas/include")
            link_directories("C:/opt/cblas/lib")
        else()
            add_compile_options(-std=c11)
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

# add the core library (includes hypertuning)
if (BUILD_SHARED)
    add_library(libann SHARED ann.c tensor.c ann_hypertune.c json.c)
    target_compile_definitions(libann PRIVATE ANN_BUILDING_DLL)
else()
    add_library(libann STATIC ann.c tensor.c ann_hypertune.c json.c)
endif()
set_target_properties(libann PROPERTIES PREFIX "")  # avoid liblibann on Unix

#
# add the executables
#
#add_executable(blas_perf blas_perf.c)
#target_link_libraries(blas_perf libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist mnist.c)
target_link_libraries(mnist libann ${BLAS_LIB} ${MATH_LIB})

add_executable(mnist_hypertune mnist_hypertune.c)
target_link_libraries(mnist_hypertune libann ${BLAS_LIB} ${MATH_LIB})

add_executable(digit5x7 digit5x7.c)
target_link_libraries(digit5x7 libann ${BLAS_LIB} ${MATH_LIB})

add_executable(logic logic.c)
target_link_libraries(logic libann ${BLAS_LIB} ${MATH_LIB})

add_executable(save_test save_test.c)
target_link_libraries(save_test libann ${BLAS_LIB} ${MATH_LIB})

add_executable(save_test_binary save_test_binary.c)
target_link_libraries(save_test_binary libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_tensor test_tensor.c testy/test_main.c)
target_link_libraries(test_tensor libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_network test_network.c testy/test_main.c)
target_link_libraries(test_network libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_activations test_activations.c testy/test_main.c)
target_link_libraries(test_activations libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_onnx_export test_onnx_export.c testy/test_main.c)
target_link_libraries(test_onnx_export libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_loss_functions test_loss_functions.c testy/test_main.c)
target_link_libraries(test_loss_functions libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_save_load test_save_load.c testy/test_main.c)
target_link_libraries(test_save_load libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_hypertune test_hypertune.c testy/test_main.c)
target_link_libraries(test_hypertune libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_optimizers test_optimizers.c testy/test_main.c)
target_link_libraries(test_optimizers libann ${BLAS_LIB} ${MATH_LIB})

add_executable(test_training_convergence test_training_convergence.c testy/test_main.c)
target_link_libraries(test_training_convergence libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_training_convergence COMMAND test_training_convergence)

add_executable(test_json test_json.c testy/test_main.c)
target_link_libraries(test_json libann ${BLAS_LIB} ${MATH_LIB})
add_test(NAME test_json COMMAND test_json)

#add_executable(test_forward_pass test_forward_pass.c testy/test_main.c)
#target_link_libraries(test_forward_pass ann ${BLAS_LIB} ${MATH_LIB})

