cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)


# testing
enable_testing()
add_test(NAME logic COMMAND logic "${CMAKE_SOURCE_DIR}/and.csv")
add_test(NAME digit5x7 COMMAND digit5x7 "${CMAKE_SOURCE_DIR}/num5x7.csv")
add_test(NAME save_test COMMAND save_test "${CMAKE_SOURCE_DIR}/mnist-fashion.nna" "${CMAKE_SOURCE_DIR}/fashion-mnist_test.csv")

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# set compiler and arch specific optimizations
# if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
#   set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS -Wall)
# elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
#   set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS -Wall)
# elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
#   set_property(TARGET foo APPEND PROPERTY COMPILE_OPTIONS /W3)

# link math lib for gcc/linux
set(MATH_LIB "")
if ( LINUX )
    message("Linux build: add -lm to linker options")
    set(MATH_LIB "m")
endif()

# build with or without BLAS library
if ( USE_BLAS )
    message("BLAS library build requested...")
    add_compile_options(-DUSE_BLAS)
    set(BLAS_LIB "openblas")
    if ( WIN32 )
        include_directories("./cblas/include")
        link_directories("/opt/lib")
    else()
        include_directories("/opt/OpenBLAS/include")
        link_directories("/opt/OpenBLAS/lib")
    endif()
else()
    set(BLAS_LIB "")
endif()

# add the library
add_library(ann STATIC ann.c tensor.c)

# add the executables
add_executable(mnist mnist.c)
target_link_libraries(mnist ann ${BLAS_LIB} ${MATH_LIB})

add_executable(digit5x7 digit5x7.c)
target_link_libraries(digit5x7 ann ${BLAS_LIB} ${MATH_LIB})

add_executable(logic logic.c)
target_link_libraries(logic ann ${BLAS_LIB} ${MATH_LIB})

add_executable(save_test save_test.c)
target_link_libraries(save_test ann ${BLAS_LIB} ${MATH_LIB})
