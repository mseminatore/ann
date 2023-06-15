cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# link math lib for gcc/linux
if ( LINUX )
    message("Linux build: add -lm to linker options")
    add_link_options("-lm")
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
target_link_libraries(mnist ann ${BLAS_LIB})

add_executable(digit5x7 digit5x7.c)
target_link_libraries(digit5x7 ann ${BLAS_LIB})

add_executable(logic logic.c)
target_link_libraries(logic ann ${BLAS_LIB})

add_executable(save_test save_test.c)
target_link_libraries(save_test ann ${BLAS_LIB})
