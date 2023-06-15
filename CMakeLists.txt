cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

if ( WIN32 )
    include_directories("./cblas/include")
    link_directories("/opt/lib")
else()
    include_directories("/opt/OpenBLAS/include")
    link_directories("/opt/OpenBLAS/lib")
endif()

# add the library
add_library(ann STATIC ann.c tensor.c)

# add the executables
add_executable(mnist mnist.c)
target_link_libraries(mnist ann openblas)

add_executable(digit5x7 digit5x7.c)
target_link_libraries(digit5x7 ann openblas)

add_executable(logic logic.c)
target_link_libraries(logic ann openblas)

add_executable(save_test save_test.c)
target_link_libraries(save_test ann openblas)
