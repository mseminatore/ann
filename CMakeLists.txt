cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ann VERSION 1.0)

# add the includes
include_directories(${PROJECT_SOURCE_DIR})

# add the library
add_library(libann STATIC ann.c tensor.c)

# add the executables
add_executable(ann main.c)
target_link_libraries(ann libann)

add_executable(test5x7 test5x7.c)
target_link_libraries(test5x7 libann)

add_executable(logic logic.c)
target_link_libraries(logic libann)

add_executable(save_test save_test.c)
target_link_libraries(save_test libann)

add_executable(pima pima.c)
target_link_libraries(pima libann)
