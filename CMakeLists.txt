cmake_minimum_required(VERSION 3.10)

project(TSP_GA LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11 -g -G)
# set(CMAKE_CXX_FLAGS_DEBUG  "-g")
# set(CMAKE_NVCC_FLAGS_DEBUG  "-g")

add_executable(TSP_GA ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/common.cpp ${PROJECT_SOURCE_DIR}/src/world.cpp ${PROJECT_SOURCE_DIR}/src/ga_gpu.cu)
