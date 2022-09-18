cmake_minimum_required(VERSION 3.10)

project(TSP_GA LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

cuda_add_executable(TSP_GA ${PROJECT_SOURCE_DIR}/src/TSP_GA.cu OPTIONS -O2 --std=c++11 --expt-extended-lambda
)

# cuda_add_executable(TSP_GA ${PROJECT_SOURCE_DIR}/test.cu OPTIONS -g -G --std=c++11 --expt-extended-lambda
# )
set_target_properties(
    TSP_GA
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)
# target_include_directories(TSP_GA PUBLIC ${PROJECT_SOURCE_DIR}/src)
