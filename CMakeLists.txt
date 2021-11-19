cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rltest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)


set(CONDA_PATH /home/zf/anaconda3/lib/python3.8/site-packages/torch/)
set(CONDA_LIB_PATH ${CONDA_PATH}/lib/)
set(CONDA_INC_PATH ${CONDA_PATH}/include/)

set(shared_libs torch torch_cpu c10 torch_cuda c10_cuda rt pthread log4cxx)
link_directories(${CONDA_LIB_PATH})

include_directories(./inc)
#tmp solution for protobuf lib version conflict between conda and os
include_directories(/home/zf/workspaces/workspace_cpp/pytorch_1_9/pytorch/third_party/protobuf/src/)
include_directories(${CONDA_INC_PATH})
include_directories(${CONDA_INC_PATH}/torch/csrc/)
include_directories(${CONDA_INC_PATH}/torch/csrc/api/include/)

set(MATPLOTCPP_INC_PATH /home/zf/workspaces/workspace_cpp/matplotlib-cpp)
include_directories(${MATPLOTCPP_INC_PATH})

add_subdirectory(src/dbtools)
add_subdirectory(src/gymtest)
add_subdirectory(src/alg)
add_subdirectory(src/probeenvs)
add_subdirectory(src/a3c)
add_subdirectory(test/)




############################# App ########################################
add_executable(hello src/helloworld.cpp)
target_compile_features(hello PUBLIC cxx_range_for)
target_link_libraries(hello ${shared_libs})
