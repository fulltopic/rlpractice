cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rltest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)


set(WORKSPACE_PATH /home/zf/workspaces/workspace_cpp/)
set(WS_PYTHON_PATH /home/zf/workspaces/workspace_python/)

set(CONDA_PATH /home/zf/anaconda3/envs/pytorch/)
set(CONDA_LIB_PATH ${CONDA_PATH}/lib/)
set(CONDA_LIB64_PATH ${CONDA_PATH}/lib64/)
set(CONDA_INC_PATH ${CONDA_PATH}/include/)

################################### self inc ##########################################
include_directories(./inc)

###################################### conda inc ######################################
include_directories(${CONDA_INC_PATH})

####################################### torch include #################################
set(TORCH_PATH ${CONDA_PATH}/lib/python3.9/site-packages/torch/)
set(TORCH_LIB_PATH ${TORCH_PATH}/lib/)
set(TORCH_INC_PATH ${TORCH_PATH}/include/)
include_directories(${TORCH_INC_PATH})
include_directories(${TORCH_INC_PATH}/torch/csrc/)
include_directories(${TORCH_INC_PATH}/torch/csrc/api/include/)

###################################### protobuf inc ####################################
#tmp solution for protobuf lib version conflict between conda and os
set(PYTORCH_SRC_PATH ${WORKSPACE_PATH}/pytorch/)
set(PROTOBUF_INC_PATH ${PYTORCH_SRC_PATH}/third_party/protobuf/src/)
include_directories(${PROTOBUF_INC_PATH})


###################################### gym inc ##########################################
set(GYM_SRC_PATH ${WS_PYTHON_PATH}/pytorch-cpp-rl/)
set(GYM_EXAMPLE_PATH ${GYM_SRC_PATH}/example/lib/)
set(MSGPACK_INC_PATH ${GYM_EXAMPLE_PATH}/msgpack-c/include/)
include_directories(${MSGPACK_INC_PATH})
set(ZMQ_INC_PATH ${GYM_EXAMPLE_PATH}/libzmq/include/)
include_directories(${ZMQ_INC_PATH})
set(ZMQHPP_INC_PATH ${GYM_SRC_PATH}/example/third_party/)
include_directories(${ZMQHPP_INC_PATH})
set(CPPRL_INC_PATH ${GYM_SRC_PATH}/include/)
include_directories(${CPPRL_INC_PATH})

#################################### tensorboardlog inc ##################################
set(TENSORBOARD_PATH ${WS_PYTHON_PATH}/tensorboard_logger/)
set(TENSORBOARD_INC_PATH ${TENSORBOARD_PATH}/include)
include_directories(${TENSORBOARD_INC_PATH})
include_directories(${TENSORBOARD_PATH}/build/)


#set(MATPLOTCPP_INC_PATH /home/zf/workspaces/workspace_cpp/matplotlib-cpp)
#include_directories(${MATPLOTCPP_INC_PATH})

add_subdirectory(src/dbtools)
add_subdirectory(src/gymtest)
add_subdirectory(src/alg)
add_subdirectory(src/probeenvs)
add_subdirectory(src/a3c)
add_subdirectory(test/)


############################# App ########################################
add_executable(hello src/helloworld.cpp)
target_compile_features(hello PUBLIC cxx_range_for)
target_link_libraries(hello rt)
