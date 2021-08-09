set(CPPRL_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/include)
include_directories(${CPPRL_INC_PATH})
set(ZMP_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/libzmq/include)
include_directories(${ZMP_INC_PATH})
set(ZMPHPP_INC_PATH  /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/third_party)
include_directories(${ZMPHPP_INC_PATH})
set(MSGPACK_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/msgpack-c/include)
include_directories(${MSGPACK_INC_PATH})

set(PYTORCH_CPP_RL_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl)
set(PYTORCH_CPP_RL_LIB_PATH ${PYTORCH_CPP_RL_PATH}/example/lib)
set(pytorch_cpp_rl_lib ${PYTORCH_CPP_RL_PATH}/build/libcpprl.a ${PYTORCH_CPP_RL_PATH}/build/example/lib/libzmq/lib/libzmq.a)

add_executable(testenv1 testenv1.cpp $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymenv>)
target_compile_features(testenv1 PUBLIC cxx_range_for)
target_link_libraries(testenv1 ${shared_libs} ${pytorch_cpp_rl_lib})

