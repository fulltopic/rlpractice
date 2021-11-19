set(PYTORCH_CPP_RL_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl)
set(PYTORCH_CPP_RL_LIB_PATH ${PYTORCH_CPP_RL_PATH}/example/lib)
set(pytorch_cpp_rl_lib ${PYTORCH_CPP_RL_PATH}/build/libcpprl.a ${PYTORCH_CPP_RL_PATH}/build/example/lib/libzmq/lib/libzmq.a)

set(PROTO_LIB_PATH /home/zf/workspaces/workspace_cpp/pytorch_1_9/pytorch/build/lib)
set(proto_libs ${PROTO_LIB_PATH}/libprotobuf.a)

#set(OPENBLAS_LIB_PATH /opt/OpenBLAS/lib)
#link_directories(${OPENBLAS_LIB_PATH})

#set(LAPACK_PATH /home/zf/anaconda3/envs/gymtest/lib/)
#link_directories(${LAPACK_PATH})

set(CPPRL_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/include)
include_directories(${CPPRL_INC_PATH})
set(ZMP_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/libzmq/include)
include_directories(${ZMP_INC_PATH})
set(ZMPHPP_INC_PATH  /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/third_party)
include_directories(${ZMPHPP_INC_PATH})
set(MSGPACK_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/msgpack-c/include)
include_directories(${MSGPACK_INC_PATH})

add_executable(testclientconn testclientconn.cpp $<TARGET_OBJECTS:a3c>)
target_compile_features(testclientconn PUBLIC cxx_range_for)
target_link_libraries(testclientconn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testserverconn testserverconn.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testserverconn PUBLIC cxx_range_for)
target_link_libraries(testserverconn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testclienthandle testclienthandle.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testclienthandle PUBLIC cxx_range_for)
target_link_libraries(testclienthandle ${shared_libs} ${pytorch_cpp_rl_lib})