set(TENSORBOARD_PATH /home/zf/workspaces/workspace_cpp/tensorboard_logger/)
set(TENSORBOARD_INC_PATH ${TENSORBOARD_PATH}/include)

set(tensorboard_libs ${TENSORBOARD_PATH}/build/libtensorboard_logger.a)

include_directories(${TENSORBOARD_INC_PATH})
include_directories(${TENSORBOARD_PATH}/build/)

set(PROTO_LIB_PATH /home/zf/workspaces/workspace_cpp/pytorch_1_9/pytorch/build/lib)
set(Protobuf_LIBRARIES ${PROTO_LIB_PATH}/libprotobuf.a)

add_executable(testboard test.cpp)
target_compile_features(testboard PUBLIC cxx_range_for)
target_link_libraries(testboard ${shared_libs} ${tensorboard_libs} ${Protobuf_LIBRARIES})
