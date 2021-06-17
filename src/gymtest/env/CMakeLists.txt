add_library(gymenv OBJECT communicator.cpp
							envutils.cpp
							lunarenv.cpp
							airenv.cpp
							)

set(CPPRL_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/include)
include_directories(${CPPRL_INC_PATH})
set(ZMP_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/libzmq/include)
include_directories(${ZMP_INC_PATH})
set(ZMPHPP_INC_PATH  /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/third_party)
include_directories(${ZMPHPP_INC_PATH})
set(MSGPACK_INC_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl/example/lib/msgpack-c/include)
include_directories(${MSGPACK_INC_PATH})

