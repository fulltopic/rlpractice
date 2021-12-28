set (PYTORCH_SRC_LIB_PATH ${PYTORCH_SRC_PATH}/build/lib)

################################### shared libs ########################################
set(shared_libs torch torch_cpu c10 torch_cuda c10_cuda rt pthread log4cxx)
link_directories(${TORCH_LIB_PATH})
link_directories(${CONDA_LIB_PATH})
link_directories(${CONDA_LIB64_PATH})

################################### tensorboard libs ####################################
set(tensorboard_libs ${TENSORBOARD_PATH}/build/libtensorboard_logger.a)

######################################### protobuf ######################################
set(proto_libs ${PYTORCH_SRC_LIB_PATH}/libprotobuf.a)

######################################## gym libs #######################################
set(PYTORCH_CPP_RL_LIB_PATH ${PYTORCH_CPP_RL_PATH}/example/lib)
set(GYM_SRC_BUILD_PATH ${GYM_SRC_PATH}/build)
set(GYM_SRC_EXAMPLE_BUILD_PATH ${GYM_SRC_BUILD_PATH}/example/lib)
set(pytorch_cpp_rl_lib ${GYM_SRC_BUILD_PATH}/libcpprl.a ${GYM_SRC_EXAMPLE_BUILD_PATH}/libzmq/lib/libzmq.a)

add_subdirectory(gymtest)
add_subdirectory(trials)
#add_subdirectory(lmdb)
add_subdirectory(envtest/)
add_subdirectory(testutils)
add_subdirectory(tensorboard)
add_subdirectory(a3c)