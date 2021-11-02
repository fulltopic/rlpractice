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

############################ Tensorboard ##################################
set(TENSORBOARD_PATH /home/zf/workspaces/workspace_cpp/tensorboard_logger/)
set(TENSORBOARD_INC_PATH ${TENSORBOARD_PATH}/include)
set(tensorboard_libs ${TENSORBOARD_PATH}/build/libtensorboard_logger.a)
include_directories(${TENSORBOARD_INC_PATH})
include_directories(${TENSORBOARD_PATH}/build/)
#set(PROTO_LIB_PATH /home/zf/workspaces/workspace_cpp/pytorch_1_9/pytorch/build/lib)
#set(Protobuf_LIBRARIES ${PROTO_LIB_PATH}/libprotobuf.a)

add_executable(gymclient gymclient.cpp $<TARGET_OBJECTS:gymenv>)
target_compile_features(gymclient PUBLIC cxx_range_for)
target_link_libraries(gymclient ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testsimplenet testsimplenet.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils>)
target_compile_features(testsimplenet PUBLIC cxx_range_for)
target_link_libraries(testsimplenet ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testairenv testairenv.cpp $<TARGET_OBJECTS:gymenv>)
target_compile_features(testairenv PUBLIC cxx_range_for)
target_link_libraries(testairenv ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testaircnn testaircnn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils>)
target_compile_features(testaircnn PUBLIC cxx_range_for)
target_link_libraries(testaircnn ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testmodelcopy testmodelcopy.cpp $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils>)
#target_compile_features(testmodelcopy PUBLIC cxx_range_for)
#target_link_libraries(testmodelcopy ${shared_libs})

#add_executable(testrawdqn testrawdqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils>)
#target_compile_features(testrawdqn PUBLIC cxx_range_for)
#target_link_libraries(testrawdqn ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testdqn dqntest.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg>)
#target_compile_features(testdqn PUBLIC cxx_range_for)
#target_link_libraries(testdqn ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testprdqn testprdqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets>  $<TARGET_OBJECTS:lunarnets>  $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg>)
#target_compile_features(testprdqn PUBLIC cxx_range_for)
#target_link_libraries(testprdqn ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testddqn ddqntest.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg>)
#target_compile_features(testddqn PUBLIC cxx_range_for)
#target_link_libraries(testddqn ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testduel dueltest.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg>)
#target_compile_features(testduel PUBLIC cxx_range_for)
#target_link_libraries(testduel ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testcart testcart.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg>)
#target_compile_features(testcart PUBLIC cxx_range_for)
#target_link_libraries(testcart ${shared_libs} ${pytorch_cpp_rl_lib})

#add_executable(testdbdqn testdbdqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
#target_compile_features(testdbdqn PUBLIC cxx_range_for)
#target_link_libraries(testdbdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})


############################################################ A2C
add_executable(testa2c testa2c.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(testa2c PUBLIC cxx_range_for)
target_link_libraries(testa2c ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testa2cnstep testa2cnstep.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(testa2cnstep PUBLIC cxx_range_for)
target_link_libraries(testa2cnstep ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testbreakout testbreakout.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(testbreakout PUBLIC cxx_range_for)
target_link_libraries(testbreakout ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testa2cnbatch testa2cnbatch.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(testa2cnbatch PUBLIC cxx_range_for)
target_link_libraries(testa2cnbatch ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(a2cnbatchpong a2cnbatchpong.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(a2cnbatchpong PUBLIC cxx_range_for)
target_link_libraries(a2cnbatchpong ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(a2cnbreakout a2cnbreakout.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(a2cnbreakout PUBLIC cxx_range_for)
target_link_libraries(a2cnbreakout ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(a2cngae a2cngae.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(a2cngae PUBLIC cxx_range_for)
target_link_libraries(a2cngae ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testrtnorm testrtnorm.cpp $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain>)
target_compile_features(testrtnorm PUBLIC cxx_range_for)
target_link_libraries(testrtnorm ${shared_libs})

#add_executable(testpriorb testpriorb.cpp $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain>)
#target_compile_features(testpriorb PUBLIC cxx_range_for)
#target_link_libraries(testpriorb ${shared_libs})


############################################################ PPO
add_executable(ppopong ppopong.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(ppopong PUBLIC cxx_range_for)
target_link_libraries(ppopong ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})


add_executable(ppobr ppobr.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(ppobr PUBLIC cxx_range_for)
target_link_libraries(ppobr ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(ppobr2 ppobr2.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools>)
target_compile_features(ppobr2 PUBLIC cxx_range_for)
target_link_libraries(ppobr2 ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testprobeenv testprobeenv.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:dbtools> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testprobeenv PUBLIC cxx_range_for)
target_link_libraries(testprobeenv ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

########################################################## DQN
add_executable(testdqn testdqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdqn PUBLIC cxx_range_for)
target_link_libraries(testdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testdoubledqn testdoubledqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdoubledqn PUBLIC cxx_range_for)
target_link_libraries(testdoubledqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testdueldqn testdueldqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdueldqn PUBLIC cxx_range_for)
target_link_libraries(testdueldqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testpriodqn testpriodqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testpriodqn PUBLIC cxx_range_for)
target_link_libraries(testpriodqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testnoisydqn testnoisydqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testnoisydqn PUBLIC cxx_range_for)
target_link_libraries(testnoisydqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testcatdqn testcatdqn.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testcatdqn PUBLIC cxx_range_for)
target_link_libraries(testcatdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

######################################################## SAC
add_executable(testsac testsac.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testsac PUBLIC cxx_range_for)
target_link_libraries(testsac ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testsaczip testsaczip.cpp $<TARGET_OBJECTS:gymenv> $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain> $<TARGET_OBJECTS:alg> $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testsaczip PUBLIC cxx_range_for)
target_link_libraries(testsaczip ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})


######################################################## Sync
add_executable(testmodelsync testmodelsync.cpp $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils>)
target_compile_features(testmodelsync PUBLIC cxx_range_for)
target_link_libraries(testmodelsync ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

