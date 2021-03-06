#set(OPENBLAS_LIB_PATH /opt/OpenBLAS/lib)
#link_directories(${OPENBLAS_LIB_PATH})

#set(LAPACK_PATH /home/zf/anaconda3/envs/gymtest/lib/)
#link_directories(${LAPACK_PATH})

set(GYM_SIMPLE_OBJS $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain>)
set(GYM_OBJS $<TARGET_OBJECTS:gymenv> ${GYM_SIMPLE_OBJS})
set(NET_OBJS $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets>)
set(RNNNET_OBJS $<TARGET_OBJECTS:lunarrnnnets> $<TARGET_OBJECTS:airrnnnets>)
set(COMMON_TRAIN_OBJS  ${GYM_OBJS} ${NET_OBJS} $<TARGET_OBJECTS:alg>)


######################################################### TOOLS
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

########################################################## DQN
add_executable(testdqn testdqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdqn PUBLIC cxx_range_for)
target_link_libraries(testdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testdoubledqn testdoubledqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdoubledqn PUBLIC cxx_range_for)
target_link_libraries(testdoubledqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testdueldqn testdueldqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdueldqn PUBLIC cxx_range_for)
target_link_libraries(testdueldqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testpriodqn testpriodqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testpriodqn PUBLIC cxx_range_for)
target_link_libraries(testpriodqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testnoisydqn testnoisydqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testnoisydqn PUBLIC cxx_range_for)
target_link_libraries(testnoisydqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testcatdqn testcatdqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testcatdqn PUBLIC cxx_range_for)
target_link_libraries(testcatdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})


############################################################ A2C
add_executable(testa2c testa2c.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testa2c PUBLIC cxx_range_for)
target_link_libraries(testa2c ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testreset testreset.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testreset PUBLIC cxx_range_for)
target_link_libraries(testreset ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

############################################################ PPO
add_executable(testppo testppo.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testppo PUBLIC cxx_range_for)
target_link_libraries(testppo ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

######################################################## SAC
add_executable(testsac testsac.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testsac PUBLIC cxx_range_for)
target_link_libraries(testsac ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testsaczip testsaczip.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testsaczip PUBLIC cxx_range_for)
target_link_libraries(testsaczip ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})


######################################################## Sync
add_executable(testmodelsync testmodelsync.cpp $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils>)
target_compile_features(testmodelsync PUBLIC cxx_range_for)
target_link_libraries(testmodelsync ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})


######################################################## A3C
add_executable(testa3c testa3c.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:a3c>)
target_compile_features(testa3c PUBLIC cxx_range_for)
target_link_libraries(testa3c ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testappo testappo.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:a3c> )
target_compile_features(testappo PUBLIC cxx_range_for)
target_link_libraries(testappo ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

############################################################ RNN
add_executable(testpackpad testpackpad.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testpackpad PUBLIC cxx_range_for)
target_link_libraries(testpackpad ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

############################################################# RNN A2C
add_executable(testa2crnn testa2crnn.cpp ${COMMON_TRAIN_OBJS}  ${RNNNET_OBJS})
target_compile_features(testa2crnn PUBLIC cxx_range_for)
target_link_libraries(testa2crnn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testa2cgrutrunc testa2cgrutrunc.cpp ${COMMON_TRAIN_OBJS}  ${RNNNET_OBJS})
target_compile_features(testa2cgrutrunc PUBLIC cxx_range_for)
target_link_libraries(testa2cgrutrunc ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

############################################################ RNN PPO
add_executable(testppornn testppornn.cpp ${COMMON_TRAIN_OBJS}  ${RNNNET_OBJS})
target_compile_features(testppornn PUBLIC cxx_range_for)
target_link_libraries(testppornn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

############################################################ RNN APPO
add_executable(testappornn testappornn.cpp ${COMMON_TRAIN_OBJS}  ${RNNNET_OBJS})
target_compile_features(testappornn PUBLIC cxx_range_for)
target_link_libraries(testappornn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

