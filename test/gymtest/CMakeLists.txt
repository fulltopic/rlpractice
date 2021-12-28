#set(OPENBLAS_LIB_PATH /opt/OpenBLAS/lib)
#link_directories(${OPENBLAS_LIB_PATH})

#set(LAPACK_PATH /home/zf/anaconda3/envs/gymtest/lib/)
#link_directories(${LAPACK_PATH})

set(GYM_SIMPLE_OBJS $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymtrain>)
set(GYM_OBJS $<TARGET_OBJECTS:gymenv> ${GYM_SIMPLE_OBJS})
set(NET_OBJS $<TARGET_OBJECTS:airnets> $<TARGET_OBJECTS:lunarnets>)
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


############################################################ A2C
add_executable(testa2c testa2c.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testa2c PUBLIC cxx_range_for)
target_link_libraries(testa2c ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testa2cnstep testa2cnstep.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testa2cnstep PUBLIC cxx_range_for)
target_link_libraries(testa2cnstep ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testbreakout testbreakout.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testbreakout PUBLIC cxx_range_for)
target_link_libraries(testbreakout ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testa2cnbatch testa2cnbatch.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(testa2cnbatch PUBLIC cxx_range_for)
target_link_libraries(testa2cnbatch ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(a2cnbatchpong a2cnbatchpong.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(a2cnbatchpong PUBLIC cxx_range_for)
target_link_libraries(a2cnbatchpong ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(a2cnbreakout a2cnbreakout.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(a2cnbreakout PUBLIC cxx_range_for)
target_link_libraries(a2cnbreakout ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(a2cngae a2cngae.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(a2cngae PUBLIC cxx_range_for)
target_link_libraries(a2cngae ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testrtnorm testrtnorm.cpp ${GYM_SIMPLE_OBJS})
target_compile_features(testrtnorm PUBLIC cxx_range_for)
target_link_libraries(testrtnorm ${shared_libs})


############################################################ PPO
add_executable(ppopong ppopong.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(ppopong PUBLIC cxx_range_for)
target_link_libraries(ppopong ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(ppobr ppobr.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(ppobr PUBLIC cxx_range_for)
target_link_libraries(ppobr ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(ppobr2 ppobr2.cpp ${COMMON_TRAIN_OBJS})
target_compile_features(ppobr2 PUBLIC cxx_range_for)
target_link_libraries(ppobr2 ${shared_libs} ${pytorch_cpp_rl_lib} ${proto_libs})

add_executable(testprobeenv testprobeenv.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testprobeenv PUBLIC cxx_range_for)
target_link_libraries(testprobeenv ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})


########################################################## DQN
add_executable(testdqn testdqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdqn PUBLIC cxx_range_for)
target_link_libraries(testdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testdoubledqn testdoubledqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdoubledqn PUBLIC cxx_range_for)
target_link_libraries(testdoubledqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testdueldqn testdueldqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testdueldqn PUBLIC cxx_range_for)
target_link_libraries(testdueldqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testpriodqn testpriodqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs>)
target_compile_features(testpriodqn PUBLIC cxx_range_for)
target_link_libraries(testpriodqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testnoisydqn testnoisydqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testnoisydqn PUBLIC cxx_range_for)
target_link_libraries(testnoisydqn ${shared_libs} ${pytorch_cpp_rl_lib})

add_executable(testcatdqn testcatdqn.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:noisynets>)
target_compile_features(testcatdqn PUBLIC cxx_range_for)
target_link_libraries(testcatdqn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})


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

add_executable(testappo testappo.cpp ${COMMON_TRAIN_OBJS} $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:a3c> $<TARGET_OBJECTS:algappo>)
target_compile_features(testappo PUBLIC cxx_range_for)
target_link_libraries(testappo ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})
