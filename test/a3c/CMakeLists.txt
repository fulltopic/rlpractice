add_executable(testclientconn testclientconn.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testclientconn PUBLIC cxx_range_for)
target_link_libraries(testclientconn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testserverconn testserverconn.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testserverconn PUBLIC cxx_range_for)
target_link_libraries(testserverconn ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testclienthandle testclienthandle.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testclienthandle PUBLIC cxx_range_for)
target_link_libraries(testclienthandle ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})

add_executable(testserverhandle testserverhandle.cpp $<TARGET_OBJECTS:a3c>  $<TARGET_OBJECTS:lunarnets> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:airnets>)
target_compile_features(testserverhandle PUBLIC cxx_range_for)
target_link_libraries(testserverhandle ${shared_libs} ${pytorch_cpp_rl_lib} ${tensorboard_libs} ${proto_libs})