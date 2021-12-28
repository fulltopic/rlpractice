add_executable(testboard test.cpp)
target_compile_features(testboard PUBLIC cxx_range_for)
target_link_libraries(testboard ${shared_libs} ${tensorboard_libs} ${proto_libs})
