add_executable(testenv1 testenv1.cpp $<TARGET_OBJECTS:probeenvs> $<TARGET_OBJECTS:gymutils> $<TARGET_OBJECTS:gymenv>)
target_compile_features(testenv1 PUBLIC cxx_range_for)
target_link_libraries(testenv1 ${shared_libs} ${pytorch_cpp_rl_lib})

