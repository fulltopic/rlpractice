add_executable(testtensor testtensor.cpp)
target_compile_features(testtensor PUBLIC cxx_range_for)
target_link_libraries(testtensor ${shared_libs})

add_executable(testrandom testrandom.cpp)
target_compile_features(testrandom PUBLIC cxx_range_for)
target_link_libraries(testrandom ${shared_libs})

add_executable(testgrad testgrad.cpp)
target_compile_features(testgrad PUBLIC cxx_range_for)
target_link_libraries(testgrad ${shared_libs})

add_executable(testcpp testcpp.cpp)
target_compile_features(testcpp PUBLIC cxx_range_for)
target_link_libraries(testcpp rt)

add_executable(testrawpolicy testrawpolicy.cpp)
target_compile_features(testrawpolicy PUBLIC cxx_range_for)
target_link_libraries(testrawpolicy ${shared_libs})
