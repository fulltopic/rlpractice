add_executable(testlmdb testseqreader.cpp $<TARGET_OBJECTS:dbtools>)
target_compile_features(testlmdb PUBLIC cxx_range_for)
target_link_libraries(testlmdb ${proto_libs} ${shared_libs})

add_executable(testdbwrite testdbwrite.cpp)
target_compile_features(testdbwrite PUBLIC cxx_range_for)
target_link_libraries(testdbwrite  ${shared_libs})

add_executable(testdbrb testdbrb.cpp $<TARGET_OBJECTS:dbtools>)
target_compile_features(testdbrb PUBLIC cxx_range_for)
target_link_libraries(testdbrb ${proto_libs}  ${shared_libs})