
add_executable(testmheap testmheap.cpp)
target_compile_features(testmheap PUBLIC cxx_range_for)
target_link_libraries(testmheap ${shared_libs})


add_executable(testsegtree testsegtree.cpp $<TARGET_OBJECTS:alg>)
target_compile_features(testsegtree PUBLIC cxx_range_for)
target_link_libraries(testsegtree ${shared_libs})