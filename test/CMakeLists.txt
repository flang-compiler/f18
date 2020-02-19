
# simple helper to add a test.  
function(add_flang_test EXECUTABLE TEST_NAME) 
  message(STATUS "flang: adding test '${TEST_NAME}' (${EXECUTABLE})")
  add_executable(${EXECUTABLE}-test ${EXECUTABLE}.cpp)
  target_compile_features(${EXECUTABLE}-test PRIVATE cxx_std_17)
  target_link_libraries(${EXECUTABLE}-test ${ARGN})
  add_test(${TEST_NAME} ${EXECUTABLE}-test)
endfunction()

add_subdirectory(decimal)
add_subdirectory(evaluate)
add_subdirectory(runtime)
add_subdirectory(semantics)
