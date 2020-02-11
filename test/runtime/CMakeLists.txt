
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

add_executable(format-test
  format.cpp
)

target_link_libraries(format-test
  FortranRuntime
)

add_test(Format format-test)

add_executable(hello-world
  hello.cpp
)

target_link_libraries(hello-world
  FortranRuntime
)

add_test(HelloWorld hello-world)
