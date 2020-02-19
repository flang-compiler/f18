
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

add_flang_test(format format-test
  FortranRuntime
)

add_flang_test(hello HelloWorld
  FortranRuntime)

add_flang_test(external-hello ExternalHelloWorld
  FortranRuntime)
