# Infrastructure to build flang driver entry point. Flang driver dependes on
# LLVM libraries.

add_executable(flang-tmp
  main.cpp
)

set_target_properties(flang-tmp
  PROPERTIES
    COMPILE_FLAGS
      -Wno-unused-parameter
)

llvm_map_components_to_libnames(LLVM_COMMON_LIBS
  Support
  Target
  Option
)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_DIR} ${LLVM_DIR}/../..)

target_link_libraries(flang-tmp
  PRIVATE
  clangFrontend
  clangDriver
  clangBasic
  ${LLVM_COMMON_LIBS}
)

install(TARGETS flang-tmp DESTINATION bin)
