# Infrastructure to build flang driver entry point. Flang driver dependes on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

add_executable(flang-tmp
  main.cpp
)

set_target_properties(flang-tmp
  PROPERTIES
    COMPILE_FLAGS
      -Wno-unused-parameter
)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(LLVM_COMMON_LIBS
  Support
  Target
  Option
)

# Link against LLVM and Clang libraries
target_link_libraries(flang-tmp
  PRIVATE
  ${LLVM_COMMON_LIBS}
  flangFrontend
  -lclangDriver
  -lclangBasic
  ${LLVM_LIBRARY_DIR}
)

install(TARGETS flang-tmp DESTINATION bin)
