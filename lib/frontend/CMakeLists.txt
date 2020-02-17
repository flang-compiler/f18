set(LLVM_LINK_COMPONENTS
  Option
  Support
  )


llvm_add_library(flangFrontend
  TextDiagnosticPrinter.cpp

  LINK_LIBS
  clangBasic
  clangDriver
  )

include_directories(${LLVM_DIR} ${LLVM_DIR}/../..)

target_include_directories(flangFrontend
  PUBLIC
  ${LLVM_DIR}/../..)

set_target_properties(flangFrontend
  PROPERTIES
  COMPILE_FLAGS -Wno-unused-parameter)
