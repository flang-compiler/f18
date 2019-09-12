set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Option
  ProfileData
  Support
  )

llvm_add_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  DiagnosticRenderer.cpp
  FrontendOptions.cpp
  LogDiagnosticPrinter.cpp
  TextDiagnostic.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp

  # DEPENDS
  # ClangDriverOptions

  LINK_LIBS
  clangAST
  clangBasic
  clangDriver
  clangEdit
  clangLex
  clangParse
  clangSema
  )

target_link_directories(flangFrontend
  PUBLIC
  ${LLVM_DIR}/../..)

set_target_properties(flangFrontend
  PROPERTIES
  COMPILE_FLAGS -Wno-unused-parameter)
