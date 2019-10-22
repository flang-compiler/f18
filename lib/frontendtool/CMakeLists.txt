set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(link_libs
  clangBasic
  clangDriver
  flangFrontend
  )

llvm_add_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  # DEPENDS
  # ClangDriverOptions

  LINK_LIBS
  ${link_libs}
  )

set_target_properties(flangFrontendTool
  PROPERTIES
  COMPILE_FLAGS -Wno-unused-parameter)
