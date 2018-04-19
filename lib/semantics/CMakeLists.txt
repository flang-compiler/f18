
add_library(FlangSemantics
  attr.cc
  make-types.cc
  resolve-names.cc
  scope.cc
  symbol.cc
  type.cc
  ParseTreeDump.cc
  StatementMap.cc
  pass1.cc
)

target_link_libraries( FlangSemantics
  ${llvm_libs}
#  ${llvm_support}
#  ${llvm_target}
)
