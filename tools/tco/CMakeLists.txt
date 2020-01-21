set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-error")
set(LIBS
  FIRDialect
  FortranOptimizer
  MLIRAffineOps
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
)

add_llvm_library(FirTcoLib tco.cpp)
target_link_libraries(FirTcoLib ${LIBS} FIRTransforms)

set(EXE_LIBS
  ${LIBS}
  FIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRLoopToStandard
  MLIREDSC
  MLIRLinalgOps
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
)

add_llvm_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
whole_archive_link(tco ${LIBS})
target_link_libraries(tco PRIVATE FIRDialect MLIRIR FirTcoLib ${EXE_LIBS} LLVMSupport)
