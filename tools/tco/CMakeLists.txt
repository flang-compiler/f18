#===-- tools/tco/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===--#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-error")
set(LIBS
  FIR
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
  MLIRLinalg
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
)

add_llvm_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
whole_archive_link(tco ${LIBS})
target_link_libraries(tco PRIVATE FIR MLIRIR FirTcoLib ${EXE_LIBS} LLVMSupport)
