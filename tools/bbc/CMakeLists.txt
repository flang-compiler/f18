#===-- tools/bbc/CMakeLists.txt --------------------------------------------===#
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
  MLIRLLVMIR
  MLIRLoopOps
  MLIRStandardOps
)

add_llvm_library(FirBbcLib bbc.cpp)
target_link_libraries(FirBbcLib ${LIBS} FIRTransforms)

set(EXE_LIBS
  ${LIBS}
  FIRTransforms
  MLIRAffineToStandard
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
)

add_llvm_tool(bbc bbc.cpp)
llvm_update_compile_flags(bbc)
whole_archive_link(bbc ${LIBS})
target_link_libraries(bbc PRIVATE FIR MLIRIR FirBbcLib ${EXE_LIBS} LLVMSupport)
