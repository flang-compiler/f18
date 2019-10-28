# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LIBS
  FIR
  FIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)

add_llvm_library(FirTcoLib tco.cpp)
target_link_libraries(FirTcoLib ${LIBS})

set(EXE_LIBS
  ${LIBS}
  MLIRAffineOps
  MLIRLoopToStandard
  MLIREDSC
  MLIRLinalg
  MLIRLoopOps
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRVectorOps
  MLIRVectorToLLVM
)

add_llvm_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
whole_archive_link(tco ${EXE_LIBS})
target_link_libraries(tco PRIVATE FIR MLIRIR FirTcoLib ${EXE_LIBS} LLVMSupport)
