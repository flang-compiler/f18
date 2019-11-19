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
  MLIRAffineOps
  MLIRLLVMIR
  MLIRLoopOps
  MLIRStandardOps
)

add_llvm_library(FirBbcLib bbc.cc)
target_link_libraries(FirBbcLib ${LIBS} FIRTransforms)

set(EXE_LIBS
  ${LIBS}
  FIRTransforms
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranBurnside
)

add_llvm_tool(bbc bbc.cc)
llvm_update_compile_flags(bbc)
whole_archive_link(bbc ${LIBS})
target_link_libraries(bbc PRIVATE FIR MLIRIR FirBbcLib ${EXE_LIBS} LLVMSupport)
