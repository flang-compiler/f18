# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

add_subdirectory(Transforms)

add_llvm_library(FIR
  Attribute.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  InternalNames.cpp
  IteratedDominanceFrontier.cpp
  KindMapping.cpp
  StdConverter.cpp
  Tilikum.cpp
  LINK_LIBS
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIREDSC
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRStandardToLLVM
  MLIRTransforms
  LLVMAsmParser
  LLVMAsmPrinter
  LLVMRemarks
)

add_dependencies(FIR FIROpsIncGen)

install (TARGETS FIR
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
