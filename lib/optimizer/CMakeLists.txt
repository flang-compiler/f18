#===-- lib/fir/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===--#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_subdirectory(Transforms)

add_llvm_library(FIR
  Attribute.cpp
  CodeGen.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
  InternalNames.cpp
  IteratedDominanceFrontier.cpp
  KindMapping.cpp
  StdConverter.cpp
)

add_dependencies(FIR FIROpsIncGen)

target_link_libraries(FIR
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

install (TARGETS FIR
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
