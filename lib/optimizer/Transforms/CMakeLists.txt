#===-- lib/fir/Transforms/CMakeLists.txt -----------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===--#

add_llvm_library(FIRTransforms
  CSE.cpp
  MemToReg.cpp
  RewriteLoop.cpp
)

add_dependencies(FIRTransforms FIROpsIncGen)

target_link_libraries(FIRTransforms
  FIR
)

install (TARGETS FIRTransforms
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
