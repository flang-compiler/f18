#===-- lib/lower/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===--#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_library(FortranLower
  ast-builder.cc
  bridge.cc
  builder.cc
  convert-expr.cc
  convert-type.cc
  intrinsics.cc
  mangler.cc
  runtime.cc
  io.cc
)

target_link_libraries(FortranLower
  FIR
  MLIRAffineOps
  MLIRLLVMIR
  MLIRStandardOps
)

install (TARGETS FortranLower
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
