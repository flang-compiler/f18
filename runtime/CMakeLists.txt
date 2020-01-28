#===-- runtime/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# FIXME: we need to evaluate if we need functionality similar to clang's 
# add_clang_library() macro. 
add_library(FortranRuntime
  ISO_Fortran_binding.cc
  derived-type.cc
  descriptor.cc
  format.cc
  main.cc
  stop.cc
  terminator.cc
  transformational.cc
  type-code.cc
)

target_link_libraries(FortranRuntime
  FortranEvaluate
)
