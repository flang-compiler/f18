#===-- lib/CMakeLists.txt --------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#------------------------------------------------------------------------------#

if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion -Wno-covered-switch-default")
  endif()
endif()

add_subdirectory(common)
add_subdirectory(decimal)
add_subdirectory(evaluate)
add_subdirectory(lower)
add_subdirectory(optimizer)
add_subdirectory(parser)
add_subdirectory(semantics)
