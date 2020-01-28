#===-- lib/CMakeLists.txt --------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(common)
add_subdirectory(evaluate)
add_subdirectory(decimal)
add_subdirectory(parser)
add_subdirectory(semantics)

# FIXME: Added install of headers to give a directory structure 
# (upon install) that follows aspects of the Clang layout.  If 
# we decide to move header files out of lib this install target
# will need to be moved elsewhere... 
if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY common evaluate decimal parser semantics 
    DESTINATION include/flang
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".git"     EXCLUDE
    PATTERN ".svn"     EXCLUDE)
endif()
