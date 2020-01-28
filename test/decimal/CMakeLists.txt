#===-- test/decimal/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flang_test(quick-sanity Sanity
  FortranDecimal
)

add_flang_test(thorough Thorough
  FortranDecimal
)

