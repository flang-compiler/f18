#===-- test/evaluate/CMakeLists.txt ----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


add_library(FortranEvaluateTesting
  testing.cpp
  fp-testing.cpp
)

# These routines live in lib/common but we test them here.
add_flang_test(leading-zero-bit-count Leadz
  FortranEvaluateTesting
)
add_flang_test(bit-population-count PopPar
  FortranEvaluateTesting
)
add_flang_test(uint128 UINT128
  FortranEvaluateTesting
)
add_flang_test(expression Expression
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranParser
)

add_flang_test(integer Integer
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

add_flang_test(intrinsics Intrinsics
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  FortranParser
  FortranRuntime
)

add_flang_test(logical Logical
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

# GCC -fno-exceptions breaks the fenv.h interfaces needed to capture
# IEEE exception flags (different use of the word "exception")
# in the actual hardware floating-point status register, so ensure that
# C++ exceptions are enabled for this test.
set_source_files_properties(real.cpp PROPERTIES COMPILE_FLAGS -fexceptions)
add_flang_test(real Real
  FortranEvaluateTesting
  FortranEvaluate
  FortranDecimal
  FortranSemantics
  m
)

add_flang_test(reshape RESHAPE
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranRuntime
)

add_flang_test(ISO-Fortran-binding ISO-binding
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranRuntime
)

add_flang_test(folding folding
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
)

set(FOLDING_TESTS
  folding01.f90
  folding02.f90
  folding03.f90
  folding04.f90
  folding05.f90
  folding06.f90
  folding07.f90
  folding08.f90
  folding09.f90
)


set(TEST_LIBPGMATH "-pgmath=false")
if (LIBPGMATH_DIR)
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    set(TEST_LIBPGMATH "-pgmath=true")
  endif()
endif()

foreach(test ${FOLDING_TESTS})
  add_test(NAME ${test} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_folding.sh
    ${test} ${TEST_LIBPGMATH})
endforeach()

