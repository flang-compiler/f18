//===-- runtime/errmsg.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Common handling of STAT= and ERRMSG= specifiers.

namespace Fortran::runtime {
class Descriptor;

// Must be disjoint from the values of the standard-defined codes in
// magic-numbers.h & ISO_FORTRAN_ENV.
enum ProcessorStatCodes {
  GenericError = 1,
  AllocatableAlreadyAllocated = 101,
  AllocatableLengthTypeParameterMismatch,
};

int StatAndErrmsg(Descriptor *errMsg, int statCode = GenericError);
int StatAndErrmsg(Descriptor *errMsg, int statCode, const char *message, ...);
} // namespace Fortran::runtime
