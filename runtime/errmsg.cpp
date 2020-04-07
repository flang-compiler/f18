//===-- runtime/errmsg.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "errmsg.h"
#include "character.h"
#include "descriptor.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime {

int StatAndErrmsg(Descriptor *errMsg, int statCode) {
  const char *rhs{"generic error"};
  switch (statCode) {
  case GenericError:
  default:
    break;
  case AllocatableAlreadyAllocated:
    rhs = "Object in ALLOCATE statement is already allocated";
    break;
  case AllocatableLengthTypeParameterMismatch:
    rhs = "Explicit length type parameter value mismatch";
    break;
  }
  return StatAndErrmsg(errMsg, statCode, rhs);
}

int StatAndErrmsg(Descriptor *errMsg, int statCode, const char *message, ...) {
  if (errMsg) {
    if (char *lhs{reinterpret_cast<char *>(errMsg->raw().base_addr)}) {
      char buffer[200];
      va_list ap;
      va_start(ap, message);
      vsnprintf(buffer, sizeof buffer, message, ap);
      va_end(ap);
      auto at{RTNAME(CharacterAppend)(
          lhs, errMsg->raw().elem_len, 0, buffer, std::strlen(buffer))};
      RTNAME(CharacterPad)(lhs, errMsg->raw().elem_len, at);
    } else {
      // TODO: Fortran 202X automatic allocation
    }
  }
  return statCode;
}

} // namespace Fortran::runtime
