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
#include <cstring>

namespace Fortran::runtime {
int StatAndErrmsg(Descriptor *errMsg, int statCode) {
  if (errMsg) {
    if (char *lhs{reinterpret_cast<char *>(errMsg->raw().base_addr)}) {
      const char *rhs{"generic error"};
      switch (statCode) {
      case AllocatableAlreadyAllocated:
        rhs = "Object in ALLOCATE statement is already allocated";
        break;
      case GenericError:
      default:
        break;
      }
      auto at{RTNAME(CharacterAppend)(
          lhs, errMsg->raw().elem_len, 0, rhs, std::strlen(rhs))};
      RTNAME(CharacterPad)(lhs, errMsg->raw().elem_len, at);
    } else {
      // TODO: Fortran 202X automatic allocation
    }
  }
  return statCode;
}
} // namespace Fortran::runtime
