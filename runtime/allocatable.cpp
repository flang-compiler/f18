//===-- runtime/allocatable.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocatable.h"
#include "errmsg.h"
#include "terminator.h"

namespace Fortran::runtime {
extern "C" {

void RTNAME(AllocatableInitIntrinsic)(Descriptor &descriptor,
    TypeCategory category, int kind, int rank, int /*corank*/) {
  descriptor.Establish(
      category, kind, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitCharacter)(Descriptor &descriptor,
    SubscriptValue length, int kind, int rank, int /*corank*/) {
  descriptor.Establish(
      kind, length, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitDerived)(
    Descriptor &descriptor, const DerivedType &type, int rank, int /*corank*/) {
  descriptor.Establish(type, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

int RTNAME(AllocatableCheckAllocated)(
    const Descriptor &descriptor, Descriptor *errMsg) {
  if (descriptor.raw().base_addr) {
    return StatAndErrmsg(errMsg, AllocatableAlreadyAllocated);
  }
  return 0;
}

void RTNAME(AllocatableAssign)(Descriptor &to, const Descriptor & /*from*/) {}

int RTNAME(MoveAlloc)(Descriptor &to, const Descriptor & /*from*/,
    bool /*hasStat*/, Descriptor * /*errMsg*/, const char * /*sourceFile*/,
    int /*sourceLine*/) {
  // TODO
  return 0;
}

int RTNAME(AllocatableDeallocate)(Descriptor &, bool /*hasStat*/,
    Descriptor * /*errMsg*/, const char * /*sourceFile*/, int /*sourceLine*/) {
  // TODO
  return 0;
}
}
} // namespace Fortran::runtime
