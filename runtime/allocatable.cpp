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
#include <cstring>

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

void RTNAME(AllocatableApplyMold)(
    Descriptor &descriptor, const Descriptor &mold) {
  if (descriptor.raw().base_addr) {
    return; // error caught later in AllocatableAllocate().
  }
  descriptor.raw().elem_len = mold.raw().elem_len;
  Terminator terminator{__FILE__, __LINE__};
  int rank{descriptor.rank()};
  RUNTIME_CHECK(terminator, rank == mold.rank());
  if (rank) {
    std::memcpy(&descriptor.GetDimension(0), &mold.GetDimension(0),
        rank * sizeof(Dimension));
  }
  if (auto *addendum{descriptor.Addendum()}) {
    if (const auto *type{addendum->derivedType()}) {
      auto *moldAddendum{mold.Addendum()};
      RUNTIME_CHECK(terminator, moldAddendum);
      auto *moldType{moldAddendum->derivedType()};
      RUNTIME_CHECK(terminator, moldAddendum);
      int lenParms{type->lenParameters()};
      RUNTIME_CHECK(terminator, lenParms == moldType->lenParameters());
      for (int j{0}; j < lenParms; ++j) {
        addendum->SetLenParameterValue(j, moldAddendum->LenParameterValue(j));
      }
    }
  }
  // TODO: cobounds
}

void RTNAME(AllocatableSetBounds)(Descriptor &descriptor, int dim,
    SubscriptValue lower, SubscriptValue upper) {
  descriptor.GetDimension(dim).SetBounds(lower, upper);
  descriptor.SetStrides(); // TODO: call only once
}

void RTNAME(AllocatableSetCoBounds)(
    Descriptor &, int, SubscriptValue, SubscriptValue) {
  // TODO
}

void RTNAME(AllocatableSetDerivedLength)(
    Descriptor &descriptor, int which, SubscriptValue x) {
  auto *addendum{descriptor.Addendum()};
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, addendum);
  addendum->SetLenParameterValue(which, x);
}

int RTNAME(AllocatableCheckLengthParameter)(Descriptor &descriptor, int which,
    SubscriptValue other, bool hasStat, Descriptor *errMsg,
    const char *sourceFile, int sourceLine) {
  if (auto *addendum{descriptor.Addendum()}) {
    if (const auto *type{addendum->derivedType()}) {
      auto value{addendum->LenParameterValue(which)};
      if (value == other) {
        return 0;
      }
      const char *msg{"Length type parameter mismatch on ALLOCATE: %s must be "
                      "%jd, but explicit value is %jd"};
      if (hasStat) {
        return StatAndErrmsg(errMsg, AllocatableLengthTypeParameterMismatch,
            msg, type->lenTypeParameter(which).name(),
            static_cast<std::intmax_t>(value),
            static_cast<std::intmax_t>(other));
      }
      Terminator{sourceFile, sourceLine}.Crash(msg,
          type->lenTypeParameter(which).name(),
          static_cast<std::intmax_t>(value), static_cast<std::intmax_t>(other));
    }
  }
  auto characterLength{descriptor.CharacterLength()};
  if (characterLength == other) {
    return 0;
  }
  const char *msg{"CHARACTER length type parameter mismatch on ALLOCATE: must "
                  "be %jd, but explicit value is %jd"};
  if (hasStat) {
    return StatAndErrmsg(errMsg, AllocatableLengthTypeParameterMismatch, msg,
        static_cast<std::intmax_t>(characterLength),
        static_cast<std::intmax_t>(other));
  }
  Terminator{sourceFile, sourceLine}.Crash(msg,
      static_cast<std::intmax_t>(characterLength),
      static_cast<std::intmax_t>(other));
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
