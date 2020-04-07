//===-- runtime/descriptor.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "memory.h"
#include "terminator.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace Fortran::runtime {

Descriptor::Descriptor(const Descriptor &that) {
  std::memcpy(this, &that, that.SizeInBytes());
}

Descriptor::~Descriptor() {
  if (raw_.attribute != CFI_attribute_pointer) {
    Deallocate();
  }
}

void Descriptor::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator,
      ISO::CFI_establish(&raw_, p, attribute, t.raw(), elementBytes, rank,
          extent) == CFI_SUCCESS);
  raw_.flags_ = addendum ? AddendumPresent : 0;
  DescriptorAddendum *a{Addendum()};
  RUNTIME_CHECK(terminator, addendum == (a != nullptr));
  if (a) {
    new (a) DescriptorAddendum{};
  }
}

void Descriptor::Establish(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  std::size_t elementBytes = kind;
  if (c == TypeCategory::Complex) {
    elementBytes *= 2;
  }
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator,
      ISO::CFI_establish(&raw_, p, attribute, TypeCode(c, kind).raw(),
          elementBytes, rank, extent) == CFI_SUCCESS);
  raw_.flags_ = addendum ? AddendumPresent : 0;
  DescriptorAddendum *a{Addendum()};
  RUNTIME_CHECK(terminator, addendum == (a != nullptr));
  if (a) {
    new (a) DescriptorAddendum{};
  }
}

void Descriptor::Establish(const DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator,
      ISO::CFI_establish(&raw_, p, attribute, CFI_type_struct, dt.SizeInBytes(),
          rank, extent) == CFI_SUCCESS);
  raw_.flags_ = AddendumPresent;
  DescriptorAddendum *a{Addendum()};
  RUNTIME_CHECK(terminator, a);
  new (a) DescriptorAddendum{&dt};
}

OwningPtr<Descriptor> Descriptor::Create(TypeCode t, std::size_t elementBytes,
    void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Terminator terminator{__FILE__, __LINE__};
  Descriptor *result{
      reinterpret_cast<Descriptor *>(AllocateMemoryOrCrash(terminator, bytes))};
  result->Establish(t, elementBytes, p, rank, extent, attribute, true);
  return OwningPtr<Descriptor>{result};
}

OwningPtr<Descriptor> Descriptor::Create(TypeCategory c, int kind, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Terminator terminator{__FILE__, __LINE__};
  Descriptor *result{
      reinterpret_cast<Descriptor *>(AllocateMemoryOrCrash(terminator, bytes))};
  result->Establish(c, kind, p, rank, extent, attribute, true);
  return OwningPtr<Descriptor>{result};
}

OwningPtr<Descriptor> Descriptor::Create(const DerivedType &dt, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true, dt.lenParameters())};
  Terminator terminator{__FILE__, __LINE__};
  Descriptor *result{
      reinterpret_cast<Descriptor *>(AllocateMemoryOrCrash(terminator, bytes))};
  result->Establish(dt, p, rank, extent, attribute);
  return OwningPtr<Descriptor>{result};
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

std::size_t Descriptor::Elements() const {
  int n{rank()};
  std::size_t elements{1};
  for (int j{0}; j < n; ++j) {
    elements *= GetDimension(j).Extent();
  }
  return elements;
}

int Descriptor::Allocate(
    const SubscriptValue lb[], const SubscriptValue ub[], std::size_t charLen) {
  int result{ISO::CFI_allocate(&raw_, lb, ub, charLen)};
  if (result == CFI_SUCCESS) {
    Initialize(reinterpret_cast<char *>(raw_.base_addr));
  }
  return result;
}

void Descriptor::Initialize(char *data) const {
  if (auto *addendum{Addendum()}) {
    if (const auto *type{addendum->derivedType()}) {
      if (type->IsInitializable()) {
        std::size_t elements{Elements()};
        std::size_t elementBytes{ElementBytes()};
        char *data{static_cast<char *>(raw_.base_addr)};
        for (std::size_t j{0}; j < elements; ++j) {
          char *element{data + j * elementBytes};
          type->Initialize(element);
        }
      }
    }
  }
}

int Descriptor::Deallocate(bool finalize) {
  Destroy(finalize);
  return ISO::CFI_deallocate(&raw_);
}

void Descriptor::Destroy(char *data, bool finalize) const {
  if (data) {
    if (const DescriptorAddendum * addendum{Addendum()}) {
      if (const DerivedType * dt{addendum->derivedType()}) {
        if (raw_.flags_ & DoNotFinalize) {
          finalize = false;
        }
        Destroy(data, *dt, finalize);
      }
    }
  }
}

void Descriptor::Destroy(bool finalize) const {
  Destroy(static_cast<char *>(raw_.base_addr));
}

void Descriptor::Destroy(
    char *data, const DerivedType &type, bool finalize) const {
  // FINAL procedures for the array (or each element thereof) must be called
  // before the FINAL procedures of the components.  7.5.6.2 in F'2018.
  void (*elementalFinal)(char *){nullptr};
  if (finalize && type.IsFinalizable()) {
    int tbps{type.typeBoundProcedures()};
    for (int j{0}; j < tbps; ++j) {
      const auto &tbp{type.typeBoundProcedure(j)};
      if (tbp.IsFinalForRank(raw_.rank)) {
        if (auto f{reinterpret_cast<void (*)(const Descriptor *)>(
                tbp.code.host)}) {
          f(this);
        }
        elementalFinal = nullptr;
        break;
      }
      if ((tbp.flags & TypeBoundProcedure::ELEMENTAL) && (tbp.finalRank & 1)) {
        elementalFinal = reinterpret_cast<void (*)(char *)>(tbp.code.host);
      }
    }
  }
  // Subtle: The FINAL subroutines of any allocatable non-parent
  // component are also called now as the allocatables are
  // deallocated, after the finalization of the instance as a
  // whole (9.7.3.2 para 9), but before the finalization of the
  // parent component.
  // This is the Fortran community's desired order of events.
  std::size_t elements{Elements()};
  std::size_t elementBytes{ElementBytes()};
  for (std::size_t j{0}; j < elements; ++j) {
    char *element{data + j * elementBytes};
    if (elementalFinal) {
      elementalFinal(element);
    }
    type.DestroyNonParentComponents(element, finalize);
  }
  if (type.IsExtension()) {
    // Parent FINAL subroutine(s) must be called after the FINAL
    // subroutines of the components.
    const Descriptor *staticParentDescriptor{
        type.component(0).staticDescriptor()};
    Terminator terminator{__FILE__, __LINE__};
    RUNTIME_CHECK(terminator, staticParentDescriptor);
    const auto *parentAddendum{staticParentDescriptor->Addendum()};
    RUNTIME_CHECK(terminator, parentAddendum);
    const auto *parentType{parentAddendum->derivedType()};
    RUNTIME_CHECK(terminator, parentType);
    Destroy(data, *parentType, finalize);
  }
}

bool Descriptor::IncrementSubscripts(
    SubscriptValue *subscript, const int *permutation) const {
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    if (subscript[k]++ < dim.UpperBound()) {
      return true;
    }
    subscript[k] = dim.LowerBound();
  }
  return false;
}

bool Descriptor::DecrementSubscripts(
    SubscriptValue *subscript, const int *permutation) const {
  for (int j{raw_.rank - 1}; j >= 0; --j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    if (--subscript[k] >= dim.LowerBound()) {
      return true;
    }
    subscript[k] = dim.UpperBound();
  }
  return false;
}

std::size_t Descriptor::ZeroBasedElementNumber(
    const SubscriptValue *subscript, const int *permutation) const {
  std::size_t result{0};
  std::size_t coefficient{1};
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    result += coefficient * (subscript[k] - dim.LowerBound());
    coefficient *= dim.Extent();
  }
  return result;
}

bool Descriptor::SubscriptsForZeroBasedElementNumber(SubscriptValue *subscript,
    std::size_t elementNumber, const int *permutation) const {
  std::size_t coefficient{1};
  std::size_t dimCoefficient[maxRank];
  for (int j{0}; j < raw_.rank; ++j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    dimCoefficient[j] = coefficient;
    coefficient *= dim.Extent();
  }
  if (elementNumber >= coefficient) {
    return false; // out of range
  }
  for (int j{raw_.rank - 1}; j >= 0; --j) {
    int k{permutation ? permutation[j] : j};
    const Dimension &dim{GetDimension(k)};
    std::size_t quotient{j ? elementNumber / dimCoefficient[j] : 0};
    subscript[k] =
        dim.LowerBound() + elementNumber - dimCoefficient[j] * quotient;
    elementNumber = quotient;
  }
  return true;
}

void Descriptor::Check(const Terminator &terminator) const {
  RUNTIME_CHECK(terminator, raw_.version == CFI_VERSION);
  RUNTIME_CHECK(terminator, raw_.rank <= maxRank);
  RUNTIME_CHECK(terminator,
      raw_.type == CFI_type_other ||
          (raw_.type > 0 && raw_.type <= CFI_type_struct));
}

void Descriptor::Dump(FILE *f) const {
  std::fprintf(f, "Descriptor @ %p:\n", reinterpret_cast<const void *>(this));
  std::fprintf(f, "  base_addr %p\n", raw_.base_addr);
  std::fprintf(f, "  elem_len  %zd\n", static_cast<std::size_t>(raw_.elem_len));
  std::fprintf(f, "  version   %d\n", static_cast<int>(raw_.version));
  std::fprintf(f, "  rank      %d\n", static_cast<int>(raw_.rank));
  std::fprintf(f, "  type      %d\n", static_cast<int>(raw_.type));
  std::fprintf(f, "  attribute %d\n", static_cast<int>(raw_.attribute));
  std::fprintf(
      f, "  flags_    0x%jx\n", static_cast<std::uintmax_t>(raw_.flags_));
  for (int j{0}; j < raw_.rank; ++j) {
    std::fprintf(f, "  dim[%d] lower_bound %jd\n", j,
        static_cast<std::intmax_t>(raw_.dim[j].lower_bound));
    std::fprintf(f, "         extent      %jd\n",
        static_cast<std::intmax_t>(raw_.dim[j].extent));
    std::fprintf(f, "         sm          %jd\n",
        static_cast<std::intmax_t>(raw_.dim[j].sm));
  }
  if (const DescriptorAddendum * addendum{Addendum()}) {
    addendum->Dump(f);
  }
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(LenParameters());
}

void DescriptorAddendum::Dump(FILE *f) const {
  if (derivedType_) {
    std::fprintf(
        f, "  derivedType @ %p", reinterpret_cast<const void *>(derivedType_));
    std::fflush(f); // in case of bad pointer
    std::fprintf(f, ": %s\n", derivedType_->name());
    for (int j{0}; j < derivedType_->kindParameters(); ++j) {
      std::fprintf(f, "    KIND %s = %jd\n",
          derivedType_->kindTypeParameter(j).name(),
          static_cast<std::intmax_t>(
              derivedType_->kindTypeParameter(j).StaticValue()));
    }
    for (int j{0}; j < derivedType_->lenParameters(); ++j) {
      std::fprintf(f, "    LEN  %s = %jd\n",
          derivedType_->lenTypeParameter(j).name(),
          static_cast<std::intmax_t>(len_[j]));
    }
  }
}
} // namespace Fortran::runtime
