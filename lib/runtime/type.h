// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_RUNTIME_TYPE_H_
#define FORTRAN_RUNTIME_TYPE_H_

#include <cinttypes>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <vector>

namespace Fortran::runtime {

// Fortran 2018 section 7.1

class KindSpecificType;

// A generic type all its KIND and LEN parameters unbound.
class Type {
public:
  enum class Classification {
    Integer,
    Real,
    Complex,
    Character,
    Logical,
    Derived
  };

  class Parameter {
  public:
    constexpr Parameter(
        const char *n, const KindSpecificType &t, std::int64_t x)
      : name_{n}, type_{t}, defaultValue_{x} {}
    constexpr Parameter(const Parameter &) = default;
    constexpr Parameter &operator=(const Parameter &) = default;
    const char *name() const { return name_; }
    const KindSpecificType &type() const { return type_; }
    std::int64_t defaultValue() const { return defaultValue_; }

  private:
    const char *name_;
    const KindSpecificType &type_;
    std::int64_t defaultValue_;
  };

  Type(Classification classification,
      const std::initializer_list<Parameter> &kinds,
      const std::initializer_list<Parameter> &lens)
    : classification_{classification}, kind_{kinds}, len_{lens} {}

  const char *name() const { return name_; }
  Classification classification() const { return classification_; }

  bool IsIntrinsic() const {
    return classification_ != Classification::Derived;
  }
  bool IsDerived() const { return classification_ == Classification::Derived; }
  std::size_t KindParameters() const { return kind_.size(); }
  std::size_t LenParameters() const { return len_.size(); }
  const Parameter &KindParameter(std::size_t which) const {
    return kind_.at(which);
  }
  const Parameter &LenParameter(std::size_t which) const {
    return len_.at(which);
  }

private:
  const char *name_;
  Classification classification_;
  std::vector<Parameter> kind_, len_;
};

// A generic Type with all its KIND type parameters bound.  LEN type parameters
// remain unbound.
class KindSpecificType {
public:
  KindSpecificType(const Type *t,
      const std::initializer_list<std::int64_t> &kinds, std::size_t bytes)
    : type_{t}, kind_{kinds}, bytes_{bytes} {}

  const Type &type() const { return *type_; }

  std::int64_t KindParameterValue(std::size_t which) const {
    return kind_.at(which);
  }
  std::size_t SizeInBytes() const { return bytes_; }
  bool IsSameType(const KindSpecificType) const;

private:
  friend class IntrinsicType;
  void set_type(const Type *t) { type_ = t; }

  const Type *type_;
  std::vector<std::int64_t> kind_;
  std::size_t bytes_{0};
};

// Meant to be a singleton; instantiates all kind-specific intrinsic types.
class IntrinsicType {
public:
  // Default INTEGER must be the same size as default REAL, since both
  // are defined to occupy a single numeric storage unit.  Default REAL
  // just has to be 32-bit IEEE-754 floating-point.  Ergo:
  using DefaultInteger = std::int32_t;
  static_assert(sizeof(DefaultInteger) == 4);

  IntrinsicType();

  const KindSpecificType *Integer(
      const std::optional<DefaultInteger> &kind) const {
    return Find(kindSpecificInteger_, kind);
  }
  const KindSpecificType *Real(std::optional<DefaultInteger> kind) const {
    return Find(kindSpecificReal_, kind);
  }
  const KindSpecificType *DoublePrecision() const { return Real(2 * 4); }
  const KindSpecificType *Complex(std::optional<DefaultInteger> kind) const {
    return Find(kindSpecificComplex_, kind);
  }
  const KindSpecificType *Character(std::optional<DefaultInteger> kind) const {
    return Find(kindSpecificCharacter_, kind);
  }
  const KindSpecificType *Logical(std::optional<DefaultInteger> kind) const {
    return Find(kindSpecificLogical_, kind);
  }

private:
  const KindSpecificType *Find(const std::vector<KindSpecificType> &,
      const std::optional<DefaultInteger> &kind) const;

  KindSpecificType defaultInteger_;
  Type genericInteger_;
  Type genericReal_;
  Type genericComplex_;
  Type genericCharacter_;
  Type genericLogical_;
  std::vector<KindSpecificType> kindSpecificInteger_;
  std::vector<KindSpecificType> kindSpecificReal_;
  std::vector<KindSpecificType> kindSpecificComplex_;
  std::vector<KindSpecificType> kindSpecificCharacter_;
  std::vector<KindSpecificType> kindSpecificLogical_;
};
}  // namespace Fortran::runtime
#endif  // FORTRAN_RUNTIME_TYPE_H_
