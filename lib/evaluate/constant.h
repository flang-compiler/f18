// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_CONSTANT_H_
#define FORTRAN_EVALUATE_CONSTANT_H_

#include "formatting.h"
#include "type.h"
#include "../common/default-kinds.h"
#include <map>
#include <ostream>
#include <vector>

namespace Fortran::evaluate {

// Wraps a constant value in a class templated by its resolved type.
// This Constant<> template class should be instantiated only for
// concrete intrinsic types and SomeDerived.  There is no instance
// Constant<Expr<SomeType>> since there is no way to constrain each
// element of its array to hold the same type.  To represent a generic
// constants, use a generic expression like Expr<SomeInteger> &
// Expr<SomeType>) to wrap the appropriate instantiation of Constant<>.

template<typename> class Constant;

// When describing shapes of constants or specifying 1-based subscript
// values as indices into constants, use a vector of integers.
using ConstantSubscripts = std::vector<ConstantSubscript>;
inline int GetRank(const ConstantSubscripts &s) {
  return static_cast<int>(s.size());
}

std::size_t TotalElementCount(const ConstantSubscripts &);

// If no optional dimension order argument is passed, increments a vector of
// subscripts in Fortran array order (first dimension varying most quickly).
// Otherwise, increments the vector of subscripts according to the given
// dimension order (dimension dimOrder[0] varying most quickly. Dimensions
// indexing is zero based here.) Returns false when last element was visited.
bool IncrementSubscripts(ConstantSubscripts &, const ConstantSubscripts &shape,
    const ConstantSubscripts &lbound,
    const std::vector<int> *dimOrder = nullptr);

// Validate dimension re-ordering like ORDER in RESHAPE.
// On success, return a vector that can be used as dimOrder in
// IncrementSubscripts.
std::optional<std::vector<int>> ValidateDimensionOrder(
    int rank, const std::vector<int> &order);

bool IsValidShape(const ConstantSubscripts &);

// Constant<> is specialized for Character kinds and SomeDerived.
// The non-Character intrinsic types, and SomeDerived, share enough
// common behavior that they use this common base class.
template<typename RESULT, typename ELEMENT = Scalar<RESULT>>
class ConstantBase {
  static_assert(RESULT::category != TypeCategory::Character);

public:
  using Result = RESULT;
  using Element = ELEMENT;

  template<typename A>
  ConstantBase(const A &x, Result res = Result{}) : result_{res}, values_{x} {}
  template<typename A, typename = common::NoLvalue<A>>
  ConstantBase(A &&x, Result res = Result{})
    : result_{res}, values_{std::move(x)} {}
  ConstantBase(
      std::vector<Element> &&, ConstantSubscripts &&, Result = Result{});

  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ConstantBase)
  ~ConstantBase();

  int Rank() const { return GetRank(shape_); }
  bool operator==(const ConstantBase &) const;
  bool empty() const { return values_.empty(); }
  std::size_t size() const { return values_.size(); }
  const std::vector<Element> &values() const { return values_; }
  const ConstantSubscripts &shape() const { return shape_; }
  const ConstantSubscripts &lbounds() const { return lbounds_; }
  void set_lbounds(ConstantSubscripts &&);
  constexpr Result result() const { return result_; }

  constexpr DynamicType GetType() const { return result_.GetType(); }
  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  std::vector<Element> Reshape(const ConstantSubscripts &) const;
  std::size_t CopyFrom(const ConstantBase &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);

  Result result_;
  std::vector<Element> values_;
  ConstantSubscripts shape_;
  ConstantSubscripts lbounds_;
};

template<typename T> class Constant : public ConstantBase<T> {
public:
  using Result = T;
  using Base = ConstantBase<T>;
  using Element = Scalar<T>;

  using Base::Base;
  CLASS_BOILERPLATE(Constant)

  std::optional<Scalar<T>> GetScalarValue() const {
    if (Base::shape_.empty()) {
      return Base::values_.at(0);
    } else {
      return std::nullopt;
    }
  }

  // Apply subscripts.
  Element At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);
};

template<int KIND> class Constant<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using Element = Scalar<Result>;

  CLASS_BOILERPLATE(Constant)
  explicit Constant(const Scalar<Result> &);
  explicit Constant(Scalar<Result> &&);
  Constant(ConstantSubscript, std::vector<Element> &&, ConstantSubscripts &&);
  ~Constant();

  int Rank() const { return GetRank(shape_); }
  bool operator==(const Constant &that) const {
    return shape_ == that.shape_ && values_ == that.values_;
  }
  bool empty() const;
  std::size_t size() const;
  const ConstantSubscripts &shape() const { return shape_; }
  const ConstantSubscripts &lbounds() const { return lbounds_; }
  void set_lbounds(ConstantSubscripts &&);

  ConstantSubscript LEN() const { return length_; }

  std::optional<Scalar<Result>> GetScalarValue() const {
    if (shape_.empty()) {
      return values_;
    } else {
      return std::nullopt;
    }
  }

  // Apply subscripts
  Scalar<Result> At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;
  static constexpr DynamicType GetType() {
    return {TypeCategory::Character, KIND};
  }
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);

private:
  Scalar<Result> values_;  // one contiguous string
  ConstantSubscript length_;
  ConstantSubscripts shape_;
  ConstantSubscripts lbounds_;
};

class StructureConstructor;
using StructureConstructorValues = std::map<const semantics::Symbol *,
    common::CopyableIndirection<Expr<SomeType>>>;

template<>
class Constant<SomeDerived>
  : public ConstantBase<SomeDerived, StructureConstructorValues> {
public:
  using Result = SomeDerived;
  using Element = StructureConstructorValues;
  using Base = ConstantBase<SomeDerived, StructureConstructorValues>;

  Constant(const StructureConstructor &);
  Constant(StructureConstructor &&);
  Constant(const semantics::DerivedTypeSpec &,
      std::vector<StructureConstructorValues> &&, ConstantSubscripts &&);
  Constant(const semantics::DerivedTypeSpec &,
      std::vector<StructureConstructor> &&, ConstantSubscripts &&);
  CLASS_BOILERPLATE(Constant)

  std::optional<StructureConstructor> GetScalarValue() const;
  StructureConstructor At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);
};

FOR_EACH_LENGTHLESS_INTRINSIC_KIND(extern template class ConstantBase, )
extern template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(extern template class Constant, )

#define INSTANTIATE_CONSTANT_TEMPLATES \
  FOR_EACH_LENGTHLESS_INTRINSIC_KIND(template class ConstantBase, ) \
  template class ConstantBase<SomeDerived, StructureConstructorValues>; \
  FOR_EACH_INTRINSIC_KIND(template class Constant, )
}
#endif  // FORTRAN_EVALUATE_CONSTANT_H_
