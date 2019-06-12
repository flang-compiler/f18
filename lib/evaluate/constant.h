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
#include "../common/abstract-descriptor.h"
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

inline ConstantSubscripts InitialSubscripts(int rank) {
  return ConstantSubscripts(rank, 1);  // parens, not braces: "rank" copies of 1
}
inline ConstantSubscripts InitialSubscripts(const ConstantSubscripts &shape) {
  return InitialSubscripts(GetRank(shape));
}

// Increments a vector of subscripts in Fortran array order (first dimension
// varying most quickly).  Returns false when last element was visited.
bool IncrementSubscripts(ConstantSubscripts &, const ConstantSubscripts &shape);
// Increments a vector of subscripts according to given dimension order
bool IncrementSubscripts(ConstantSubscripts &, const ConstantSubscripts &shape,
    const std::vector<int> &dimOrder);

class ConstantDescriptor;

// Constant<> is specialized for Character kinds and SomeDerived.
// The non-Character intrinsic types, and SomeDerived, share enough
// common behavior that they use this common base class.
template<typename RESULT, typename ELEMENT = Scalar<RESULT>>
class ConstantBase {
  static_assert(RESULT::category != TypeCategory::Character);
  friend ConstantDescriptor;

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
  constexpr Result result() const { return result_; }

  constexpr DynamicType GetType() const { return result_.GetType(); }
  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  std::vector<Element> Reshape(const ConstantSubscripts &) const;

  Result result_;
  std::vector<Element> values_;
  ConstantSubscripts shape_;
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

  // Apply 1-based subscripts
  Element At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
};

template<int KIND> class Constant<Type<TypeCategory::Character, KIND>> {
  friend ConstantDescriptor;

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

  ConstantSubscript LEN() const { return length_; }

  std::optional<Scalar<Result>> GetScalarValue() const {
    if (shape_.empty()) {
      return values_;
    } else {
      return std::nullopt;
    }
  }

  // Apply 1-based subscripts
  Scalar<Result> At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;
  static constexpr DynamicType GetType() {
    return {TypeCategory::Character, KIND};
  }

private:
  Scalar<Result> values_;  // one contiguous string
  ConstantSubscript length_;
  ConstantSubscripts shape_;
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
};

FOR_EACH_LENGTHLESS_INTRINSIC_KIND(extern template class ConstantBase, )
extern template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(extern template class Constant, )

#define INSTANTIATE_CONSTANT_TEMPLATES \
  FOR_EACH_LENGTHLESS_INTRINSIC_KIND(template class ConstantBase, ) \
  template class ConstantBase<SomeDerived, StructureConstructorValues>; \
  FOR_EACH_INTRINSIC_KIND(template class Constant, )

// ConstantDescriptor is a descriptor that wraps a Constant<T> and is compliant
// with DescriptorInterface. It can be used for folding with the
// transformational intrinsic function implementation taking a descriptor type
// as template argument.
class ConstantDescriptor {
public:
  using SubscriptValue = ConstantSubscript;
  using SizeValue = std::size_t;
  using Attribute = int;
  using Rank = int;

  template<typename A> using RankedSizedArray = std::vector<A>;
  using SubscriptArray = RankedSizedArray<SubscriptValue>;
  template<typename A> using OwningPointer = std::unique_ptr<A>;

  struct FortranType {
    bool IsInteger() const { return type.category() == TypeCategory::Integer; }
    bool operator==(const FortranType &that) const { return type == that.type; }
    bool operator!=(const FortranType &that) const { return !(*this == that); }
    DynamicType type;
  };
  struct Dimension {
    constexpr SubscriptValue LowerBound() const { return 1; }
    constexpr SubscriptValue UpperBound() const { return extent; }
    SubscriptValue Extent() const { return extent; };
    SubscriptValue extent;
  };
  static constexpr Rank maxRank{15};

  template<typename T>
  ConstantDescriptor(const Constant<T> &constant)
    : type_{constant.GetType()}, rank_{constant.Rank()},
      elementBytes_{ElementBytes(constant)}, someConstant_{
                                                 AllocatableConstant<T>{
                                                     constant}} {}
  template<typename T>
  ConstantDescriptor(Constant<T> &&constant)
    : type_{constant.GetType()}, rank_{constant.Rank()},
      elementBytes_{ElementBytes(constant)}, someConstant_{
                                                 AllocatableConstant<T>{
                                                     std::move(constant)}} {}
  template<typename T>
  ConstantDescriptor(
      const DynamicType &type, int rank, SizeValue elementBytes, const T &)
    : type_{std::move(type)}, rank_{rank}, elementBytes_{elementBytes},
      someConstant_{AllocatableConstant<T>{std::nullopt}} {}

  int rank() const { return rank_; }
  FortranType type() const { return FortranType{type_}; }
  Dimension GetDimension(int i) const {
    CHECK(i < rank_);
    return std::visit(
        [&](const auto &optionalConstant) -> Dimension {
          return Dimension{optionalConstant.value().shape()[i]};
        },
        someConstant_);
  }
  void GetLowerBounds(SubscriptArray &subscripts) const {
    for (int i{0}; i < rank(); ++i) {
      subscripts[i] = 1;
    }
  }
  bool HasSameTypeAs(const ConstantDescriptor &other) const {
    return type() == other.type();
  }
  SizeValue Elements() const {
    return std::visit(
        [&](const auto &optionalConstant) -> SizeValue {
          return Elements(optionalConstant.value().shape());
        },
        someConstant_);
  }
  constexpr void SetCompilerCreatedFlag() { /* no-op in front-end */
  }
  SubscriptValue GetSubscriptValueAt(const SubscriptArray &subscripts) const;
  static OwningPointer<ConstantDescriptor> CreateWithSameTypeAs(
      const ConstantDescriptor &source, int rank, Attribute);
  void Allocate(const SubscriptArray &extents);
  // Copies are only allowed when the underlying type is the same in source
  SizeValue CopyFrom(const ConstantDescriptor &source,
      SubscriptArray &resultSubscripts, SizeValue count,
      const RankedSizedArray<int> *dimOrder);

  template<typename T> Constant<T> ReleaseConstant() {
    auto &optConstant = std::get<AllocatableConstant<T>>(someConstant_);
    CHECK(optConstant.has_value());
    return std::move(optConstant.value());
  }
  constexpr static Attribute AllocatableAttribute() {
    // ConstantDescriptor only acts as a descriptor for allocatable, so this
    // does not really matter
    return 0;
  }

private:
  static SizeValue Elements(const SubscriptArray &shape) {
    SizeValue elements{1};
    for (auto i : shape) {
      elements *= static_cast<SizeValue>(i);
    }
    return elements;
  }
  template<typename T>
  static SizeValue ElementBytes(const Constant<T> &constant) {
    return sizeof(typename Constant<T>::Element);
  }
  template<int KIND>
  static SizeValue ElementBytes(
      const Constant<Type<TypeCategory::Character, KIND>> &constant) {
    return sizeof(typename Scalar<
               Type<TypeCategory::Character, KIND>>::value_type) *
        constant.LEN();
  }
  template<typename T>
  static SizeValue Offset(
      const Constant<T> &constant, const SubscriptArray &subscripts) {
    SizeValue offset{0};
    SizeValue stride{ElementBytes(constant)};
    for (int i{0}; i < constant.Rank(); ++i) {
      offset += (subscripts[i] - 1) * stride;
      stride *= constant.shape()[i];
    }
    return offset;
  }
  template<typename T, typename A>
  static A *Element(Constant<T> &constant, const SubscriptArray &subscripts) {
    return reinterpret_cast<A *>(
        reinterpret_cast<char *>(&constant.values_[0]) +
        Offset(constant, subscripts));
  }
  template<typename T, typename A>
  static const A *Element(
      const Constant<T> &constant, const SubscriptArray &subscripts) {
    return reinterpret_cast<const A *>(
        reinterpret_cast<const char *>(&constant.values_[0]) +
        Offset(constant, subscripts));
  }

  template<typename T> using AllocatableConstant = std::optional<Constant<T>>;

  DynamicType type_;
  int rank_;
  SizeValue elementBytes_;
  Fortran::common::MapTemplate<AllocatableConstant, AllTypes> someConstant_;
};

}

namespace Fortran::common {
extern template class Transformational<evaluate::ConstantDescriptor>;
}

#endif  // FORTRAN_EVALUATE_CONSTANT_H_
