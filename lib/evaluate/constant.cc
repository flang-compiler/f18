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

#include "constant.h"
#include "expression.h"
#include "shape.h"
#include "type.h"
#include <string>

namespace Fortran::evaluate {

std::size_t TotalElementCount(const ConstantSubscripts &shape) {
  std::size_t size{1};
  for (auto dim : shape) {
    CHECK(dim >= 0);
    size *= dim;
  }
  return size;
}

bool IncrementSubscripts(
    ConstantSubscripts &indices, const ConstantSubscripts &shape) {
  int rank{GetRank(shape)};
  CHECK(GetRank(indices) == rank);
  for (int j{0}; j < rank; ++j) {
    CHECK(indices[j] >= 1);
    if (++indices[j] <= shape[j]) {
      return true;
    } else {
      CHECK(indices[j] == shape[j] + 1);
      indices[j] = 1;
    }
  }
  return false;  // all done
}
bool IncrementSubscripts(ConstantSubscripts &indices,
    const ConstantSubscripts &shape, const std::vector<int> &dimOrder) {
  int rank{GetRank(shape)};
  CHECK(GetRank(indices) == rank);
  CHECK(static_cast<int>(dimOrder.size()) == rank);
  for (int j{0}; j < rank; ++j) {
    ConstantSubscript k{dimOrder[j]};
    CHECK(indices[k] >= 1);
    if (++indices[k] <= shape[k]) {
      return true;
    } else {
      CHECK(indices[k] == shape[k] + 1);
      indices[k] = 1;
    }
  }
  return false;  // all done
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::ConstantBase(
    std::vector<Element> &&x, ConstantSubscripts &&dims, Result res)
  : result_{res}, values_(std::move(x)), shape_(std::move(dims)) {
  CHECK(size() == TotalElementCount(shape_));
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::~ConstantBase() {}

template<typename RESULT, typename ELEMENT>
bool ConstantBase<RESULT, ELEMENT>::operator==(const ConstantBase &that) const {
  return shape_ == that.shape_ && values_ == that.values_;
}

static ConstantSubscript SubscriptsToOffset(
    const ConstantSubscripts &index, const ConstantSubscripts &shape) {
  CHECK(GetRank(index) == GetRank(shape));
  ConstantSubscript stride{1}, offset{0};
  int dim{0};
  for (auto j : index) {
    auto bound{shape[dim++]};
    CHECK(j >= 1 && j <= bound);
    offset += stride * (j - 1);
    stride *= bound;
  }
  return offset;
}

template<typename RESULT, typename ELEMENT>
Constant<SubscriptInteger> ConstantBase<RESULT, ELEMENT>::SHAPE() const {
  return AsConstantShape(shape_);
}

template<typename RESULT, typename ELEMENT>
auto ConstantBase<RESULT, ELEMENT>::Reshape(
    const ConstantSubscripts &dims) const -> std::vector<Element> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  auto iter{values().cbegin()};
  while (n-- > 0) {
    elements.push_back(*iter);
    if (++iter == values().cend()) {
      iter = values().cbegin();
    }
  }
  return elements;
}

template<typename T>
auto Constant<T>::At(const ConstantSubscripts &index) const -> Element {
  return Base::values_.at(SubscriptsToOffset(index, Base::shape_));
}

template<typename T>
auto Constant<T>::Reshape(ConstantSubscripts &&dims) const -> Constant {
  return {Base::Reshape(dims), std::move(dims)};
}

// Constant<Type<TypeCategory::Character, KIND> specializations
template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(
    const Scalar<Result> &str)
  : values_{str}, length_{static_cast<ConstantSubscript>(values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(Scalar<Result> &&str)
  : values_{std::move(str)}, length_{static_cast<ConstantSubscript>(
                                 values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(ConstantSubscript len,
    std::vector<Scalar<Result>> &&strings, ConstantSubscripts &&dims)
  : length_{len}, shape_{std::move(dims)} {
  CHECK(strings.size() == TotalElementCount(shape_));
  values_.assign(strings.size() * length_,
      static_cast<typename Scalar<Result>::value_type>(' '));
  ConstantSubscript at{0};
  for (const auto &str : strings) {
    auto strLen{static_cast<ConstantSubscript>(str.size())};
    if (strLen > length_) {
      values_.replace(at, length_, str.substr(0, length_));
    } else {
      values_.replace(at, strLen, str);
    }
    at += length_;
  }
  CHECK(at == static_cast<ConstantSubscript>(values_.size()));
}

template<int KIND> Constant<Type<TypeCategory::Character, KIND>>::~Constant() {}

template<int KIND>
bool Constant<Type<TypeCategory::Character, KIND>>::empty() const {
  return size() == 0;
}

template<int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::size() const {
  if (length_ == 0) {
    return TotalElementCount(shape_);
  } else {
    return static_cast<ConstantSubscript>(values_.size()) / length_;
  }
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const ConstantSubscripts &index) const -> Scalar<Result> {
  auto offset{SubscriptsToOffset(index, shape_)};
  return values_.substr(offset * length_, length_);
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::Reshape(
    ConstantSubscripts &&dims) const -> Constant<Result> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  ConstantSubscript at{0},
      limit{static_cast<ConstantSubscript>(values_.size())};
  while (n-- > 0) {
    elements.push_back(values_.substr(at, length_));
    at += length_;
    if (at == limit) {  // subtle: at > limit somehow? substr() will catch it
      at = 0;
    }
  }
  return {length_, std::move(elements), std::move(dims)};
}

template<int KIND>
Constant<SubscriptInteger>
Constant<Type<TypeCategory::Character, KIND>>::SHAPE() const {
  return AsConstantShape(shape_);
}

// Constant<SomeDerived> specialization
Constant<SomeDerived>::Constant(const StructureConstructor &x)
  : Base{x.values(), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(StructureConstructor &&x)
  : Base{std::move(x.values()), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructorValues> &&x, ConstantSubscripts &&s)
  : Base{std::move(x), std::move(s), Result{spec}} {}

static std::vector<StructureConstructorValues> AcquireValues(
    std::vector<StructureConstructor> &&x) {
  std::vector<StructureConstructorValues> result;
  for (auto &&structure : std::move(x)) {
    result.emplace_back(std::move(structure.values()));
  }
  return result;
}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructor> &&x, ConstantSubscripts &&s)
  : Base{AcquireValues(std::move(x)), std::move(s), Result{spec}} {}

std::optional<StructureConstructor>
Constant<SomeDerived>::GetScalarValue() const {
  if (shape_.empty()) {
    return StructureConstructor{result().derivedTypeSpec(), values_.at(0)};
  } else {
    return std::nullopt;
  }
}

StructureConstructor Constant<SomeDerived>::At(
    const ConstantSubscripts &index) const {
  return {result().derivedTypeSpec(),
      values_.at(SubscriptsToOffset(index, shape_))};
}

auto Constant<SomeDerived>::Reshape(ConstantSubscripts &&dims) const
    -> Constant {
  return {result().derivedTypeSpec(), Base::Reshape(dims), std::move(dims)};
}

INSTANTIATE_CONSTANT_TEMPLATES

typename ConstantDescriptor::SubscriptValue
ConstantDescriptor::GetSubscriptValueAt(
    const SubscriptArray &subscripts) const {
  return std::visit(
      [&](const auto &constant) -> SubscriptValue {
        using T = typename std::decay_t<decltype(constant.value())>::Result;
        if constexpr (T::category == TypeCategory::Integer) {
          return constant.value().At(subscripts).ToInt64();
        } else {
          CHECK(false); /* Cannot use non-integer as subscripts */
          return 0;
        }
      },
      someConstant_);
}

typename ConstantDescriptor::OwningPointer<ConstantDescriptor>
ConstantDescriptor::CreateWithSameTypeAs(
    const ConstantDescriptor &source, int rank, Attribute) {
  ConstantDescriptor *ptr{std::visit(
      [&](const auto &constant) -> ConstantDescriptor * {
        using T = typename std::decay_t<decltype(constant.value())>::Result;
        return new ConstantDescriptor(
            source.type_, rank, source.elementBytes_, T{});
      },
      source.someConstant_)};
  return OwningPointer<ConstantDescriptor>{ptr};
}

void ConstantDescriptor::Allocate(const SubscriptArray &extents) {
  std::visit(
      [&](auto &constant) {
        using T = typename std::decay_t<decltype(constant.value())>::Result;
        SizeValue elements{Elements(extents)};
        std::vector<ConstantSubscript> shape = extents;
        if constexpr (T::category == TypeCategory::Character) {
          SubscriptValue len{static_cast<SubscriptValue>(
              elementBytes_ / sizeof(typename Scalar<T>::value_type))};
          Scalar<T> scalar(len, ' ');
          constant = Constant<T>{
              len, std::vector<Scalar<T>>(elements, scalar), std::move(shape)};
        } else if constexpr (T::category == TypeCategory::Derived) {
          constant = Constant<T>{type_.GetDerivedTypeSpec(),
              std::vector<StructureConstructorValues>(elements),
              std::move(shape)};
        } else {
          constant =
              Constant<T>{std::vector<Scalar<T>>(elements), std::move(shape)};
        }
      },
      someConstant_);
}

typename ConstantDescriptor::SizeValue ConstantDescriptor::CopyFrom(
    const ConstantDescriptor &source, SubscriptArray &resultSubscripts,
    SizeValue count, const RankedSizedArray<int> *dimOrder) {
  SubscriptArray sourceSubscripts(source.rank());
  source.GetLowerBounds(sourceSubscripts);
  return std::visit(
      [&](auto &optionalConstant) -> SizeValue {
        auto &constant{optionalConstant.value()};
        using T = typename std::decay_t<decltype(constant)>::Result;
        using ElementT = typename Constant<T>::Element;
        const auto &constantSource{
            std::get<AllocatableConstant<T>>(source.someConstant_).value()};
        SizeValue copied{0};
        while (copied < count) {
          if constexpr (T::category == TypeCategory::Character) {
            // Character array elements cannot be represented as string in a
            // mutable way, memcpy the content manually instead.
            std::memcpy(Element<T, void>(constant, resultSubscripts),
                Element<T, void>(constantSource, sourceSubscripts),
                elementBytes_);
          } else {
            // Do not memcpy derived type because they are currently represented
            // as maps so they need to be deep copied.
            *Element<T, ElementT>(constant, resultSubscripts) =
                *Element<T, ElementT>(constantSource, sourceSubscripts);
          }
          copied++;
          IncrementSubscripts(sourceSubscripts, constantSource.shape());
          if (dimOrder) {
            IncrementSubscripts(resultSubscripts, constant.shape(), *dimOrder);
          } else {
            IncrementSubscripts(resultSubscripts, constant.shape());
          }
        }
        return copied;
      },
      someConstant_);
}

}
namespace Fortran::common {
template class Transformational<evaluate::ConstantDescriptor>;
}
