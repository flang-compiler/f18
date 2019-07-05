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

bool IncrementSubscripts(ConstantSubscripts &indices,
    const ConstantSubscripts &shape, const ConstantSubscripts &lbound,
    const std::vector<int> *dimOrder) {
  int rank{GetRank(shape)};
  CHECK(GetRank(indices) == rank);
  CHECK(GetRank(lbound) == rank);
  CHECK(!dimOrder || static_cast<int>(dimOrder->size()) == rank);
  for (int j{0}; j < rank; ++j) {
    auto lb{lbound[j]};
    ConstantSubscript k{dimOrder ? (*dimOrder)[j] : j};
    CHECK(indices[k] >= lb);
    if (++indices[k] < lb + shape[k]) {
      return true;
    } else {
      CHECK(indices[k] == shape[k] + lb);
      indices[k] = lb;
    }
  }
  return false;  // all done
}

std::optional<std::vector<int>> IsValidDimensionOrder(
    int rank, const std::optional<std::vector<int>> &order) {
  std::vector<int> dimOrder(rank);
  if (!order.has_value()) {
    for (int j{0}; j < rank; ++j) {
      dimOrder[j] = j;
    }
    return dimOrder;
  } else if (static_cast<int>(order.value().size()) == rank) {
    std::bitset<common::maxRank> seenDimensions;
    for (int j{0}; j < rank; ++j) {
      int dim{order.value()[j]};
      if (dim < 1 || dim > rank || seenDimensions.test(dim - 1)) {
        return std::nullopt;
      }
      dimOrder[dim - 1] = j;
      seenDimensions.set(dim - 1);
    }
    return dimOrder;
  } else {
    return std::nullopt;
  }
}

bool IsValidShape(const ConstantSubscripts &shape) {
  for (ConstantSubscript extent : shape) {
    if (extent < 0) {
      return false;
    }
  }
  return shape.size() <= common::maxRank;
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::ConstantBase(
    std::vector<Element> &&x, ConstantSubscripts &&dims, Result res)
  : result_{res}, values_(std::move(x)), shape_(std::move(dims)),
    lbounds_(shape_.size(), 1) {
  CHECK(size() == TotalElementCount(shape_));
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::~ConstantBase() {}

template<typename RESULT, typename ELEMENT>
bool ConstantBase<RESULT, ELEMENT>::operator==(const ConstantBase &that) const {
  return shape_ == that.shape_ && values_ == that.values_;
}

template<typename RESULT, typename ELEMENT>
void ConstantBase<RESULT, ELEMENT>::set_lbounds(ConstantSubscripts &&lb) {
  CHECK(lb.size() == shape_.size());
  lbounds_ = std::move(lb);
}

static ConstantSubscript SubscriptsToOffset(const ConstantSubscripts &index,
    const ConstantSubscripts &shape, const ConstantSubscripts &lbound) {
  CHECK(GetRank(index) == GetRank(shape));
  ConstantSubscript stride{1}, offset{0};
  int dim{0};
  for (auto j : index) {
    auto lb{lbound[dim]};
    auto extent{shape[dim++]};
    CHECK(j >= lb && j < lb + extent);
    offset += stride * (j - lb);
    stride *= extent;
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

template<typename RESULT, typename ELEMENT>
std::size_t ConstantBase<RESULT, ELEMENT>::CopyFrom(
    const ConstantBase<RESULT, ELEMENT> &source, std::size_t count,
    ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder) {
  std::size_t copied{0};
  ConstantSubscripts sourceSubscripts{source.lbounds_};
  while (copied < count) {
    values_.at(SubscriptsToOffset(resultSubscripts, shape_, lbounds_)) =
        source.values_.at(SubscriptsToOffset(
            sourceSubscripts, source.shape_, source.lbounds_));
    copied++;
    IncrementSubscripts(sourceSubscripts, source.shape_, source.lbounds_);
    IncrementSubscripts(resultSubscripts, shape_, lbounds_, dimOrder);
  }
  return copied;
}

template<typename T>
auto Constant<T>::At(const ConstantSubscripts &index) const -> Element {
  return Base::values_.at(
      SubscriptsToOffset(index, Base::shape_, Base::lbounds_));
}

template<typename T>
auto Constant<T>::Reshape(ConstantSubscripts &&dims) const -> Constant {
  return {Base::Reshape(dims), std::move(dims)};
}

template<typename T>
std::size_t Constant<T>::CopyFrom(const Constant<T> &source, std::size_t count,
    ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder) {
  return Base::CopyFrom(source, count, resultSubscripts, dimOrder);
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
  : length_{len}, shape_{std::move(dims)}, lbounds_(shape_.size(), 1) {
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
void Constant<Type<TypeCategory::Character, KIND>>::set_lbounds(
    ConstantSubscripts &&lb) {
  CHECK(lb.size() == shape_.size());
  lbounds_ = std::move(lb);
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const ConstantSubscripts &index) const -> Scalar<Result> {
  auto offset{SubscriptsToOffset(index, shape_, lbounds_)};
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

template<int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::CopyFrom(
    const Constant<Type<TypeCategory::Character, KIND>> &source,
    std::size_t count, ConstantSubscripts &resultSubscripts,
    const std::vector<int> *dimOrder) {
  CHECK(length_ == source.length_);
  std::size_t copied{0};
  std::size_t elementBytes{length_ * sizeof(decltype(values_[0]))};
  ConstantSubscripts sourceSubscripts{source.lbounds_};
  while (copied < count) {
    auto *dest{&values_.at(
        SubscriptsToOffset(resultSubscripts, shape_, lbounds_) * length_)};
    const auto *src{&source.values_.at(
        SubscriptsToOffset(sourceSubscripts, source.shape_, source.lbounds_) *
        length_)};
    std::memcpy(dest, src, elementBytes);
    copied++;
    IncrementSubscripts(sourceSubscripts, source.shape_, source.lbounds_);
    IncrementSubscripts(resultSubscripts, shape_, lbounds_, dimOrder);
  }
  return copied;
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
      values_.at(SubscriptsToOffset(index, shape_, lbounds_))};
}

auto Constant<SomeDerived>::Reshape(ConstantSubscripts &&dims) const
    -> Constant {
  return {result().derivedTypeSpec(), Base::Reshape(dims), std::move(dims)};
}

std::size_t Constant<SomeDerived>::CopyFrom(const Constant<SomeDerived> &source,
    std::size_t count, ConstantSubscripts &resultSubscripts,
    const std::vector<int> *dimOrder) {
  return Base::CopyFrom(source, count, resultSubscripts, dimOrder);
}

INSTANTIATE_CONSTANT_TEMPLATES
}
