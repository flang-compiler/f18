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

// GetShape() analyzes an expression and determines its shape, if possible,
// representing the result as a vector of scalar integer expressions.

#ifndef FORTRAN_EVALUATE_SHAPE_H_
#define FORTRAN_EVALUATE_SHAPE_H_

#include "expression.h"
#include "tools.h"
#include "traverse.h"
#include "type.h"
#include "variable.h"
#include "../common/indirection.h"
#include <optional>
#include <variant>

namespace Fortran::parser {
class ContextualMessages;
}

namespace Fortran::evaluate {

class FoldingContext;

using ExtentType = SubscriptInteger;
using ExtentExpr = Expr<ExtentType>;
using MaybeExtentExpr = std::optional<ExtentExpr>;
using Shape = std::vector<MaybeExtentExpr>;

bool IsImpliedShape(const Symbol &);
bool IsExplicitShape(const Symbol &);

template<typename A> std::optional<Shape> GetShape(FoldingContext &, const A &);

// Conversions between various representations of shapes.
Shape AsShape(const Constant<ExtentType> &);
std::optional<Shape> AsShape(FoldingContext &, ExtentExpr &&);

std::optional<ExtentExpr> AsExtentArrayExpr(const Shape &);

std::optional<Constant<ExtentType>> AsConstantShape(const Shape &);
Constant<ExtentType> AsConstantShape(const ConstantSubscripts &);

ConstantSubscripts AsConstantExtents(const Constant<ExtentType> &);
std::optional<ConstantSubscripts> AsConstantExtents(const Shape &);

inline int GetRank(const Shape &s) { return static_cast<int>(s.size()); }

// The dimension argument to these inquiries is zero-based,
// unlike the DIM= arguments to many intrinsics.
ExtentExpr GetLowerBound(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetUpperBound(
    FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr ComputeUpperBound(
    FoldingContext &, ExtentExpr &&lower, MaybeExtentExpr &&extent);
Shape GetLowerBounds(FoldingContext &, const NamedEntity &);
Shape GetUpperBounds(FoldingContext &, const NamedEntity &);
MaybeExtentExpr GetExtent(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(
    FoldingContext &, const Subscript &, const NamedEntity &, int dimension);

// Compute an element count for a triplet or trip count for a DO.
ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride);
ExtentExpr CountTrips(
    const ExtentExpr &lower, const ExtentExpr &upper, const ExtentExpr &stride);
MaybeExtentExpr CountTrips(
    MaybeExtentExpr &&lower, MaybeExtentExpr &&upper, MaybeExtentExpr &&stride);

// Computes SIZE() == PRODUCT(shape)
MaybeExtentExpr GetSize(Shape &&);

// Utility predicate: does an expression reference any implied DO index?
bool ContainsAnyImpliedDoIndex(const ExtentExpr &);

// GetShape()
class GetShapeHelper
  : public AnyTraverse<GetShapeHelper, std::optional<Shape>> {
public:
  using Result = std::optional<Shape>;
  using Base = AnyTraverse<GetShapeHelper, Result>;
  using Base::operator();
  GetShapeHelper(FoldingContext &c) : Base{*this}, context_{c} {}

  Result operator()(const ImpliedDoIndex &) const { return Scalar(); }
  Result operator()(const DescriptorInquiry &) const { return Scalar(); }
  template<int KIND> Result operator()(const TypeParamInquiry<KIND> &) const {
    return Scalar();
  }
  Result operator()(const BOZLiteralConstant &) const { return Scalar(); }
  Result operator()(const StaticDataObject::Pointer &) const {
    return Scalar();
  }
  Result operator()(const StructureConstructor &) const { return Scalar(); }

  template<typename T> Result operator()(const Constant<T> &c) const {
    return AsShape(c.SHAPE());
  }

  Result operator()(const Symbol &) const;
  Result operator()(const Component &) const;
  Result operator()(const NamedEntity &) const;
  Result operator()(const ArrayRef &) const;
  Result operator()(const CoarrayRef &) const;
  Result operator()(const Substring &) const;
  Result operator()(const ProcedureRef &) const;

  template<typename T>
  Result operator()(const ArrayConstructor<T> &aconst) const {
    return Shape{GetArrayConstructorExtent(aconst)};
  }
  template<typename D, typename R, typename LO, typename RO>
  Result operator()(const Operation<D, R, LO, RO> &operation) const {
    if (operation.right().Rank() > 0) {
      return (*this)(operation.right());
    } else {
      return (*this)(operation.left());
    }
  }

private:
  static Result Scalar() { return Shape{}; }

  template<typename T>
  MaybeExtentExpr GetArrayConstructorValueExtent(
      const ArrayConstructorValue<T> &value) const {
    return std::visit(
        common::visitors{
            [&](const Expr<T> &x) -> MaybeExtentExpr {
              if (std::optional<Shape> xShape{GetShape(context_, x)}) {
                // Array values in array constructors get linearized.
                return GetSize(std::move(*xShape));
              } else {
                return std::nullopt;
              }
            },
            [&](const ImpliedDo<T> &ido) -> MaybeExtentExpr {
              // Don't be heroic and try to figure out triangular implied DO
              // nests.
              if (!ContainsAnyImpliedDoIndex(ido.lower()) &&
                  !ContainsAnyImpliedDoIndex(ido.upper()) &&
                  !ContainsAnyImpliedDoIndex(ido.stride())) {
                if (auto nValues{GetArrayConstructorExtent(ido.values())}) {
                  return std::move(*nValues) *
                      CountTrips(ido.lower(), ido.upper(), ido.stride());
                }
              }
              return std::nullopt;
            },
        },
        value.u);
  }

  template<typename T>
  MaybeExtentExpr GetArrayConstructorExtent(
      const ArrayConstructorValues<T> &values) const {
    ExtentExpr result{0};
    for (const auto &value : values) {
      if (MaybeExtentExpr n{GetArrayConstructorValueExtent(value)}) {
        result = std::move(result) + std::move(*n);
      } else {
        return std::nullopt;
      }
    }
    return result;
  }

  FoldingContext &context_;
};

template<typename A>
std::optional<Shape> GetShape(FoldingContext &context, const A &x) {
  return GetShapeHelper{context}(x);
}

// Compilation-time shape conformance checking, when corresponding extents
// are known.
bool CheckConformance(parser::ContextualMessages &, const Shape &,
    const Shape &, const char * = "left operand",
    const char * = "right operand");

}
#endif  // FORTRAN_EVALUATE_SHAPE_H_
