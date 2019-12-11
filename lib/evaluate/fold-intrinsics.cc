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

#include "fold-intrinsics.h"
#include "character.h"
#include "characteristics.h"
#include "common.h"
#include "constant.h"
#include "expression.h"
#include "fold.h"
#include "host.h"
#include "intrinsics-library-templates.h"
#include "shape.h"
#include "tools.h"
#include "type.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"
#include <optional>
#include <type_traits>
#include <variant>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace Fortran::evaluate {

// helpers to fold intrinsic function references
// Define callable types used in a common utility that
// takes care of array and cast/conversion aspects for elemental intrinsics

template<typename TR, typename... TArgs>
using ScalarFunc = std::function<Scalar<TR>(const Scalar<TArgs> &...)>;
template<typename TR, typename... TArgs>
using ScalarFuncWithContext =
    std::function<Scalar<TR>(FoldingContext &, const Scalar<TArgs> &...)>;

// Apply type conversion and re-folding if necessary.
// This is where BOZ arguments are converted.
template<typename T>
static inline Constant<T> *FoldConvertedArg(
    FoldingContext &context, std::optional<ActualArgument> &arg) {
  if (auto *expr{UnwrapExpr<Expr<SomeType>>(arg)}) {
    if (!UnwrapExpr<Expr<T>>(*expr)) {
      if (auto converted{ConvertToType(T::GetType(), std::move(*expr))}) {
        *expr = Fold(context, std::move(*converted));
      }
    }
    return UnwrapConstantValue<T>(*expr);
  }
  return nullptr;
}

template<template<typename, typename...> typename WrapperType, typename TR,
    typename... TA, std::size_t... I>
static inline Expr<TR> FoldElementalIntrinsicHelper(FoldingContext &context,
    FunctionRef<TR> &&funcRef, WrapperType<TR, TA...> func,
    std::index_sequence<I...>) {
  static_assert(
      (... && IsSpecificIntrinsicType<TA>));  // TODO derived types for MERGE?
  static_assert(sizeof...(TA) > 0);
  std::tuple<const Constant<TA> *...> args{
      FoldConvertedArg<TA>(context, funcRef.arguments()[I])...};
  if ((... && (std::get<I>(args)))) {
    // Compute the shape of the result based on shapes of arguments
    ConstantSubscripts shape;
    int rank{0};
    const ConstantSubscripts *shapes[sizeof...(TA)]{
        &std::get<I>(args)->shape()...};
    const int ranks[sizeof...(TA)]{std::get<I>(args)->Rank()...};
    for (unsigned int i{0}; i < sizeof...(TA); ++i) {
      if (ranks[i] > 0) {
        if (rank == 0) {
          rank = ranks[i];
          shape = *shapes[i];
        } else {
          if (shape != *shapes[i]) {
            // TODO: Rank compatibility was already checked but it seems to be
            // the first place where the actual shapes are checked to be the
            // same. Shouldn't this be checked elsewhere so that this is also
            // checked for non constexpr call to elemental intrinsics function?
            context.messages().Say(
                "Arguments in elemental intrinsic function are not conformable"_err_en_US);
            return Expr<TR>{std::move(funcRef)};
          }
        }
      }
    }
    CHECK(rank == GetRank(shape));

    // Compute all the scalar values of the results
    std::vector<Scalar<TR>> results;
    if (TotalElementCount(shape) > 0) {
      ConstantBounds bounds{shape};
      ConstantSubscripts index(rank, 1);
      do {
        if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                          ScalarFuncWithContext<TR, TA...>>) {
          results.emplace_back(func(context,
              (ranks[I] ? std::get<I>(args)->At(index)
                        : std::get<I>(args)->GetScalarValue().value())...));
        } else if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                                 ScalarFunc<TR, TA...>>) {
          results.emplace_back(func(
              (ranks[I] ? std::get<I>(args)->At(index)
                        : std::get<I>(args)->GetScalarValue().value())...));
        }
      } while (bounds.IncrementSubscripts(index));
    }
    // Build and return constant result
    if constexpr (TR::category == TypeCategory::Character) {
      auto len{static_cast<ConstantSubscript>(
          results.size() ? results[0].length() : 0)};
      return Expr<TR>{Constant<TR>{len, std::move(results), std::move(shape)}};
    } else {
      return Expr<TR>{Constant<TR>{std::move(results), std::move(shape)}};
    }
  }
  return Expr<TR>{std::move(funcRef)};
}

template<typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFunc<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFunc, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}
template<typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFuncWithContext<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFuncWithContext, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}

static std::optional<std::int64_t> GetInt64Arg(
    const std::optional<ActualArgument> &arg) {
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

static std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &arg, std::int64_t defaultValue) {
  if (!arg) {
    return defaultValue;
  } else if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

template<typename A, typename B>
std::optional<std::vector<A>> GetIntegerVector(const B &x) {
  static_assert(std::is_integral_v<A>);
  if (const auto *someInteger{UnwrapExpr<Expr<SomeInteger>>(x)}) {
    return std::visit(
        [](const auto &typedExpr) -> std::optional<std::vector<A>> {
          using T = ResultType<decltype(typedExpr)>;
          if (const auto *constant{UnwrapConstantValue<T>(typedExpr)}) {
            if (constant->Rank() == 1) {
              std::vector<A> result;
              for (const auto &value : constant->values()) {
                result.push_back(static_cast<A>(value.ToInt64()));
              }
              return result;
            }
          }
          return std::nullopt;
        },
        someInteger->u);
  }
  return std::nullopt;
}

// Transform an intrinsic function reference that contains user errors
// into an intrinsic with the same characteristic but the "invalid" name.
// This to prevent generating warnings over and over if the expression
// gets re-folded.
template<typename T> Expr<T> MakeInvalidIntrinsic(FunctionRef<T> &&funcRef) {
  SpecificIntrinsic invalid{std::get<SpecificIntrinsic>(funcRef.proc().u)};
  invalid.name = "(invalid intrinsic function call)";
  return Expr<T>{FunctionRef<T>{ProcedureDesignator{std::move(invalid)},
      ActualArguments{ActualArgument{AsGenericExpr(std::move(funcRef))}}}};
}

template<typename T>
Expr<T> Reshape(FoldingContext &context, FunctionRef<T> &&funcRef) {
  auto args{funcRef.arguments()};
  CHECK(args.size() == 4);
  const auto *source{UnwrapConstantValue<T>(args[0])};
  const auto *pad{UnwrapConstantValue<T>(args[2])};
  std::optional<std::vector<ConstantSubscript>> shape{
      GetIntegerVector<ConstantSubscript>(args[1])};
  std::optional<std::vector<int>> order{GetIntegerVector<int>(args[3])};
  if (!source || !shape || (args[2] && !pad) || (args[3] && !order)) {
    return Expr<T>{std::move(funcRef)};  // Non-constant arguments
  } else if (!IsValidShape(shape.value())) {
    context.messages().Say("Invalid SHAPE in RESHAPE"_en_US);
  } else {
    int rank{GetRank(shape.value())};
    std::size_t resultElements{TotalElementCount(shape.value())};
    std::optional<std::vector<int>> dimOrder;
    if (order) {
      dimOrder = ValidateDimensionOrder(rank, *order);
    }
    std::vector<int> *dimOrderPtr{dimOrder ? &dimOrder.value() : nullptr};
    if (order && !dimOrder) {
      context.messages().Say("Invalid ORDER in RESHAPE"_en_US);
    } else if (resultElements > source->size() && (!pad || pad->empty())) {
      context.messages().Say("Too few SOURCE elements in RESHAPE and PAD"
                             "is not present or has null size"_en_US);
    } else {
      Constant<T> result{!source->empty()
              ? source->Reshape(std::move(shape.value()))
              : pad->Reshape(std::move(shape.value()))};
      ConstantSubscripts subscripts{result.lbounds()};
      auto copied{result.CopyFrom(*source,
          std::min(source->size(), resultElements), subscripts, dimOrderPtr)};
      if (copied < resultElements) {
        CHECK(pad);
        copied += result.CopyFrom(
            *pad, resultElements - copied, subscripts, dimOrderPtr);
      }
      CHECK(copied == resultElements);
      return Expr<T>{std::move(result)};
    }
  }
  // Invalid, prevent re-folding
  return MakeInvalidIntrinsic(std::move(funcRef));
}

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> LBOUND(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  ActualArguments &args{funcRef.arguments()};
  if (const auto *array{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    if (int rank{array->Rank()}; rank > 0) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (auto dim64{GetInt64Arg(args[1])}) {
          if (*dim64 < 1 || *dim64 > rank) {
            context.messages().Say("DIM=%jd dimension is out of range for "
                                   "rank-%d array"_en_US,
                static_cast<std::intmax_t>(*dim64), rank);
            return MakeInvalidIntrinsic<T>(std::move(funcRef));
          } else {
            dim = *dim64 - 1;  // 1-based to 0-based
          }
        } else {
          // DIM= is present but not constant
          return Expr<T>{std::move(funcRef)};
        }
      }
      bool lowerBoundsAreOne{true};
      if (auto named{ExtractNamedEntity(*array)}) {
        const Symbol &symbol{named->GetLastSymbol()};
        if (symbol.Rank() == rank) {
          lowerBoundsAreOne = false;
          if (dim) {
            return Fold(context,
                ConvertToType<T>(GetLowerBound(context, *named, *dim)));
          } else if (auto extents{
                         AsExtentArrayExpr(GetLowerBounds(context, *named))}) {
            return Fold(context,
                ConvertToType<T>(Expr<ExtentType>{std::move(*extents)}));
          }
        } else {
          lowerBoundsAreOne = symbol.Rank() == 0;  // LBOUND(array%component)
        }
      }
      if (lowerBoundsAreOne) {
        if (dim) {
          return Expr<T>{1};
        } else {
          std::vector<Scalar<T>> ones(rank, Scalar<T>{1});
          return Expr<T>{
              Constant<T>{std::move(ones), ConstantSubscripts{rank}}};
        }
      }
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> UBOUND(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  ActualArguments &args{funcRef.arguments()};
  if (auto *array{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    if (int rank{array->Rank()}; rank > 0) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (auto dim64{GetInt64Arg(args[1])}) {
          if (*dim64 < 1 || *dim64 > rank) {
            context.messages().Say("DIM=%jd dimension is out of range for "
                                   "rank-%d array"_en_US,
                static_cast<std::intmax_t>(*dim64), rank);
            return MakeInvalidIntrinsic<T>(std::move(funcRef));
          } else {
            dim = *dim64 - 1;  // 1-based to 0-based
          }
        } else {
          // DIM= is present but not constant
          return Expr<T>{std::move(funcRef)};
        }
      }
      bool takeBoundsFromShape{true};
      if (auto named{ExtractNamedEntity(*array)}) {
        const Symbol &symbol{named->GetLastSymbol()};
        if (symbol.Rank() == rank) {
          takeBoundsFromShape = false;
          if (dim) {
            if (semantics::IsAssumedSizeArray(symbol) && *dim == rank) {
              return Expr<T>{-1};
            } else if (auto ub{GetUpperBound(context, *named, *dim)}) {
              return Fold(context, ConvertToType<T>(std::move(*ub)));
            }
          } else {
            Shape ubounds{GetUpperBounds(context, *named)};
            if (semantics::IsAssumedSizeArray(symbol)) {
              CHECK(!ubounds.back());
              ubounds.back() = ExtentExpr{-1};
            }
            if (auto extents{AsExtentArrayExpr(ubounds)}) {
              return Fold(context,
                  ConvertToType<T>(Expr<ExtentType>{std::move(*extents)}));
            }
          }
        } else {
          takeBoundsFromShape = symbol.Rank() == 0;  // UBOUND(array%component)
        }
      }
      if (takeBoundsFromShape) {
        if (auto shape{GetShape(context, *array)}) {
          if (dim) {
            if (auto &dimSize{shape->at(*dim)}) {
              return Fold(context,
                  ConvertToType<T>(Expr<ExtentType>{std::move(*dimSize)}));
            }
          } else if (auto shapeExpr{AsExtentArrayExpr(*shape)}) {
            return Fold(context, ConvertToType<T>(std::move(*shapeExpr)));
          }
        }
      }
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<typename T>
Expr<T> FoldMINorMAX(
    FoldingContext &context, FunctionRef<T> &&funcRef, Ordering order) {
  std::vector<Constant<T> *> constantArgs;
  for (auto &arg : funcRef.arguments()) {
    if (auto *cst{FoldConvertedArg<T>(context, arg)}) {
      constantArgs.push_back(cst);
    } else {
      return Expr<T>(std::move(funcRef));
    }
  }
  CHECK(constantArgs.size() > 0);
  Expr<T> result{std::move(*constantArgs[0])};
  for (std::size_t i{1}; i < constantArgs.size(); ++i) {
    Extremum<T> extremum{order, result, Expr<T>{std::move(*constantArgs[i])}};
    Expr<SomeType> newRes{Fold(context, AsGenericExpr(std::move(extremum)))};
    result = std::move(DEREF(UnwrapExpr<Expr<T>>(newRes)));
  }
  return result;
}
template<typename T>
Expr<T> FoldMerge(FoldingContext &context, FunctionRef<T> &&funcRef) {
  return FoldElementalIntrinsic<T, T, T, LogicalResult>(context,
      std::move(funcRef),
      ScalarFunc<T, T, T, LogicalResult>(
          [](const Scalar<T> &ifTrue, const Scalar<T> &ifFalse,
              const Scalar<LogicalResult> &predicate) -> Scalar<T> {
            return predicate.IsTrue() ? ifTrue : ifFalse;
          }));
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&);
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Integer, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Real, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Complex, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Logical, KIND>> &&);
template<typename T> Expr<T> FoldOperation(FoldingContext &, Designator<T> &&);

template<typename T>
Expr<T> IntrinsicFunctionFolder<T>::Fold(
    FoldingContext &context, FunctionRef<T> &&funcRef) {
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    const std::string name{intrinsic->name};
    if (name == "reshape") {
      return Reshape(context, std::move(funcRef));
    }
    // TODO: other type independent transformational
    if constexpr (!std::is_same_v<T, SomeDerived>) {
      return FoldIntrinsicFunction(context, std::move(funcRef));
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  using Int4 = Type<TypeCategory::Integer, 4>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "abs") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&context](const Scalar<T> &i) -> Scalar<T> {
          typename Scalar<T>::ValueWithOverflow j{i.ABS()};
          if (j.overflow) {
            context.messages().Say(
                "abs(integer(kind=%d)) folding overflowed"_en_US, KIND);
          }
          return j.value;
        }));
  } else if (name == "bit_size") {
    return Expr<T>{Scalar<T>::bits};
  } else if (name == "count") {
    if (!args[1]) {  // TODO: COUNT(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<LogicalResult>(args[0])}) {
        std::int64_t result{0};
        for (const auto &element : constant->values()) {
          if (element.IsTrue()) {
            ++result;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "digits") {
    if (const auto *cx{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::DIGITS;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::DIGITS;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::DIGITS;
          },
          cx->u)};
    }
  } else if (name == "dim") {
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::DIM);
  } else if (name == "dshiftl" || name == "dshiftr") {
    const auto fptr{
        name == "dshiftl" ? &Scalar<T>::DSHIFTL : &Scalar<T>::DSHIFTR};
    // Third argument can be of any kind. However, it must be smaller or equal
    // than BIT_SIZE. It can be converted to Int4 to simplify.
    return FoldElementalIntrinsic<T, T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, T, Int4>(
            [&fptr](const Scalar<T> &i, const Scalar<T> &j,
                const Scalar<Int4> &shift) -> Scalar<T> {
              return std::invoke(fptr, i, j, static_cast<int>(shift.ToInt64()));
            }));
  } else if (name == "exponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [&funcRef, &context](const auto &x) -> Expr<T> {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return FoldElementalIntrinsic<T, TR>(context, std::move(funcRef),
                &Scalar<TR>::template EXPONENT<Scalar<T>>);
          },
          sx->u);
    } else {
      common::die("exponent argument must be real");
    }
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "iachar" || name == "ichar") {
    auto *someChar{UnwrapExpr<Expr<SomeCharacter>>(args[0])};
    CHECK(someChar);
    if (auto len{ToInt64(someChar->LEN())}) {
      if (len.value() != 1) {
        // Do not die, this was not checked before
        context.messages().Say(
            "Character in intrinsic function %s must have length one"_en_US,
            name);
      } else {
        return std::visit(
            [&funcRef, &context](const auto &str) -> Expr<T> {
              using Char = typename std::decay_t<decltype(str)>::Result;
              return FoldElementalIntrinsic<T, Char>(context,
                  std::move(funcRef),
                  ScalarFunc<T, Char>([](const Scalar<Char> &c) {
                    return Scalar<T>{CharacterUtils<Char::kind>::ICHAR(c)};
                  }));
            },
            someChar->u);
      }
    }
  } else if (name == "iand" || name == "ior" || name == "ieor") {
    auto fptr{&Scalar<T>::IAND};
    if (name == "iand") {  // done in fptr declaration
    } else if (name == "ior") {
      fptr = &Scalar<T>::IOR;
    } else if (name == "ieor") {
      fptr = &Scalar<T>::IEOR;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), ScalarFunc<T, T, T>(fptr));
  } else if (name == "ibclr" || name == "ibset" || name == "ishft" ||
      name == "shifta" || name == "shiftr" || name == "shiftl") {
    // Second argument can be of any kind. However, it must be smaller or
    // equal than BIT_SIZE. It can be converted to Int4 to simplify.
    auto fptr{&Scalar<T>::IBCLR};
    if (name == "ibclr") {  // done in fprt definition
    } else if (name == "ibset") {
      fptr = &Scalar<T>::IBSET;
    } else if (name == "ishft") {
      fptr = &Scalar<T>::ISHFT;
    } else if (name == "shifta") {
      fptr = &Scalar<T>::SHIFTA;
    } else if (name == "shiftr") {
      fptr = &Scalar<T>::SHIFTR;
    } else if (name == "shiftl") {
      fptr = &Scalar<T>::SHIFTL;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>(
            [&fptr](const Scalar<T> &i, const Scalar<Int4> &pos) -> Scalar<T> {
              return std::invoke(fptr, i, static_cast<int>(pos.ToInt64()));
            }));
  } else if (name == "int") {
    if (auto *expr{UnwrapExpr<Expr<SomeType>>(args[0])}) {
      return std::visit(
          [&](auto &&x) -> Expr<T> {
            using From = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<From, BOZLiteralConstant> ||
                IsNumericCategoryExpr<From>()) {
              return Fold(context, ConvertToType<T>(std::move(x)));
            }
            common::die("int() argument type not valid");
          },
          std::move(expr->u));
    }
  } else if (name == "int_ptr_kind") {
    return Expr<T>{8};
  } else if (name == "kind") {
    if constexpr (common::HasMember<T, IntegerTypes>) {
      return Expr<T>{args[0].value().GetType()->kind()};
    } else {
      common::die("kind() result not integral");
    }
  } else if (name == "lbound") {
    return LBOUND(context, std::move(funcRef));
  } else if (name == "leadz" || name == "trailz" || name == "poppar" ||
      name == "popcnt") {
    if (auto *sn{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return std::visit(
          [&funcRef, &context, &name](const auto &n) -> Expr<T> {
            using TI = typename std::decay_t<decltype(n)>::Result;
            if (name == "poppar") {
              return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                  ScalarFunc<T, TI>([](const Scalar<TI> &i) -> Scalar<T> {
                    return Scalar<T>{i.POPPAR() ? 1 : 0};
                  }));
            }
            auto fptr{&Scalar<TI>::LEADZ};
            if (name == "leadz") {  // done in fprt definition
            } else if (name == "trailz") {
              fptr = &Scalar<TI>::TRAILZ;
            } else if (name == "popcnt") {
              fptr = &Scalar<TI>::POPCNT;
            } else {
              common::die(
                  "missing case to fold intrinsic function %s", name.c_str());
            }
            return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                ScalarFunc<T, TI>([&fptr](const Scalar<TI> &i) -> Scalar<T> {
                  return Scalar<T>{std::invoke(fptr, i)};
                }));
          },
          sn->u);
    } else {
      common::die("leadz argument must be integer");
    }
  } else if (name == "len") {
    if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(args[0])}) {
      return std::visit(
          [&](auto &kx) {
            if (auto len{kx.LEN()}) {
              return Fold(context, ConvertToType<T>(*std::move(len)));
            } else {
              return Expr<T>{std::move(funcRef)};
            }
          },
          charExpr->u);
    } else {
      common::die("len() argument must be of character type");
    }
  } else if (name == "maskl" || name == "maskr") {
    // Argument can be of any kind but value has to be smaller than BIT_SIZE.
    // It can be safely converted to Int4 to simplify.
    const auto fptr{name == "maskl" ? &Scalar<T>::MASKL : &Scalar<T>::MASKR};
    return FoldElementalIntrinsic<T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, Int4>([&fptr](const Scalar<Int4> &places) -> Scalar<T> {
          return fptr(static_cast<int>(places.ToInt64()));
        }));
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "maxexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MAXEXPONENT};
          },
          sx->u);
    }
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "merge_bits") {
    return FoldElementalIntrinsic<T, T, T, T>(
        context, std::move(funcRef), &Scalar<T>::MERGE_BITS);
  } else if (name == "minexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MINEXPONENT};
          },
          sx->u);
    }
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "precision") {
    if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::PRECISION;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::PRECISION;
          },
          cx->u)};
    }
  } else if (name == "radix") {
    return Expr<T>{2};
  } else if (name == "range") {
    if (const auto *cx{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::RANGE;
          },
          cx->u)};
    }
  } else if (name == "rank") {
    // TODO assumed-rank dummy argument
    return Expr<T>{args[0].value().Rank()};
  } else if (name == "selected_char_kind") {
    if (const auto *chCon{UnwrapExpr<Constant<TypeOf<std::string>>>(args[0])}) {
      if (std::optional<std::string> value{chCon->GetScalarValue()}) {
        int defaultKind{
            context.defaults().GetDefaultKind(TypeCategory::Character)};
        return Expr<T>{SelectedCharKind(*value, defaultKind)};
      }
    }
  } else if (name == "selected_int_kind") {
    if (auto p{GetInt64Arg(args[0])}) {
      return Expr<T>{SelectedIntKind(*p)};
    }
  } else if (name == "selected_real_kind") {
    if (auto p{GetInt64ArgOr(args[0], 0)}) {
      if (auto r{GetInt64ArgOr(args[1], 0)}) {
        if (auto radix{GetInt64ArgOr(args[2], 2)}) {
          return Expr<T>{SelectedRealKind(*p, *r, *radix)};
        }
      }
    }
  } else if (name == "shape") {
    if (auto shape{GetShape(context, args[0])}) {
      if (auto shapeExpr{AsExtentArrayExpr(*shape)}) {
        return Fold(context, ConvertToType<T>(std::move(*shapeExpr)));
      }
    }
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>(
            [&context](const Scalar<T> &j, const Scalar<T> &k) -> Scalar<T> {
              typename Scalar<T>::ValueWithOverflow result{j.SIGN(k)};
              if (result.overflow) {
                context.messages().Say(
                    "sign(integer(kind=%d)) folding overflowed"_en_US, KIND);
              }
              return result.value;
            }));
  } else if (name == "size") {
    if (auto shape{GetShape(context, args[0])}) {
      if (auto &dimArg{args[1]}) {  // DIM= is present, get one extent
        if (auto dim{GetInt64Arg(args[1])}) {
          int rank{GetRank(*shape)};
          if (*dim >= 1 && *dim <= rank) {
            if (auto &extent{shape->at(*dim - 1)}) {
              return Fold(context, ConvertToType<T>(std::move(*extent)));
            }
          } else {
            context.messages().Say(
                "size(array,dim=%jd) dimension is out of range for rank-%d array"_en_US,
                static_cast<std::intmax_t>(*dim), static_cast<int>(rank));
          }
        }
      } else if (auto extents{common::AllElementsPresent(std::move(*shape))}) {
        // DIM= is absent; compute PRODUCT(SHAPE())
        ExtentExpr product{1};
        for (auto &&extent : std::move(*extents)) {
          product = std::move(product) * std::move(extent);
        }
        return Expr<T>{ConvertToType<T>(Fold(context, std::move(product)))};
      }
    }
  } else if (name == "ubound") {
    return UBOUND(context, std::move(funcRef));
  }
  // TODO:
  // ceiling, cshift, dot_product, eoshift,
  // findloc, floor, iall, iany, iparity, ibits, image_status, index, ishftc,
  // len_trim, matmul, maxloc, maxval,
  // minloc, minval, mod, modulo, nint, not, pack, product, reduce,
  // scan, sign, spread, sum, transfer, transpose, unpack, verify
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> ToReal(
    FoldingContext &context, Expr<SomeType> &&expr) {
  using Result = Type<TypeCategory::Real, KIND>;
  std::optional<Expr<Result>> result;
  std::visit(
      [&](auto &&x) {
        using From = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<From, BOZLiteralConstant>) {
          // Move the bits without any integer->real conversion
          From original{x};
          result = ConvertToType<Result>(std::move(x));
          const auto *constant{UnwrapExpr<Constant<Result>>(*result)};
          CHECK(constant);
          Scalar<Result> real{constant->GetScalarValue().value()};
          From converted{From::ConvertUnsigned(real.RawBits()).value};
          if (original != converted) {  // C1601
            context.messages().Say(
                "Nonzero bits truncated from BOZ literal constant in REAL intrinsic"_en_US);
          }
        } else if constexpr (IsNumericCategoryExpr<From>()) {
          result = Fold(context, ConvertToType<Result>(std::move(x)));
        } else {
          common::die("ToReal: bad argument expression");
        }
      },
      std::move(expr.u));
  return result.value();
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Real, KIND>;
  using ComplexT = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      (name == "atan" && args.size() == 1) || name == "atanh" ||
      name == "bessel_j0" || name == "bessel_j1" || name == "bessel_y0" ||
      name == "bessel_y1" || name == "cos" || name == "cosh" || name == "erf" ||
      name == "erfc" || name == "erfc_scaled" || name == "exp" ||
      name == "gamma" || name == "log" || name == "log10" ||
      name == "log_gamma" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    CHECK(args.size() == 1);
    if (auto callable{context.hostIntrinsicsLibrary()
                          .GetHostProcedureWrapper<Scalar, T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  }
  if (name == "atan" || name == "atan2" || name == "hypot" || name == "mod") {
    std::string localName{name == "atan2" ? "atan" : name};
    CHECK(args.size() == 2);
    if (auto callable{
            context.hostIntrinsicsLibrary()
                .GetHostProcedureWrapper<Scalar, T, T, T>(localName)}) {
      return FoldElementalIntrinsic<T, T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d), real(kind%d)) cannot be folded on host"_en_US,
          name, KIND, KIND);
    }
  } else if (name == "bessel_jn" || name == "bessel_yn") {
    if (args.size() == 2) {  // elemental
      // runtime functions use int arg
      using Int4 = Type<TypeCategory::Integer, 4>;
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, Int4, T>(name)}) {
        return FoldElementalIntrinsic<T, Int4, T>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "%s(integer(kind=4), real(kind=%d)) cannot be folded on host"_en_US,
            name, KIND);
      }
    }
  } else if (name == "abs") {
    // Argument can be complex or real
    if (auto *x{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), &Scalar<T>::ABS);
    } else if (auto *z{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, ComplexT>("abs")}) {
        return FoldElementalIntrinsic<T, ComplexT>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "abs(complex(kind=%d)) cannot be folded on host"_en_US, KIND);
      }
    } else {
      common::die(" unexpected argument type inside abs");
    }
  } else if (name == "aimag") {
    return FoldElementalIntrinsic<T, ComplexT>(
        context, std::move(funcRef), &Scalar<ComplexT>::AIMAG);
  } else if (name == "aint") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&name, &context](const Scalar<T> &x) -> Scalar<T> {
          ValueWithRealFlags<Scalar<T>> y{x.AINT()};
          if (y.flags.test(RealFlag::Overflow)) {
            context.messages().Say("%s intrinsic folding overflow"_en_US, name);
          }
          return y.value;
        }));
  } else if (name == "dprod") {
    if (auto *x{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      if (auto *y{UnwrapExpr<Expr<SomeReal>>(args[1])}) {
        return Fold(context,
            Expr<T>{Multiply<T>{ConvertToType<T>(std::move(*x)),
                ConvertToType<T>(std::move(*y))}});
      }
    }
    common::die("Wrong argument type in dprod()");
  } else if (name == "epsilon") {
    return Expr<T>{Scalar<T>::EPSILON()};
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "real") {
    if (auto *expr{args[0].value().UnwrapExpr()}) {
      return ToReal<KIND>(context, std::move(*expr));
    }
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::SIGN);
  } else if (name == "tiny") {
    return Expr<T>{Scalar<T>::TINY()};
  }
  // TODO: anint, cshift, dim, dot_product, eoshift, fraction, matmul,
  // maxval, minval, modulo, nearest, norm2, pack, product,
  // reduce, rrspacing, scale, set_exponent, spacing, spread,
  // sum, transfer, transpose, unpack, bessel_jn (transformational) and
  // bessel_yn (transformational)
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Complex, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      name == "atan" || name == "atanh" || name == "cos" || name == "cosh" ||
      name == "exp" || name == "log" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    if (auto callable{context.hostIntrinsicsLibrary()
                          .GetHostProcedureWrapper<Scalar, T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(complex(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  } else if (name == "conjg") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), &Scalar<T>::CONJG);
  } else if (name == "cmplx") {
    using Part = typename T::Part;
    if (args.size() == 1) {
      if (auto *x{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
        return Fold(context, ConvertToType<T>(std::move(*x)));
      }
      Expr<SomeType> re{std::move(*args[0].value().UnwrapExpr())};
      Expr<SomeType> im{AsGenericExpr(Constant<Part>{Scalar<Part>{}})};
      return Fold(context,
          Expr<T>{ComplexConstructor<KIND>{ToReal<KIND>(context, std::move(re)),
              ToReal<KIND>(context, std::move(im))}});
    }
    CHECK(args.size() == 2 || args.size() == 3);
    Expr<SomeType> re{std::move(*args[0].value().UnwrapExpr())};
    Expr<SomeType> im{args[1] ? std::move(*args[1].value().UnwrapExpr())
                              : AsGenericExpr(Constant<Part>{Scalar<Part>{}})};
    return Fold(context,
        Expr<T>{ComplexConstructor<KIND>{ToReal<KIND>(context, std::move(re)),
            ToReal<KIND>(context, std::move(im))}});
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  }
  // TODO: cshift, dot_product, eoshift, matmul, pack, product,
  // reduce, spread, sum, transfer, transpose, unpack
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "all") {
    if (!args[1]) {  // TODO: ALL(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{true};
        for (const auto &element : constant->values()) {
          if (!element.IsTrue()) {
            result = false;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "any") {
    if (!args[1]) {  // TODO: ANY(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{false};
        for (const auto &element : constant->values()) {
          if (element.IsTrue()) {
            result = true;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "bge" || name == "bgt" || name == "ble" || name == "blt") {
    using LargestInt = Type<TypeCategory::Integer, 16>;
    static_assert(std::is_same_v<Scalar<LargestInt>, BOZLiteralConstant>);
    // Arguments do not have to be of the same integer type. Convert all
    // arguments to the biggest integer type before comparing them to
    // simplify.
    for (int i{0}; i <= 1; ++i) {
      if (auto *x{UnwrapExpr<Expr<SomeInteger>>(args[i])}) {
        *args[i] = AsGenericExpr(
            Fold(context, ConvertToType<LargestInt>(std::move(*x))));
      } else if (auto *x{UnwrapExpr<BOZLiteralConstant>(args[i])}) {
        *args[i] = AsGenericExpr(Constant<LargestInt>{std::move(*x)});
      }
    }
    auto fptr{&Scalar<LargestInt>::BGE};
    if (name == "bge") {  // done in fptr declaration
    } else if (name == "bgt") {
      fptr = &Scalar<LargestInt>::BGT;
    } else if (name == "ble") {
      fptr = &Scalar<LargestInt>::BLE;
    } else if (name == "blt") {
      fptr = &Scalar<LargestInt>::BLT;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, LargestInt, LargestInt>(context,
        std::move(funcRef),
        ScalarFunc<T, LargestInt, LargestInt>(
            [&fptr](const Scalar<LargestInt> &i, const Scalar<LargestInt> &j) {
              return Scalar<T>{std::invoke(fptr, i, j)};
            }));
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  }
  // TODO: btest, cshift, dot_product, eoshift, is_iostat_end,
  // is_iostat_eor, lge, lgt, lle, llt, logical, matmul, out_of_range,
  // pack, parity, reduce, spread, transfer, transpose, unpack
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Character, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Character, KIND>;
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    return FoldElementalIntrinsic<T, IntT>(context, std::move(funcRef),
        ScalarFunc<T, IntT>([](const Scalar<IntT> &i) {
          return CharacterUtils<KIND>::CHAR(i.ToUInt64());
        }));
  } else if (name == "adjustl") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTL);
  } else if (name == "adjustr") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTR);
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "new_line") {
    return Expr<T>{Constant<T>{CharacterUtils<KIND>::NEW_LINE()}};
  }
  // TODO: cshift, eoshift, maxval, minval, pack, reduce,
  // repeat, spread, transfer, transpose, trim, unpack
  return Expr<T>{std::move(funcRef)};
}

FOR_EACH_INTRINSIC_KIND(template class IntrinsicFunctionFolder, )
template class IntrinsicFunctionFolder<SomeDerived>;
}
