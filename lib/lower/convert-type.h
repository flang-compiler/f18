//===-- lib/burnside/convert-type.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#ifndef FORTRAN_LOWER_CONVERT_TYPE_H_
#define FORTRAN_LOWER_CONVERT_TYPE_H_

/// Conversion of front-end TYPE, KIND, ATTRIBUTE (TKA) information to FIR/MLIR.
/// This is meant to be the single point of truth (SPOT) for all type
/// conversions when lowering to FIR.  This implements all lowering of parse
/// tree TKA to the FIR type system. If one is converting front-end types and
/// not using one of the routines provided here, it's being done wrong.
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

#include "../common/Fortran.h"
#include "mlir/IR/Types.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
template <typename>
class Reference;
} // namespace common

namespace evaluate {
struct DataRef;
template <typename>
class Designator;
template <typename>
class Expr;
template <common::TypeCategory>
struct SomeKind;
struct SomeType;
template <common::TypeCategory, int>
class Type;
} // namespace evaluate

namespace parser {
class CharBlock;
class CookedSource;
} // namespace parser

namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

constexpr common::TypeCategory IntegerCat{common::TypeCategory::Integer};
constexpr common::TypeCategory RealCat{common::TypeCategory::Real};
constexpr common::TypeCategory ComplexCat{common::TypeCategory::Complex};
constexpr common::TypeCategory CharacterCat{common::TypeCategory::Character};
constexpr common::TypeCategory LogicalCat{common::TypeCategory::Logical};
constexpr common::TypeCategory DerivedCat{common::TypeCategory::Derived};

mlir::Type getFIRType(mlir::MLIRContext *ctxt,
                      common::IntrinsicTypeDefaultKinds const &defaults,
                      common::TypeCategory tc, int kind);
mlir::Type getFIRType(mlir::MLIRContext *ctxt,
                      common::IntrinsicTypeDefaultKinds const &defaults,
                      common::TypeCategory tc);

mlir::Type
translateDataRefToFIRType(mlir::MLIRContext *ctxt,
                          common::IntrinsicTypeDefaultKinds const &defaults,
                          const evaluate::DataRef &dataRef);

template <common::TypeCategory TC, int KIND>
inline mlir::Type translateDesignatorToFIRType(
    mlir::MLIRContext *ctxt, common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::Type<TC, KIND>> &) {
  return getFIRType(ctxt, defaults, TC, KIND);
}

template <common::TypeCategory TC>
inline mlir::Type translateDesignatorToFIRType(
    mlir::MLIRContext *ctxt, common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::SomeKind<TC>> &) {
  return getFIRType(ctxt, defaults, TC);
}

mlir::Type
translateSomeExprToFIRType(mlir::MLIRContext *ctxt,
                           common::IntrinsicTypeDefaultKinds const &defaults,
                           const SomeExpr *expr);

mlir::Type
translateSymbolToFIRType(mlir::MLIRContext *ctxt,
                         common::IntrinsicTypeDefaultKinds const &defaults,
                         const SymbolRef symbol);

mlir::FunctionType translateSymbolToFIRFunctionType(
    mlir::MLIRContext *ctxt, common::IntrinsicTypeDefaultKinds const &defaults,
    const SymbolRef symbol);

mlir::Type convertReal(mlir::MLIRContext *ctxt, int KIND);

// Given a ReferenceType of a base type, returns the ReferenceType to
// the SequenceType of this base type.
// The created SequenceType has one dimension of unknown extent.
// This is useful to do pointer arithmetic using fir::CoordinateOp that requires
// a memory reference to a sequence type.
mlir::Type getSequenceRefType(mlir::Type referenceType);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_TYPE_H_
