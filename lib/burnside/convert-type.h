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

#ifndef FORTRAN_BURNSIDE_CONVERT_TYPE_H_
#define FORTRAN_BURNSIDE_CONVERT_TYPE_H_

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
}

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
template<typename> class Reference;
}  // common

namespace evaluate {
struct DataRef;
template<typename> class Designator;
template<typename> class Expr;
template<common::TypeCategory> struct SomeKind;
struct SomeType;
template<common::TypeCategory, int> class Type;
}  // evaluate

namespace parser {
class CharBlock;
class CookedSource;
}  // parser

namespace semantics {
class Symbol;
}  // semantics

namespace burnside {

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

constexpr common::TypeCategory IntegerCat{common::TypeCategory::Integer};
constexpr common::TypeCategory RealCat{common::TypeCategory::Real};
constexpr common::TypeCategory ComplexCat{common::TypeCategory::Complex};
constexpr common::TypeCategory CharacterCat{common::TypeCategory::Character};
constexpr common::TypeCategory LogicalCat{common::TypeCategory::Logical};
constexpr common::TypeCategory DerivedCat{common::TypeCategory::Derived};

mlir::Type getFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults, common::TypeCategory tc,
    int kind);
mlir::Type getFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults, common::TypeCategory tc);

mlir::Type translateDataRefToFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::DataRef &dataRef);

template<common::TypeCategory TC, int KIND>
inline mlir::Type translateDesignatorToFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::Type<TC, KIND>> &) {
  return getFIRType(ctxt, defaults, TC, KIND);
}

template<common::TypeCategory TC>
inline mlir::Type translateDesignatorToFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::SomeKind<TC>> &) {
  return getFIRType(ctxt, defaults, TC);
}

mlir::Type translateSomeExprToFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults, const SomeExpr *expr);

mlir::Type translateSymbolToFIRType(mlir::MLIRContext *ctxt,
    common::IntrinsicTypeDefaultKinds const &defaults, const SymbolRef symbol);

mlir::Type convertReal(mlir::MLIRContext *ctxt, int KIND);

}  // burnside
}  // Fortran

#endif  // FORTRAN_BURNSIDE_CONVERT_TYPE_H_
