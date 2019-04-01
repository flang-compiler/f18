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

#ifndef FORTRAN_BURNSIDE_FE_HELPER_H_
#define FORTRAN_BURNSIDE_FE_HELPER_H_

/// Traversal and coversion of various Fortran::parser data structures into the
/// FIR dialect of MLIR. These traversals are isolated in this file to hopefully
/// make maintenance easier.

#include "../common/Fortran.h"
#include "mlir/IR/Types.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
}

namespace Fortran::evaluate {
struct DataRef;
template<typename> class Designator;
template<typename> class Expr;
template<common::TypeCategory> struct SomeKind;
struct SomeType;
template<common::TypeCategory, int> class Type;
}

namespace Fortran::parser {
class CharBlock;
}

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::burnside {

using SomeExpr = evaluate::Expr<evaluate::SomeType>;

constexpr common::TypeCategory IntegerCat{common::TypeCategory::Integer};
constexpr common::TypeCategory RealCat{common::TypeCategory::Real};
constexpr common::TypeCategory ComplexCat{common::TypeCategory::Complex};
constexpr common::TypeCategory CharacterCat{common::TypeCategory::Character};
constexpr common::TypeCategory LogicalCat{common::TypeCategory::Logical};
constexpr common::TypeCategory DerivedCat{common::TypeCategory::Derived};

// In the Fortran::burnside namespace, the code will default follow the
// LLVM/MLIR coding standards

mlir::Location dummyLoc(mlir::MLIRContext *ctxt);

/// Translate a CharBlock position to (source-file, line, column)
mlir::Location parserPosToLoc(
    mlir::MLIRContext &context, const parser::CharBlock &position);

mlir::Type genTypeFromCategoryAndKind(
    mlir::MLIRContext *ctxt, common::TypeCategory tc, int kind);
mlir::Type genTypeFromCategory(
    mlir::MLIRContext *ctxt, common::TypeCategory tc);

mlir::Type translateDataRefToFIRType(
    mlir::MLIRContext *ctxt, const evaluate::DataRef &dataRef);

template<common::TypeCategory TC, int KIND>
inline mlir::Type translateDesignatorToFIRType(mlir::MLIRContext *ctxt,
    const evaluate::Designator<evaluate::Type<TC, KIND>> &) {
  return genTypeFromCategoryAndKind(ctxt, TC, KIND);
}

template<common::TypeCategory TC>
inline mlir::Type translateDesignatorToFIRType(mlir::MLIRContext *ctxt,
    const evaluate::Designator<evaluate::SomeKind<TC>> &) {
  return genTypeFromCategory(ctxt, TC);
}

mlir::Type translateSomeExprToFIRType(
    mlir::MLIRContext *ctxt, const SomeExpr *expr);

mlir::Type translateSymbolToFIRType(
    mlir::MLIRContext *ctxt, const semantics::Symbol *symbol);

mlir::Type convertReal(int KIND, mlir::MLIRContext *context);

}  // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_FE_HELPER_H_
