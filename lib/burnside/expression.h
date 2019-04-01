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

#ifndef FORTRAN_BURNSIDE_EXPRESSION_H_
#define FORTRAN_BURNSIDE_EXPRESSION_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <tuple>
#include <variant>

/// Conversion of expressions with type Fortran::evaluate::Expr<A> into the FIR
/// dialect of MLIR.
///
/// Fortran expressions appear in syntactic contexts such as, for example,
/// assignment statements.  The semantics of these expressions fall into two
/// cases: (1) an expression that computes a value and (2) an expression that
/// computes an address in storage.  In the FIR dialect, case (1) is represented
/// as a "fir.apply_expr" operation. Case (2) is represented as a
/// "fir.locate_expr" operation.

namespace mlir {
class MLIRContext;
class Value;
class OpBuilder;
}

namespace Fortran::evaluate {
template<typename> class Expr;
struct SomeType;
}

namespace Fortran::burnside {

// In the Fortran::burnside namespace, the code will default follow the
// LLVM/MLIR coding standards

class ApplyExpr;
class LocateExpr;

enum ExprType {
  ET_NONE,  // Expr is unclassified type
  ET_ArrayCtor,
  ET_ArrayRef,
  ET_CoarrayRef,
  ET_ComplexPart,
  ET_Component,
  ET_Constant,
  ET_DescriptorInquiry,
  ET_FunctionRef,
  ET_NullPointer,
  ET_Operation,
  ET_Relational,
  ET_StructureCtor,
  ET_Substring,
  ET_Symbol,
  ET_TypeParamInquiry
};

class SymMap;
using Args = llvm::SmallVector<mlir::Value *, 8>;
using Dict = std::map<unsigned, void *>;
using Values = std::tuple<Args, Dict, ExprType>;
using RewriteVals = mlir::Value *;
using OperandTy = llvm::ArrayRef<mlir::Value *>;
using SomeExpr = evaluate::Expr<evaluate::SomeType>;

inline Args getArgs(const Values &values) { return std::get<Args>(values); }
inline Dict getDict(const Values &values) { return std::get<Dict>(values); }
inline ExprType getExprType(const Values &values) {
  return std::get<ExprType>(values);
}

/// Convert an Expr<T> in its implicit dataflow arguments
Values translateSomeExpr(
    mlir::OpBuilder *bldr, const SomeExpr *exp, SymMap &map);

Values translateSomeAddrExpr(
    mlir::OpBuilder *bldr, const SomeExpr *exp, SymMap &map);

}  // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_EXPRESSION_H_
