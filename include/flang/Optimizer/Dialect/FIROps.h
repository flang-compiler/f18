//===-- Optimizer/Dialect/FIROps.h - FIR operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIROPS_H
#define OPTIMIZER_DIALECT_FIROPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace fir {

class FirEndOp;

enum class CmpFPredicate {
  FirstValidValue,
  // Always false
  AlwaysFalse = FirstValidValue,
  // Ordered comparisons
  OEQ,
  OGT,
  OGE,
  OLT,
  OLE,
  ONE,
  // Both ordered
  ORD,
  // Unordered comparisons
  UEQ,
  UGT,
  UGE,
  ULT,
  ULE,
  UNE,
  // Any unordered
  UNO,
  // Always true
  AlwaysTrue,
  // Number of predicates.
  NumPredicates
};

ParseResult isValidCaseAttr(Attribute attr);
unsigned getCaseArgumentOffset(ArrayRef<Attribute> cases, unsigned dest);
ParseResult parseSelector(OpAsmParser *parser, OperationState *result,
                          OpAsmParser::OperandType &selector, mlir::Type &type);

void buildCmpFOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value lhs, Value rhs);
void buildCmpCOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value lhs, Value rhs);
ParseResult parseCmpfOp(OpAsmParser &parser, OperationState &result);
ParseResult parseCmpcOp(OpAsmParser &parser, OperationState &result);

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.h.inc"

LoopOp getForInductionVarOwner(Value val);

bool isReferenceLike(mlir::Type type);

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIROPS_H
