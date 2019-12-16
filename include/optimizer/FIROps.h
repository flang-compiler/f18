//===-- FIROps.h - FIR operations -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_FIROPS_H
#define OPTIMIZER_FIROPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using llvm::ArrayRef;
using llvm::StringRef;

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

/// `fir.global` is a typed symbol with an optional list of initializers.
class GlobalOp
    : public mlir::Op<GlobalOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                      OpTrait::IsIsolatedFromAbove,
                      OpTrait::SingleBlockImplicitTerminator<FirEndOp>::Impl> {
public:
  using Op::Op;
  using Op::print;

  static llvm::StringRef getOperationName() { return "fir.global"; }
  static llvm::StringRef getTypeAttrName() { return "type"; }

  static void build(mlir::Builder *builder, OperationState &result,
                    llvm::StringRef name, mlir::Type type,
                    llvm::ArrayRef<NamedAttribute> attrs);

  static GlobalOp create(Location loc, llvm::StringRef name, mlir::Type type,
                         llvm::ArrayRef<NamedAttribute> attrs = {});

  /// Operation hooks.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  mlir::Type getType() {
    return getAttrOfType<TypeAttr>(getTypeAttrName()).getValue();
  }

  void appendInitialValue(mlir::Operation *op);

private:
  mlir::Region &front();
};

/// `fir.dispatch_table` is an untyped symbol that is a list of associations
/// between method identifiers and a FuncOp symbol.
class DispatchTableOp
    : public mlir::Op<DispatchTableOp, OpTrait::ZeroOperands,
                      OpTrait::ZeroResult, OpTrait::IsIsolatedFromAbove,
                      OpTrait::SingleBlockImplicitTerminator<FirEndOp>::Impl> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.dispatch_table"; }

  static void build(mlir::Builder *builder, OperationState *result,
                    llvm::StringRef name, mlir::Type type,
                    llvm::ArrayRef<NamedAttribute> attrs);

  /// Operation hooks.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  void appendTableEntry(mlir::Operation *op);

private:
  mlir::Region &front();
};

ParseResult isValidCaseAttr(mlir::Attribute attr);
unsigned getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                               unsigned dest);
ParseResult parseSelector(OpAsmParser *parser, OperationState *result,
                          OpAsmParser::OperandType &selector, mlir::Type &type);

void buildCmpFOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value lhs, Value rhs);
void buildCmpCOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value lhs, Value rhs);
ParseResult parseCmpfOp(OpAsmParser &parser, OperationState &result);
ParseResult parseCmpcOp(OpAsmParser &parser, OperationState &result);

#define GET_OP_CLASSES
#include "optimizer/FIROps.h.inc"

LoopOp getForInductionVarOwner(mlir::Value val);

bool isReferenceLike(mlir::Type type);

} // namespace fir

#endif // OPTIMIZER_FIROPS_H
