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

#include "fir/FIROps.h"
#include "fir/Attribute.h"
#include "fir/FIRType.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSwitch.h"

namespace L = llvm;
namespace M = mlir;

namespace fir {

// AllocaOp

M::Type AllocaOp::getAllocatedType() {
  // return getAttrOfType<M::TypeAttr>("in_type").getType();
  return getType().cast<ReferenceType>().getEleTy();
}

M::Type AllocaOp::getRefTy(M::Type ty) { return ReferenceType::get(ty); }

// AllocMemOp

M::Type AllocMemOp::getAllocatedType() {
  // return getAttrOfType<M::TypeAttr>("in_type").getType();
  return getType().cast<HeapType>().getEleTy();
}

M::Type AllocMemOp::getRefTy(M::Type ty) { return HeapType::get(ty); }

// BoxDimsOp

/// Get the result types packed in a tuple tuple
M::Type BoxDimsOp::getTupleType() {
  L::SmallVector<M::Type, 3> triple{getResult(0)->getType(),
                                    getResult(1)->getType(),
                                    getResult(2)->getType()};
  return mlir::TupleType::get(triple, getContext());
}

// CallOp

void printCallOp(M::OpAsmPrinter &p, fir::CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();
  p << op.getOperationName() << ' ';
  if (isDirect)
    p << callee.getValue();
  else
    p << *op.getOperand(0);
  p << '(';
  p.printOperands(L::drop_begin(op.getOperands(), isDirect ? 0 : 1));
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});
  L::SmallVector<Type, 1> resultTypes(op.getResultTypes());
  L::SmallVector<Type, 8> argTypes(
      L::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << FunctionType::get(argTypes, resultTypes, op.getContext());
}

M::ParseResult parseCallOp(M::OpAsmParser &parser, M::OperationState &result) {
  L::SmallVector<M::OpAsmParser::OperandType, 8> operands;

  if (parser.parseOperandList(operands))
    return M::failure();
  bool isDirect = operands.empty();
  SmallVector<NamedAttribute, 4> attrs;
  SymbolRefAttr funcAttr;

  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", attrs))
      return M::failure();
  Type type;

  if (parser.parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return M::failure();

  auto funcType = type.dyn_cast<M::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return M::failure();
  } else {
    auto funcArgs =
        L::ArrayRef<M::OpAsmParser::OperandType>(operands).drop_front();
    L::SmallVector<M::Value *, 8> resultArgs(
        result.operands.begin() + (result.operands.empty() ? 0 : 1),
        result.operands.end());
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), resultArgs))
      return M::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return M::success();
}

// CmpfOp

fir::CmpFPredicate CmpfOp::getPredicateByName(llvm::StringRef name) {
  return llvm::StringSwitch<fir::CmpFPredicate>(name)
      .Case("false", CmpFPredicate::AlwaysFalse)
      .Case("oeq", CmpFPredicate::OEQ)
      .Case("ogt", CmpFPredicate::OGT)
      .Case("oge", CmpFPredicate::OGE)
      .Case("olt", CmpFPredicate::OLT)
      .Case("ole", CmpFPredicate::OLE)
      .Case("one", CmpFPredicate::ONE)
      .Case("ord", CmpFPredicate::ORD)
      .Case("ueq", CmpFPredicate::UEQ)
      .Case("ugt", CmpFPredicate::UGT)
      .Case("uge", CmpFPredicate::UGE)
      .Case("ult", CmpFPredicate::ULT)
      .Case("ule", CmpFPredicate::ULE)
      .Case("une", CmpFPredicate::UNE)
      .Case("uno", CmpFPredicate::UNO)
      .Case("true", CmpFPredicate::AlwaysTrue)
      .Default(CmpFPredicate::NumPredicates);
}

void buildCmpFOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value *lhs, Value *rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder->getI1Type());
  result.addAttribute(
      CmpfOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

template <typename OPTY>
void printCmpOp(OpAsmPrinter &p, OPTY op) {
  static const char *predicateNames[] = {
      /*AlwaysFalse*/ "false",
      /*OEQ*/ "oeq",
      /*OGT*/ "ogt",
      /*OGE*/ "oge",
      /*OLT*/ "olt",
      /*OLE*/ "ole",
      /*ONE*/ "one",
      /*ORD*/ "ord",
      /*UEQ*/ "ueq",
      /*UGT*/ "ugt",
      /*UGE*/ "uge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UNE*/ "une",
      /*UNO*/ "uno",
      /*AlwaysTrue*/ "true",
  };
  static_assert(std::extent<decltype(predicateNames)>::value ==
                    (size_t)fir::CmpFPredicate::NumPredicates,
                "wrong number of predicate names");
  p << op.getOperationName() << ' ';
  auto predicateValue =
      op.template getAttrOfType<M::IntegerAttr>(OPTY::getPredicateAttrName())
          .getInt();
  assert(predicateValue >= static_cast<int>(CmpFPredicate::FirstValidValue) &&
         predicateValue < static_cast<int>(CmpFPredicate::NumPredicates) &&
         "unknown predicate index");
  Builder b(op.getContext());
  auto predicateStringAttr = b.getStringAttr(predicateNames[predicateValue]);
  p.printAttribute(predicateStringAttr);
  p << ", ";
  p.printOperand(op.lhs());
  p << ", ";
  p.printOperand(op.rhs());
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.lhs()->getType();
}

void printCmpfOp(OpAsmPrinter &p, CmpfOp op) { printCmpOp(p, op); }

template <typename OPTY>
M::ParseResult parseCmpOp(M::OpAsmParser &parser, M::OperationState &result) {
  L::SmallVector<M::OpAsmParser::OperandType, 2> ops;
  L::SmallVector<M::NamedAttribute, 4> attrs;
  M::Attribute predicateNameAttr;
  M::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<M::StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  L::StringRef predicateName =
      predicateNameAttr.cast<M::StringAttr>().getValue();
  auto predicate = CmpfOp::getPredicateByName(predicateName);
  if (predicate == CmpFPredicate::NumPredicates)
    return parser.emitError(parser.getNameLoc(),
                            "unknown comparison predicate \"" + predicateName +
                                "\"");

  auto builder = parser.getBuilder();
  M::Type i1Type = builder.getI1Type();
  attrs[0].second = builder.getI64IntegerAttr(static_cast<int64_t>(predicate));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return success();
}

M::ParseResult parseCmpfOp(M::OpAsmParser &parser, M::OperationState &result) {
  return parseCmpOp<fir::CmpfOp>(parser, result);
}

// CmpcOp

void buildCmpCOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value *lhs, Value *rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder->getI1Type());
  result.addAttribute(
      CmpcOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

void printCmpcOp(OpAsmPrinter &p, CmpcOp op) { printCmpOp(p, op); }

M::ParseResult parseCmpcOp(M::OpAsmParser &parser, M::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

// DispatchOp

M::FunctionType DispatchOp::getFunctionType() {
  auto attr = getAttr("fn_type").cast<M::TypeAttr>();
  return attr.getValue().cast<M::FunctionType>();
}

// DispatchTableOp

void DispatchTableOp::build(M::Builder *builder, M::OperationState *result,
                            L::StringRef name, M::Type type,
                            L::ArrayRef<M::NamedAttribute> attrs) {
  result->addAttribute("method", builder->getStringAttr(name));
  for (const auto &pair : attrs)
    result->addAttribute(pair.first, pair.second);
}

M::ParseResult DispatchTableOp::parse(M::OpAsmParser &parser,
                                      M::OperationState &result) {
  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "method", result.attributes))
    return failure();

  // Convert the parsed name attr into a string attr.
  result.attributes.back().second =
      parser.getBuilder().getStringAttr(nameAttr.getRootReference());

  // Parse the optional table body.
  M::Region *body = result.addRegion();
  if (parser.parseOptionalRegion(*body,
                                 L::ArrayRef<M::OpAsmParser::OperandType>{},
                                 L::ArrayRef<M::Type>{}))
    return M::failure();

  ensureTerminator(*body, parser.getBuilder(), result.location);
  return M::success();
}

void DispatchTableOp::print(M::OpAsmPrinter &p) {
  auto tableName = getAttrOfType<StringAttr>("method").getValue();
  p << getOperationName() << " @" << tableName;

  Region &body = getOperation()->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

M::LogicalResult DispatchTableOp::verify() { return M::success(); }

M::Region &DispatchTableOp::front() {
  return this->getOperation()->getRegion(0);
}

void DispatchTableOp::appendTableEntry(M::Operation *op) {
  assert(M::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  front().front().push_back(op);
}

// GlobalOp

void GlobalOp::build(M::Builder *builder, M::OperationState *result,
                     L::StringRef name, M::Type type,
                     L::ArrayRef<M::NamedAttribute> attrs) {
  result->addAttribute(getTypeAttrName(), M::TypeAttr::get(type));
  result->addAttribute(M::SymbolTable::getSymbolAttrName(),
                       builder->getStringAttr(name));
  for (const auto &pair : attrs)
    result->addAttribute(pair.first, pair.second);
}

M::ParseResult GlobalOp::parse(M::OpAsmParser &parser,
                               M::OperationState &result) {
  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, M::SymbolTable::getSymbolAttrName(),
                            result.attributes))
    return failure();

  auto &builder = parser.getBuilder();
  auto name = nameAttr.getRootReference();
  result.attributes.back().second = builder.getStringAttr(name);

  if (!parser.parseOptionalKeyword("constant")) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute("constant", builder.getUnitAttr());
  }

  M::Type globalType;
  if (parser.parseColonType(globalType))
    return M::failure();

  result.addAttribute(getTypeAttrName(), M::TypeAttr::get(globalType));

  // Parse the optional initializer body.
  M::Region *body = result.addRegion();
  if (parser.parseOptionalRegion(*body,
                                 L::ArrayRef<M::OpAsmParser::OperandType>{},
                                 L::ArrayRef<M::Type>{}))
    return M::failure();

  ensureTerminator(*body, builder, result.location);
  return M::success();
}

void GlobalOp::print(M::OpAsmPrinter &p) {
  auto varName =
      getAttrOfType<StringAttr>(M::SymbolTable::getSymbolAttrName()).getValue();
  p << getOperationName() << " @" << varName;
  if (getAttr("constant"))
    p << " constant";
  p << " : ";
  p.printType(getType());
  Region &body = getOperation()->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

M::LogicalResult GlobalOp::verify() { return M::success(); }

void GlobalOp::appendInitialValue(M::Operation *op) {
  front().front().push_back(op);
}

M::Region &GlobalOp::front() { return this->getOperation()->getRegion(0); }

// LoadOp

/// Get the element type of a reference like type; otherwise null
M::Type elementTypeOf(M::Type ref) {
  if (auto r = ref.dyn_cast_or_null<ReferenceType>())
    return r.getEleTy();
  if (auto r = ref.dyn_cast_or_null<PointerType>())
    return r.getEleTy();
  if (auto r = ref.dyn_cast_or_null<HeapType>())
    return r.getEleTy();
  return {};
}

M::ParseResult LoadOp::getElementOf(M::Type &ele, M::Type ref) {
  if (M::Type r = elementTypeOf(ref)) {
    ele = r;
    return M::success();
  }
  return M::failure();
}

// LoopOp

void LoopOp::build(M::Builder *builder, M::OperationState &result, M::Value *lb,
                   M::Value *ub, L::ArrayRef<M::Value *> step) {
  if (step.empty())
    result.addOperands({lb, ub});
  else
    result.addOperands({lb, ub, step[0]});
  M::Region *bodyRegion = result.addRegion();
  LoopOp::ensureTerminator(*bodyRegion, *builder, result.location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

M::ParseResult parseLoopOp(M::OpAsmParser &parser, M::OperationState &result) {
  auto &builder = parser.getBuilder();
  M::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return M::failure();

  // Parse loop bounds.
  M::Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands))
    return M::failure();

  if (parser.parseOptionalKeyword(LoopOp::getStepKeyword()))
    result.addAttribute(LoopOp::getStepKeyword(),
                        builder.getIntegerAttr(builder.getIndexType(), 1));
  else if (parser.parseOperand(step) ||
           parser.resolveOperand(step, indexType, result.operands))
    return M::failure();

  if (!parser.parseOptionalKeyword("unordered"))
    result.addAttribute("unordered", builder.getUnitAttr());

  // Parse the body region.
  M::Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable, indexType))
    return M::failure();

  fir::LoopOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return M::failure();
  return M::success();
}

fir::LoopOp getForInductionVarOwner(M::Value *val) {
  auto *ivArg = dyn_cast<M::BlockArgument>(val);
  if (!ivArg)
    return fir::LoopOp();
  assert(ivArg->getOwner() && "unlinked block argument");
  auto *containingInst = ivArg->getOwner()->getParentOp();
  return dyn_cast_or_null<fir::LoopOp>(containingInst);
}

// StoreOp

M::Type StoreOp::elementType(M::Type refType) {
  if (auto ref = refType.dyn_cast<ReferenceType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<PointerType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<HeapType>())
    return ref.getEleTy();
  return {};
}

// WhereOp

void WhereOp::build(M::Builder *builder, M::OperationState &result,
                    M::Value *cond, bool withElseRegion) {
  result.addOperands(cond);
  M::Region *thenRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();
  WhereOp::ensureTerminator(*thenRegion, *builder, result.location);
  if (withElseRegion)
    WhereOp::ensureTerminator(*elseRegion, *builder, result.location);
}

M::ParseResult parseWhereOp(M::OpAsmParser &parser, M::OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  M::Region *thenRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  M::OpAsmParser::OperandType cond;
  M::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return M::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return M::failure();

  WhereOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (!parser.parseOptionalKeyword("otherwise")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return M::failure();
    WhereOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                              result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return M::failure();

  return M::success();
}

M::ParseResult isValidCaseAttr(M::Attribute attr) {
  if (attr.dyn_cast_or_null<M::UnitAttr>() ||
      attr.dyn_cast_or_null<ClosedIntervalAttr>() ||
      attr.dyn_cast_or_null<PointIntervalAttr>() ||
      attr.dyn_cast_or_null<LowerBoundAttr>() ||
      attr.dyn_cast_or_null<UpperBoundAttr>())
    return M::success();
  return M::failure();
}

unsigned getCaseArgumentOffset(L::ArrayRef<M::Attribute> cases, unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<M::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<ClosedIntervalAttr>())
        ++o;
    }
  }
  return o;
}

M::ParseResult parseSelector(M::OpAsmParser &parser, M::OperationState &result,
                             M::OpAsmParser::OperandType &selector,
                             M::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return M::failure();
  return M::success();
}

/// Generic pretty-printer of a binary operation
void printBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op must have two operands");
  assert(op->getNumResults() == 1 && "binary op must have one result");

  p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0)->getType();
}

/// Generic pretty-printer of an unary operation
void printUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op must have one operand");
  assert(op->getNumResults() == 1 && "unary op must have one result");

  p << op->getName() << ' ' << *op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0)->getType();
}

// Tablegen operators

#define GET_OP_CLASSES
#include "fir/FIROps.cpp.inc"

} // namespace fir
