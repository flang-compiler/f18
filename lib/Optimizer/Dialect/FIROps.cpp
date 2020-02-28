//===-- FIROps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSwitch.h"

namespace fir {

/// return true if the sequence type is abstract or the record type is malformed
/// or contains an abstract sequence type
static bool verifyInType(mlir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited) {
  if (auto st = inType.dyn_cast<fir::SequenceType>()) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (auto ext : shape)
      if (ext < 0)
        return true;
  } else if (auto rt = inType.dyn_cast<fir::RecordType>()) {
    // don't recurse if we're already visiting this one
    for (auto name : visited)
      if (name == rt.getName())
        return false;
    // keep track of record types currently being visited
    visited.push_back(rt.getName());
    for (auto &field : rt.getTypeList())
      if (verifyInType(field.second, visited))
        return true;
    visited.pop_back();
  } else if (auto rt = inType.dyn_cast<fir::PointerType>()) {
    return verifyInType(rt.getEleTy(), visited);
  }
  return false;
}

static bool verifyRecordLenParams(mlir::Type inType, unsigned numLenParams) {
  if (numLenParams > 0) {
    if (auto rt = inType.dyn_cast<fir::RecordType>())
      return numLenParams != rt.getNumLenParams();
    return true;
  }
  return false;
}

// AllocaOp

mlir::Type AllocaOp::getAllocatedType() {
  return getType().cast<ReferenceType>().getEleTy();
}

/// Create a legal memory reference as return type
mlir::Type AllocaOp::wrapResultType(mlir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (intype.dyn_cast<ReferenceType>())
    return {};
  return ReferenceType::get(intype);
}

mlir::Type AllocaOp::getRefTy(mlir::Type ty) { return ReferenceType::get(ty); }

// AllocMemOp

mlir::Type AllocMemOp::getAllocatedType() {
  return getType().cast<HeapType>().getEleTy();
}

mlir::Type AllocMemOp::getRefTy(mlir::Type ty) { return HeapType::get(ty); }

/// Create a legal heap reference as return type
mlir::Type AllocMemOp::wrapResultType(mlir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (intype.dyn_cast<ReferenceType>() || intype.dyn_cast<HeapType>() ||
      intype.dyn_cast<PointerType>() || intype.dyn_cast<FunctionType>())
    return {};
  return HeapType::get(intype);
}

// BoxDimsOp

/// Get the result types packed in a tuple tuple
mlir::Type BoxDimsOp::getTupleType() {
  llvm::SmallVector<mlir::Type, 3> triple{
      getResult(0).getType(), getResult(1).getType(), getResult(2).getType()};
  return mlir::TupleType::get(triple, getContext());
}

// CallOp

void printCallOp(mlir::OpAsmPrinter &p, fir::CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();
  p << op.getOperationName() << ' ';
  if (isDirect)
    p << callee.getValue();
  else
    p << op.getOperand(0);
  p << '(';
  p.printOperands(llvm::drop_begin(op.getOperands(), isDirect ? 0 : 1));
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});
  auto resultTypes{op.getResultTypes()};
  llvm::SmallVector<Type, 8> argTypes(
      llvm::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << FunctionType::get(argTypes, resultTypes, op.getContext());
}

mlir::ParseResult parseCallOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;

  if (parser.parseOperandList(operands))
    return mlir::failure();
  bool isDirect = operands.empty();
  SmallVector<NamedAttribute, 4> attrs;
  SymbolRefAttr funcAttr;

  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", attrs))
      return mlir::failure();
  Type type;

  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return mlir::failure();

  auto funcType = type.dyn_cast<mlir::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<mlir::OpAsmParser::OperandType>(operands).drop_front();
    llvm::SmallVector<mlir::Value, 8> resultArgs(
        result.operands.begin() + (result.operands.empty() ? 0 : 1),
        result.operands.end());
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), resultArgs))
      return mlir::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return mlir::success();
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
                 CmpFPredicate predicate, Value lhs, Value rhs) {
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
      op.template getAttrOfType<mlir::IntegerAttr>(OPTY::getPredicateAttrName())
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
  p << " : " << op.lhs().getType();
}

void printCmpfOp(OpAsmPrinter &p, CmpfOp op) { printCmpOp(p, op); }

template <typename OPTY>
mlir::ParseResult parseCmpOp(mlir::OpAsmParser &parser,
                             mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> ops;
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
  mlir::Attribute predicateNameAttr;
  mlir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<mlir::StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      predicateNameAttr.cast<mlir::StringAttr>().getValue();
  auto predicate = CmpfOp::getPredicateByName(predicateName);
  if (predicate == CmpFPredicate::NumPredicates)
    return parser.emitError(parser.getNameLoc(),
                            "unknown comparison predicate \"" + predicateName +
                                "\"");

  auto builder = parser.getBuilder();
  mlir::Type i1Type = builder.getI1Type();
  attrs[0].second = builder.getI64IntegerAttr(static_cast<int64_t>(predicate));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return success();
}

mlir::ParseResult parseCmpfOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  return parseCmpOp<fir::CmpfOp>(parser, result);
}

// CmpcOp

void buildCmpCOp(Builder *builder, OperationState &result,
                 CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder->getI1Type());
  result.addAttribute(
      CmpcOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

void printCmpcOp(OpAsmPrinter &p, CmpcOp op) { printCmpOp(p, op); }

mlir::ParseResult parseCmpcOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

// DispatchOp

mlir::FunctionType DispatchOp::getFunctionType() {
  auto attr = getAttr("fn_type").cast<mlir::TypeAttr>();
  return attr.getValue().cast<mlir::FunctionType>();
}

// DispatchTableOp

void DispatchTableOp::appendTableEntry(mlir::Operation *op) {
  assert(mlir::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  auto &block = getBlock();
  block.getOperations().insert(block.end(), op);
}

// GenTypeDescOp

void GenTypeDescOp::build(Builder *, OperationState &result,
                          mlir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

// GlobalOp

void GlobalOp::appendInitialValue(mlir::Operation *op) {
  auto &block = getBlock();
  block.getOperations().insert(block.end(), op);
}

// LoadOp

/// Get the element type of a reference like type; otherwise null
mlir::Type elementTypeOf(mlir::Type ref) {
  if (auto r = ref.dyn_cast_or_null<ReferenceType>())
    return r.getEleTy();
  if (auto r = ref.dyn_cast_or_null<PointerType>())
    return r.getEleTy();
  if (auto r = ref.dyn_cast_or_null<HeapType>())
    return r.getEleTy();
  return {};
}

mlir::ParseResult LoadOp::getElementOf(mlir::Type &ele, mlir::Type ref) {
  if (mlir::Type r = elementTypeOf(ref)) {
    ele = r;
    return mlir::success();
  }
  return mlir::failure();
}

// LoopOp

void LoopOp::build(mlir::Builder *builder, OperationState &result,
                   mlir::Value lb, mlir::Value ub, ValueRange step) {
  if (step.empty())
    result.addOperands({lb, ub});
  else
    result.addOperands({lb, ub, step[0]});
  mlir::Region *bodyRegion = result.addRegion();
  LoopOp::ensureTerminator(*bodyRegion, *builder, result.location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

mlir::ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  mlir::Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands))
    return mlir::failure();

  if (parser.parseOptionalKeyword(LoopOp::stepKeyword())) {
    result.addAttribute(LoopOp::stepKeyword(),
                        builder.getIntegerAttr(builder.getIndexType(), 1));
  } else if (parser.parseOperand(step) ||
             parser.resolveOperand(step, indexType, result.operands)) {
    return mlir::failure();
  }

  // Parse the optional `unordered` keyword
  bool isUnordered = false;
  if (!parser.parseOptionalKeyword(LoopOp::unorderedKeyword())) {
    result.addAttribute(LoopOp::unorderedKeyword(), builder.getUnitAttr());
    isUnordered = true;
  }

  // Parse the body region.
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable, indexType))
    return mlir::failure();

  fir::LoopOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  if (!isUnordered)
    result.addTypes(builder.getIndexType());
  return mlir::success();
}

fir::LoopOp getForInductionVarOwner(mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg)
    return fir::LoopOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<fir::LoopOp>(containingInst);
}

// StoreOp

mlir::Type StoreOp::elementType(mlir::Type refType) {
  if (auto ref = refType.dyn_cast<ReferenceType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<PointerType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<HeapType>())
    return ref.getEleTy();
  return {};
}

// StringLitOp

bool StringLitOp::isWideValue() {
  auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
  return eleTy.cast<fir::CharacterType>().getFKind() != 1;
}

// WhereOp

void WhereOp::build(mlir::Builder *builder, OperationState &result,
                    mlir::Value cond, bool withElseRegion) {
  result.addOperands(cond);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();
  WhereOp::ensureTerminator(*thenRegion, *builder, result.location);
  if (withElseRegion)
    WhereOp::ensureTerminator(*elseRegion, *builder, result.location);
}

mlir::ParseResult parseWhereOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return mlir::failure();

  WhereOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (!parser.parseOptionalKeyword("otherwise")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return mlir::failure();
    WhereOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                              result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  return mlir::success();
}

mlir::ParseResult isValidCaseAttr(mlir::Attribute attr) {
  if (attr.dyn_cast_or_null<mlir::UnitAttr>() ||
      attr.dyn_cast_or_null<ClosedIntervalAttr>() ||
      attr.dyn_cast_or_null<PointIntervalAttr>() ||
      attr.dyn_cast_or_null<LowerBoundAttr>() ||
      attr.dyn_cast_or_null<UpperBoundAttr>())
    return mlir::success();
  return mlir::failure();
}

unsigned getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                               unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<ClosedIntervalAttr>())
        ++o;
    }
  }
  return o;
}

mlir::ParseResult parseSelector(mlir::OpAsmParser &parser,
                                mlir::OperationState &result,
                                mlir::OpAsmParser::OperandType &selector,
                                mlir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return mlir::failure();
  return mlir::success();
}

/// Generic pretty-printer of a binary operation
void printBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op must have two operands");
  assert(op->getNumResults() == 1 && "binary op must have one result");

  p << op->getName() << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

/// Generic pretty-printer of an unary operation
void printUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op must have one operand");
  assert(op->getNumResults() == 1 && "unary op must have one result");

  p << op->getName() << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

bool isReferenceLike(mlir::Type type) {
  return type.isa<fir::ReferenceType>() || type.isa<fir::HeapType>() ||
         type.isa<fir::PointerType>();
}

mlir::FuncOp createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                          StringRef name, mlir::FunctionType type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto f = module.lookupSymbol<mlir::FuncOp>(name))
    return f;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  return modBuilder.create<mlir::FuncOp>(loc, name, type, attrs);
}

GlobalOp createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                        StringRef name, mlir::Type type,
                        llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto g = module.lookupSymbol<GlobalOp>(name))
    return g;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  return modBuilder.create<GlobalOp>(loc, name, type, attrs);
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"

} // namespace fir
