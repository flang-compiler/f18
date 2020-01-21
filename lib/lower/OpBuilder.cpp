//===-- lib/lower/builder.cc ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/OpBuilder.h"
#include "fir/Dialect/FIROpsSupport.h"
#include "fir/Dialect/FIRType.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/ConvertType.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace B = Fortran::lower;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::lower;

M::FuncOp B::createFunction(B::AbstractConverter &converter,
                            llvm::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(converter.getCurrentLocation(),
                           converter.getModuleOp(), name, funcTy);
}

M::FuncOp B::createFunction(M::ModuleOp module, llvm::StringRef name,
                            M::FunctionType funcTy) {
  return fir::createFuncOp(M::UnknownLoc::get(module.getContext()), module,
                           name, funcTy);
}

M::FuncOp B::getNamedFunction(M::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<M::FuncOp>(name);
}

void B::SymMap::addSymbol(Se::SymbolRef symbol, M::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

M::Value B::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void B::SymMap::pushShadowSymbol(Se::SymbolRef symbol, M::Value value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const Se::Symbol *sym{nullptr};
  M::Value val;
  // if mapping exists, save it on the shadow stack
  if (iter != symbolMap.end()) {
    sym = iter->first;
    val = iter->second;
    symbolMap.erase(iter);
  }
  shadowStack.emplace_back(sym, val);
  // insert new shadow mapping
  auto r{symbolMap.try_emplace(&*symbol, value)};
  assert(r.second && "unexpected insertion failure");
  (void)r;
}

M::Value B::OpBuilderWrapper::createIntegerConstant(M::Type integerType,
                                                    std::int64_t cst) {
  return create<M::ConstantOp>(integerType,
                               builder.getIntegerAttr(integerType, cst));
}

// LoopBuilder implementation

void B::LoopBuilder::createLoop(M::Value lb, M::Value ub, M::Value step,
                                const BodyGenerator &bodyGenerator) {
  auto lbi{convertToIndexType(lb)};
  auto ubi{convertToIndexType(ub)};
  L::SmallVector<M::Value, 1> steps;
  if (step) {
    auto stepi{convertToIndexType(step)};
    steps.emplace_back(stepi);
  }
  auto loop{create<fir::LoopOp>(lbi, ubi, steps)};
  auto *insPt{builder.getInsertionBlock()};
  builder.setInsertionPointToStart(loop.getBody());
  auto index{loop.getInductionVar()};
  bodyGenerator(*this, index);
  builder.setInsertionPointToEnd(insPt);
}

void B::LoopBuilder::createLoop(M::Value lb, M::Value ub,
                                const BodyGenerator &bodyGenerator) {
  auto one{createIntegerConstant(getIndexType(), 1)};
  createLoop(lb, ub, one, bodyGenerator);
}

void B::LoopBuilder::createLoop(M::Value count,
                                const BodyGenerator &bodyGenerator) {
  auto indexType{getIndexType()};
  auto zero{createIntegerConstant(indexType, 0)};
  auto one{createIntegerConstant(indexType, 1)};
  createLoop(zero, count, one, bodyGenerator);
}

M::Type B::LoopBuilder::getIndexType() {
  return M::IndexType::get(builder.getContext());
}

M::Value B::LoopBuilder::convertToIndexType(M::Value integer) {
  auto type{integer.getType()};
  if (type.isa<M::IndexType>()) {
    return integer;
  }
  assert((type.isa<M::IntegerType>() || type.isa<fir::IntType>()) &&
         "expected integer");
  return create<fir::ConvertOp>(getIndexType(), integer);
}

// CharacterOpsBuilder implementation

void B::CharacterOpsBuilder::createCopy(CharValue &dest, CharValue &src,
                                        M::Value count) {
  auto refType{dest.getReferenceType()};
  // Cast to character sequence reference type for fir::CoordinateOp.
  auto sequenceType{getSequenceRefType(refType)};
  auto destRef{create<fir::ConvertOp>(sequenceType, dest.reference)};
  auto srcRef{create<fir::ConvertOp>(sequenceType, src.reference)};

  LoopBuilder{*this}.createLoop(count, [&](OpBuilderWrapper &handler,
                                           M::Value index) {
    auto destAddr{handler.create<fir::CoordinateOp>(refType, destRef, index)};
    auto srcAddr{handler.create<fir::CoordinateOp>(refType, srcRef, index)};
    auto val{handler.create<fir::LoadOp>(srcAddr)};
    handler.create<fir::StoreOp>(val, destAddr);
  });
}

void B::CharacterOpsBuilder::createPadding(CharValue &str, M::Value lower,
                                           M::Value upper) {
  auto refType{str.getReferenceType()};
  auto sequenceType{getSequenceRefType(refType)};
  auto strRef{create<fir::ConvertOp>(sequenceType, str.reference)};
  auto blank{createBlankConstant(str.getCharacterType())};

  LoopBuilder{*this}.createLoop(
      lower, upper, [&](OpBuilderWrapper &handler, M::Value index) {
        auto strAddr{handler.create<fir::CoordinateOp>(refType, strRef, index)};
        handler.create<fir::StoreOp>(blank, strAddr);
      });
}

M::Value B::CharacterOpsBuilder::createBlankConstant(fir::CharacterType type) {
  auto byteTy{M::IntegerType::get(8, builder.getContext())};
  auto asciiSpace{createIntegerConstant(byteTy, 0x20)};
  return create<fir::ConvertOp>(type, asciiSpace);
}

B::CharacterOpsBuilder::CharValue
B::CharacterOpsBuilder::createTemp(fir::CharacterType type, M::Value len) {
  // FIXME Does this need to be emitted somewhere safe ?
  // convert-expr.cc generates alloca at the beginning of the mlir block.
  return CharValue{create<fir::AllocaOp>(type, len), len};
}

fir::ReferenceType B::CharacterOpsBuilder::CharValue::getReferenceType() {
  auto type{reference.getType().dyn_cast<fir::ReferenceType>()};
  assert(type && "expected reference type");
  return type;
}

fir::CharacterType B::CharacterOpsBuilder::CharValue::getCharacterType() {
  auto type{getReferenceType().getEleTy().dyn_cast<fir::CharacterType>()};
  assert(type && "expected character type");
  return type;
}

// ComplexOpsBuilder implementation

mlir::Type B::ComplexOpsBuilder::getComplexPartType(fir::KindTy complexKind) {
  return convertReal(builder.getContext(), complexKind);
}
mlir::Type B::ComplexOpsBuilder::getComplexPartType(mlir::Type complexType) {
  return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
}
mlir::Type B::ComplexOpsBuilder::getComplexPartType(mlir::Value cplx) {
  assert(cplx != nullptr);
  return getComplexPartType(cplx.getType());
}

mlir::Value B::ComplexOpsBuilder::createComplex(fir::KindTy kind,
                                                mlir::Value real,
                                                mlir::Value imag) {
  mlir::Type complexTy{fir::CplxType::get(builder.getContext(), kind)};
  mlir::Value und{create<fir::UndefOp>(complexTy)};
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

using CplxPart = B::ComplexOpsBuilder::Part;
template <CplxPart partId>
mlir::Value B::ComplexOpsBuilder::createPartId() {
  auto type{mlir::IntegerType::get(32, builder.getContext())};
  return createIntegerConstant(type, static_cast<int>(partId));
}

template <CplxPart partId>
mlir::Value B::ComplexOpsBuilder::extract(mlir::Value cplx) {
  return create<fir::ExtractValueOp>(getComplexPartType(cplx), cplx,
                                     createPartId<partId>());
}
template mlir::Value B::ComplexOpsBuilder::extract<CplxPart::Real>(mlir::Value);
template mlir::Value B::ComplexOpsBuilder::extract<CplxPart::Imag>(mlir::Value);

template <CplxPart partId>
mlir::Value B::ComplexOpsBuilder::insert(mlir::Value cplx, mlir::Value part) {
  assert(cplx != nullptr);
  return create<fir::InsertValueOp>(cplx.getType(), cplx, part,
                                    createPartId<partId>());
}
template mlir::Value B::ComplexOpsBuilder::insert<CplxPart::Real>(mlir::Value,
                                                                  mlir::Value);
template mlir::Value B::ComplexOpsBuilder::insert<CplxPart::Imag>(mlir::Value,
                                                                  mlir::Value);

mlir::Value B::ComplexOpsBuilder::extractComplexPart(mlir::Value cplx,
                                                     bool isImagPart) {
  return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
}

mlir::Value B::ComplexOpsBuilder::insertComplexPart(mlir::Value cplx,
                                                    mlir::Value part,
                                                    bool isImagPart) {
  return isImagPart ? insert<Part::Imag>(cplx, part)
                    : insert<Part::Real>(cplx, part);
}

mlir::Value B::ComplexOpsBuilder::createComplexCompare(mlir::Value cplx1,
                                                       mlir::Value cplx2,
                                                       bool eq) {
  mlir::Value real1{extract<Part::Real>(cplx1)};
  mlir::Value real2{extract<Part::Real>(cplx2)};
  mlir::Value imag1{extract<Part::Imag>(cplx1)};
  mlir::Value imag2{extract<Part::Imag>(cplx2)};

  mlir::CmpFPredicate predicate{eq ? mlir::CmpFPredicate::UEQ
                                   : mlir::CmpFPredicate::UNE};
  mlir::Value realCmp{create<mlir::CmpFOp>(predicate, real1, real2)};
  mlir::Value imagCmp{create<mlir::CmpFOp>(predicate, imag1, imag2)};

  return eq ? create<mlir::AndOp>(realCmp, imagCmp).getResult()
            : create<mlir::OrOp>(realCmp, imagCmp).getResult();
}
