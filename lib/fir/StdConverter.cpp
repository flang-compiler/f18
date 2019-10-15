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

#include "fir/Transforms/StdConverter.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LowerAffine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// This module performs the conversion of FIR operations to MLIR standard and/or
// LLVM-IR dialects.

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace {

using SmallVecResult = L::SmallVector<M::Value *, 4>;
using OperandTy = L::ArrayRef<M::Value *>;
using AttributeTy = L::ArrayRef<M::NamedAttribute>;

/// FIR to standard type converter
/// This converts a subset of FIR types to standard types
class FIRToStdTypeConverter : public M::TypeConverter {
public:
  using TypeConverter::TypeConverter;

  // convert a front-end kind value to either a std dialect type
  static M::Type kindToRealType(M::MLIRContext *ctx, KindTy kind) {
    switch (kind) {
    case 2:
      return M::FloatType::getF16(ctx);
    case 3:
      return M::FloatType::getBF16(ctx);
    case 4:
      return M::FloatType::getF32(ctx);
    case 8:
      return M::FloatType::getF64(ctx);
    }
    return fir::RealType::get(ctx, kind);
  }

  /// Convert FIR types to MLIR standard dialect types
  M::Type convertType(M::Type t) override {
    if (auto cplx = t.dyn_cast<CplxType>()) {
      return M::ComplexType::get(
          kindToRealType(cplx.getContext(), cplx.getFKind()));
    }
    if (auto integer = t.dyn_cast<IntType>()) {
      return M::IntegerType::get(integer.getFKind() * 8, integer.getContext());
    }
    if (auto real = t.dyn_cast<RealType>()) {
      return kindToRealType(real.getContext(), real.getFKind());
    }
    return t;
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
inline void rewriteSelectConstruct(M::Operation *op, OperandTy operands,
                                   L::ArrayRef<M::Block *> dests,
                                   L::ArrayRef<OperandTy> destOperands,
                                   M::OpBuilder &rewriter) {
  L::SmallVector<M::Value *, 1> noargs;
  L::SmallVector<M::Block *, 8> blocks;
  auto loc{op->getLoc()};
  blocks.push_back(rewriter.getInsertionBlock());
  for (std::size_t i = 1; i < dests.size(); ++i)
    blocks.push_back(rewriter.createBlock(dests[0]));
  rewriter.setInsertionPointToEnd(blocks[0]);
  if (dests.size() == 1) {
    rewriter.create<M::BranchOp>(loc, dests[0], destOperands[0]);
    return;
  }
  rewriter.create<M::CondBranchOp>(loc, operands[1], dests[0], destOperands[0],
                                   blocks[1], noargs);
  for (std::size_t i = 1; i < dests.size() - 1; ++i) {
    rewriter.setInsertionPointToEnd(blocks[i]);
    rewriter.create<M::CondBranchOp>(loc, operands[i + 1], dests[i],
                                     destOperands[i], blocks[i + 1], noargs);
  }
  std::size_t last{dests.size() - 1};
  rewriter.setInsertionPointToEnd(blocks[last]);
  rewriter.create<M::BranchOp>(loc, dests[last], destOperands[last]);
}

/// Convert FIR dialect to standard dialect
class FIRToStdLoweringPass : public M::ModulePass<FIRToStdLoweringPass> {
  M::OpBuilder *builder;

  void lowerSelect(M::Operation *op) {
    if (M::dyn_cast<SelectCaseOp>(op) || M::dyn_cast<SelectRankOp>(op) ||
        M::dyn_cast<SelectTypeOp>(op)) {
      // build the lists of operands and successors
      L::SmallVector<M::Value *, 4> operands{op->operand_begin(),
                                             op->operand_end()};
      L::SmallVector<M::Block *, 2> destinations;
      destinations.reserve(op->getNumSuccessors());
      L::SmallVector<OperandTy, 2> destOperands;
      unsigned firstSuccOpd = op->getSuccessorOperandIndex(0);
      for (unsigned i = 0, seen = 0, e = op->getNumSuccessors(); i < e; ++i) {
        destinations.push_back(op->getSuccessor(i));
        unsigned n = op->getNumSuccessorOperands(i);
        destOperands.push_back(
            L::makeArrayRef(operands.data() + firstSuccOpd + seen, n));
        seen += n;
      }
      // do the rewrite
      rewriteSelectConstruct(
          op, L::makeArrayRef(operands.data(), operands.data() + firstSuccOpd),
          destinations, destOperands, *builder);
    }
  }

public:
  void runOnModule() override {
    return;

    for (auto fn : getModule().getOps<M::FuncOp>()) {
      M::OpBuilder rewriter{&fn.getBody()};
      builder = &rewriter;
      fn.walk([&](M::Operation *op) { lowerSelect(op); });
    }
    auto &context{getContext()};
    FIRToStdTypeConverter typeConverter;
    M::OwningRewritePatternList patterns;
    // patterns.insert<>(&context, typeConverter);
    M::populateAffineToStdConversionPatterns(patterns, &context);
    M::populateFuncOpTypeConversionPattern(patterns, &context, typeConverter);
    M::ConversionTarget target{context};
    target.addLegalDialect<M::StandardOpsDialect, fir::FIROpsDialect>();
    target.addDynamicallyLegalOp<M::FuncOp>([&](M::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    target.addDynamicallyLegalOp<M::ModuleOp>(
        [&](M::ModuleOp op) { return true; });
    if (M::failed(M::applyPartialConversion(
            getModule(), target, std::move(patterns), &typeConverter))) {
      M::emitError(M::UnknownLoc::get(&context),
                   "error in converting to standard dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<M::Pass> fir::createFIRToStdPass() {
  return std::make_unique<FIRToStdLoweringPass>();
}
