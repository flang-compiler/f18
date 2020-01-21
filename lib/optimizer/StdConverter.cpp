//===-- lib/optimizer/StdConverter.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/optimizer/Transforms/StdConverter.h"
#include "fir/Dialect/FIRAttr.h"
#include "fir/Dialect/FIRDialect.h"
#include "fir/Dialect/FIROpsSupport.h"
#include "fir/Dialect/FIRType.h"
#include "flang/optimizer/KindMapping.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

// This module performs the conversion of some FIR operations.
// Convert some FIR types to standard dialect?

namespace L = llvm;
namespace M = mlir;

using namespace fir;

static L::cl::opt<bool>
    ClDisableFirToStd("disable-fir2std",
                      L::cl::desc("disable FIR to standard pass"),
                      L::cl::init(false), L::cl::Hidden);

namespace {

using SmallVecResult = L::SmallVector<M::Value, 4>;
using OperandTy = L::ArrayRef<M::Value>;
using AttributeTy = L::ArrayRef<M::NamedAttribute>;

/// FIR to standard type converter
/// This converts a subset of FIR types to standard types
class FIRToStdTypeConverter : public M::TypeConverter {
public:
  using TypeConverter::TypeConverter;

  explicit FIRToStdTypeConverter(KindMapping &kindMap) : kindMap{kindMap} {}

  // convert front-end REAL kind value to a std dialect type, if possible
  static M::Type kindToRealType(KindMapping &kindMap, KindTy kind) {
    auto *ctx = kindMap.getContext();
    switch (kindMap.getRealTypeID(kind)) {
    case L::Type::TypeID::HalfTyID:
      return M::FloatType::getF16(ctx);
#if 0
    case L::Type::TypeID:: FIXME TyID:
      return M::FloatType::getBF16(ctx);
#endif
    case L::Type::TypeID::FloatTyID:
      return M::FloatType::getF32(ctx);
    case L::Type::TypeID::DoubleTyID:
      return M::FloatType::getF64(ctx);
    case L::Type::TypeID::X86_FP80TyID: // MLIR does not support yet
    case L::Type::TypeID::FP128TyID:    // MLIR does not support yet
    default:
      return fir::RealType::get(ctx, kind);
    }
  }

  /// Convert some FIR types to MLIR standard dialect types
  M::Type convertType(M::Type t) override {
#if 0
    // To lower types, we have to convert everything that uses these types...
    if (auto cplx = t.dyn_cast<CplxType>())
      return M::ComplexType::get(kindToRealType(kindMap, cplx.getFKind()));
    if (auto integer = t.dyn_cast<IntType>())
      return M::IntegerType::get(integer.getFKind() * 8, integer.getContext());
    if (auto real = t.dyn_cast<RealType>())
      return kindToRealType(kindMap, real.getFKind());
#endif
    return t;
  }

private:
  KindMapping &kindMap;
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public M::ConversionPattern {
public:
  explicit FIROpConversion(M::MLIRContext *ctx, FIRToStdTypeConverter &lowering)
      : ConversionPattern(FromOp::getOperationName(), 1, ctx),
        lowering(lowering) {}

protected:
  M::Type convertType(M::Type ty) const { return lowering.convertType(ty); }

  FIRToStdTypeConverter &lowering;
};

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selectType = M::cast<SelectTypeOp>(op);
    auto conds = selectType.getNumConditions();
    auto attrName = SelectTypeOp::AttrName;
    auto caseAttr = selectType.getAttrOfType<M::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Selector must be of type !fir.box<T>
    auto &selector = operands[0];
    auto loc = selectType.getLoc();
    auto mod = op->getParentOfType<M::ModuleOp>();
    for (unsigned t = 0; t != conds; ++t) {
      auto &attr = cases[t];
      if (auto a = attr.dyn_cast_or_null<fir::ExactTypeAttr>()) {
        genTypeLadderStep(loc, true, selector, a.getType(), destinations[t],
                          destOperands[t], mod, rewriter);
        continue;
      }
      if (auto a = attr.dyn_cast_or_null<fir::SubclassAttr>()) {
        genTypeLadderStep(loc, false, selector, a.getType(), destinations[t],
                          destOperands[t], mod, rewriter);
        continue;
      }
      assert(attr.dyn_cast_or_null<M::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<M::BranchOp>(selectType, destinations[t],
                                               M::ValueRange{destOperands[t]});
    }
    return matchSuccess();
  }

  static void genTypeLadderStep(M::Location loc, bool exactTest,
                                M::Value selector, M::Type ty, M::Block *dest,
                                OperandTy destOps, M::ModuleOp module,
                                M::ConversionPatternRewriter &rewriter) {
    M::Type tydesc = fir::TypeDescType::get(ty);
    auto tyattr = M::TypeAttr::get(ty);
    M::Value t = rewriter.create<GenTypeDescOp>(loc, tydesc, tyattr);
    M::Type selty = fir::BoxType::get(rewriter.getNoneType());
    M::Value csel = rewriter.create<ConvertOp>(loc, selty, selector);
    M::Type tty = fir::ReferenceType::get(rewriter.getNoneType());
    M::Value ct = rewriter.create<ConvertOp>(loc, tty, t);
    std::vector<M::Value> actuals = {csel, ct};
    auto fty = rewriter.getI1Type();
    std::vector<M::Type> argTy = {selty, tty};
    L::StringRef funName =
        exactTest ? "FIXME_exact_type_match" : "FIXME_isa_type_test";
    createFuncOp(rewriter.getUnknownLoc(), module, funName,
                 rewriter.getFunctionType(argTy, fty));
    // FIXME: need to call actual runtime routines for (1) testing if the
    // runtime type of the selector is an exact match to a derived type or (2)
    // testing if the runtime type of the selector is a derived type or one of
    // that derived type's subtypes.
    auto cmp = rewriter.create<M::CallOp>(
        loc, fty, rewriter.getSymbolRefAttr(funName), actuals);
    auto *thisBlock = rewriter.getInsertionBlock();
    auto *newBlock = rewriter.createBlock(dest);
    rewriter.setInsertionPointToEnd(thisBlock);
    rewriter.create<M::CondBranchOp>(loc, cmp.getResult(0), dest, destOps,
                                     newBlock, OperandTy{});
    rewriter.setInsertionPointToEnd(newBlock);
  }
};

/// Convert affine dialect, fir.select_type to standard dialect
class FIRToStdLoweringPass : public M::FunctionPass<FIRToStdLoweringPass> {
public:
  explicit FIRToStdLoweringPass(KindMapping &kindMap) : kindMap{kindMap} {}

  void runOnFunction() override {
    if (ClDisableFirToStd)
      return;

    auto *context{&getContext()};
    FIRToStdTypeConverter typeConverter{kindMap};
    M::OwningRewritePatternList patterns;
    patterns.insert<SelectTypeOpConversion>(context, typeConverter);
    M::populateAffineToStdConversionPatterns(patterns, context);
    M::populateFuncOpTypeConversionPattern(patterns, context, typeConverter);
    M::ConversionTarget target{*context};
    target.addLegalDialect<M::StandardOpsDialect, fir::FIROpsDialect>();
    target.addDynamicallyLegalOp<M::FuncOp>([&](M::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    target.addIllegalOp<SelectTypeOp>();
    if (M::failed(M::applyPartialConversion(
            getModule(), target, std::move(patterns), &typeConverter))) {
      M::emitError(M::UnknownLoc::get(context),
                   "error in converting to standard dialect\n");
      signalPassFailure();
    }
  }

  M::ModuleOp getModule() {
    return getFunction().getParentOfType<M::ModuleOp>();
  }

private:
  KindMapping &kindMap;
};

} // namespace

std::unique_ptr<M::Pass> fir::createFIRToStdPass(fir::KindMapping &kindMap) {
  return std::make_unique<FIRToStdLoweringPass>(kindMap);
}
