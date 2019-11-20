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

#include "fir/FIRDialect.h"
#include "fir/FIROps.h"
#include "fir/Transforms/Passes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include <memory>

namespace M = mlir;

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    ClDisableAffinePromo("disable-affine-promotion",
                         llvm::cl::desc("disable FIR to Affine pass"),
                         llvm::cl::init(false));

/// disable FIR to loop dialect conversion
static llvm::cl::opt<bool>
    ClDisableLoopConversion("disable-loop-conversion",
                            llvm::cl::desc("disable FIR to Loop pass"),
                            llvm::cl::init(false));

using namespace fir;

namespace {

template <typename FROM>
class OpRewrite : public M::RewritePattern {
public:
  explicit OpRewrite(M::MLIRContext *ctx)
      : RewritePattern(FROM::getOperationName(), 1, ctx) {}
};

/// Convert `fir.loop` to `affine.for`
class AffineLoopConv : public OpRewrite<LoopOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Convert `fir.where` to `affine.if`
class AffineWhereConv : public OpRewrite<WhereOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion : public M::FunctionPass<AffineDialectPromotion> {
public:
  void runOnFunction() override {
    if (ClDisableAffinePromo)
      return;

    auto *context{&getContext()};
    M::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConv, AffineWhereConv>(context);
    M::ConversionTarget target{*context};
    target.addLegalDialect<M::AffineOpsDialect, FIROpsDialect,
                           M::loop::LoopOpsDialect, M::StandardOpsDialect>();
    // target.addDynamicallyLegalOp<LoopOp, WhereOp>();

    // apply the patterns
    if (M::failed(M::applyPartialConversion(getFunction(), target,
                                            std::move(patterns)))) {
      M::emitError(M::UnknownLoc::get(context),
                   "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};

// Conversion to the MLIR loop dialect
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.loop` operations.  These can be converted to `loop.for` operations. MLIR
// includes a pass to lower `loop.for` operations to a CFG.

/// Convert `fir.loop` to `loop.for`
class LoopLoopConv : public M::OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  M::PatternMatchResult
  matchAndRewrite(LoopOp loop, M::PatternRewriter &rewriter) const override {
    auto *low = loop.lowerBound();
    auto *high = loop.upperBound();
    auto optStep = loop.optstep();
    auto loc = loop.getLoc();
    M::Value *step;
    if (optStep.begin() != optStep.end()) {
      step = *optStep.begin();
    } else {
      auto conStep = loop.constep();
      step = rewriter.create<M::ConstantIndexOp>(
          loc, conStep.hasValue() ? conStep.getValue().getSExtValue() : 1);
    }
    auto f = rewriter.create<M::loop::ForOp>(loc, low, high, step);
    f.region().getBlocks().clear();
    rewriter.inlineRegionBefore(loop.region(), f.region(), f.region().end());
    rewriter.eraseOp(loop);
    return matchSuccess();
  }
};

/// Convert `fir.where` to `loop.if`
class LoopWhereConv : public M::OpRewritePattern<WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  M::PatternMatchResult
  matchAndRewrite(WhereOp where, M::PatternRewriter &rewriter) const override {
    auto loc = where.getLoc();
    bool hasOtherRegion = !where.otherRegion().empty();
    auto cond = where.condition();
    auto ifOp = rewriter.create<M::loop::IfOp>(loc, cond, hasOtherRegion);
    rewriter.inlineRegionBefore(where.whereRegion(), &ifOp.thenRegion().back());
    ifOp.thenRegion().back().erase();
    if (hasOtherRegion) {
      rewriter.inlineRegionBefore(where.otherRegion(),
                                  &ifOp.elseRegion().back());
      ifOp.elseRegion().back().erase();
    }
    rewriter.eraseOp(where);
    return matchSuccess();
  }
};

/// Replace FirEndOp with TerminatorOp
class LoopFirEndConv : public M::OpRewritePattern<FirEndOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  M::PatternMatchResult
  matchAndRewrite(FirEndOp op, M::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<M::loop::TerminatorOp>(op);
    return matchSuccess();
  }
};

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.
class LoopDialectConversion : public M::FunctionPass<LoopDialectConversion> {
public:
  void runOnFunction() override {
    if (ClDisableLoopConversion)
      return;

    auto *context{&getContext()};
    M::OwningRewritePatternList patterns;
    patterns.insert<LoopLoopConv, LoopWhereConv, LoopFirEndConv>(context);
    M::ConversionTarget target{*context};
    target.addLegalDialect<M::AffineOpsDialect, FIROpsDialect,
                           M::loop::LoopOpsDialect, M::StandardOpsDialect>();
    target.addIllegalOp<FirEndOp, LoopOp, WhereOp>();

    // apply the patterns
    if (M::failed(M::applyPartialConversion(getFunction(), target,
                                            std::move(patterns)))) {
      M::emitError(M::UnknownLoc::get(context),
                   "error in converting to MLIR loop dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<M::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<M::Pass> fir::createLowerToLoopPass() {
  return std::make_unique<LoopDialectConversion>();
}
