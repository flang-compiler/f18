//===-- include/fir/Transforms/Passes.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_TRANSFORMS_PASSES_H
#define OPTIMIZER_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class FuncOp;
template <typename>
class OpPassBase;
class Pass;
} // namespace mlir

namespace fir {

/// Effects aware CSE pass
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createCSEPass();

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> createLowerToLoopPass();

/// A pass to convert the FIR dialect from "Mem-SSA" form to "Reg-SSA"
/// form. This pass is a port of LLVM's mem2reg pass, but modified for the FIR
/// dialect as well as the restructuring of MLIR's representation to present PHI
/// nodes as block arguments.
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createMemToRegPass();

} // namespace fir

#endif // OPTIMIZER_TRANSFORMS_PASSES_H
