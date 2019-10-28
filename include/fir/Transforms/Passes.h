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

#ifndef FIR_TRANSFORMS_PASSES_H
#define FIR_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class FuncOp;
template <typename>
class OpPassBase;
} // namespace mlir

namespace fir {

/// Effects aware CSE pass
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createCSEPass();

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createFirToLoopPass();

/// A pass to convert the FIR dialect from "Mem-SSA" form to "Reg-SSA"
/// form. This pass is a port of LLVM's mem2reg pass, but modified for the FIR
/// dialect as well as the restructuring of MLIR's representation to present PHI
/// nodes as block arguments.
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createMemToRegPass();

} // namespace fir

#endif // FIR_TRANSFORMS_PASSES_H
