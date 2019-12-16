//===-- include/fir/Transforms/StdConverter.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_TRANSFORMS_STDCONVERTER_H
#define OPTIMIZER_TRANSFORMS_STDCONVERTER_H

#include <memory>

namespace mlir {
class Pass;
}

namespace fir {

class KindMapping;

/// Convert FIR to the standard dialect
std::unique_ptr<mlir::Pass> createFIRToStdPass(KindMapping &);

} // namespace fir

#endif // OPTIMIZER_TRANSFORMS_STDCONVERTER_H
