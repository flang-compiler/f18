//===-- include/fir/FIRDialect.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_FIRDIALECT_H
#define OPTIMIZER_FIRDIALECT_H

#include "mlir/IR/Dialect.h"

namespace llvm {
class raw_ostream;
class StringRef;
} // namespace llvm

namespace mlir {
class Attribute;
class DialectAsmParser;
class DialectAsmPrinter;
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &p) const override;
};

} // namespace fir

#endif // OPTIMIZER_FIRDIALECT_H
