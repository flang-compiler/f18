//===-- lib/lower/bridge.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_BRIDGE_H_
#define FORTRAN_LOWER_BRIDGE_H_

#include "../../../lib/common/Fortran.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include <memory>

/// Implement the burnside bridge from Fortran to
/// [MLIR](https://github.com/tensorflow/mlir).
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace Fortran {
namespace common {
template <typename>
class Reference;
} // namespace common
namespace evaluate {
struct DataRef;
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate
namespace parser {
class CharBlock;
} // namespace parser
namespace semantics {
class Symbol;
}
} // namespace Fortran

namespace mlir {
class OpBuilder;
}

namespace Fortran::lower {

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

/// The abstract interface for converter implementations to lower Fortran
/// front-end fragments such as expressions, types, etc.
class AbstractConverter {
public:
  //
  // Expressions

  /// Generate the address of the location holding the expression
  virtual mlir::Value genExprAddr(const SomeExpr &,
                                  mlir::Location *loc = nullptr) = 0;
  /// Generate the computations of the expression to produce a value
  virtual mlir::Value genExprValue(const SomeExpr &,
                                   mlir::Location *loc = nullptr) = 0;

  //
  // Types

  /// Generate the type of a DataRef
  virtual mlir::Type genType(const evaluate::DataRef &) = 0;
  /// Generate the type of an Expr
  virtual mlir::Type genType(const SomeExpr &) = 0;
  /// Generate the type of a Symbol
  virtual mlir::Type genType(SymbolRef) = 0;
  /// Generate the type from a category
  virtual mlir::Type genType(common::TypeCategory tc) = 0;
  /// Generate the type from a category and kind
  virtual mlir::Type genType(common::TypeCategory tc, int kind) = 0;

  //
  // Locations

  /// Get the converter's current location
  virtual mlir::Location getCurrentLocation() = 0;
  /// Generate a dummy location
  virtual mlir::Location genLocation() = 0;
  /// Generate the location as converted from a CharBlock
  virtual mlir::Location genLocation(const parser::CharBlock &) = 0;

  //
  // FIR/MLIR

  /// Get the OpBuilder
  virtual mlir::OpBuilder &getOpBuilder() = 0;
  /// Get the ModuleOp
  virtual mlir::ModuleOp &getModuleOp() = 0;
  /// Unique a symbol
  virtual std::string mangleName(SymbolRef) = 0;

  virtual ~AbstractConverter() = default;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_BRIDGE_H_
