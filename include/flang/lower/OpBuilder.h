//===-- lib/lower/builder.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_BUILDER_H_
#define FORTRAN_LOWER_BUILDER_H_

#include "../../../lib/semantics/symbol.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace llvm {
class StringRef;
}

namespace fir {
class CharacterType;
class ReferenceType;
using KindTy = int;
} // namespace fir

namespace Fortran {
namespace parser {
class CookedSource;
}

namespace evaluate {
struct ProcedureDesignator;
}

namespace lower {

/// Miscellaneous helper routines for building MLIR
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

class AbstractConverter;

class SymMap {
  llvm::DenseMap<const semantics::Symbol *, mlir::Value> symbolMap;
  std::vector<std::pair<const semantics::Symbol *, mlir::Value>> shadowStack;

public:
  void addSymbol(semantics::SymbolRef symbol, mlir::Value value);

  mlir::Value lookupSymbol(semantics::SymbolRef symbol);

  void pushShadowSymbol(semantics::SymbolRef symbol, mlir::Value value);
  void popShadowSymbol() { shadowStack.pop_back(); }

  void clear() {
    symbolMap.clear();
    shadowStack.clear();
  }
};

/// Helper class that can be inherited from in order to
/// facilitate mlir::OpBuilder usage.
class OpBuilderWrapper {
public:
  OpBuilderWrapper(mlir::OpBuilder &b, mlir::Location l) : builder{b}, loc{l} {}
  /// Insert the location and forwards create call to the builder member.
  template <typename T, typename... Args>
  auto create(Args... args) {
    return builder.create<T>(loc, std::forward<Args>(args)...);
  }
  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Type integerType, std::int64_t i);

protected:
  mlir::OpBuilder &builder;
  mlir::Location loc;
};

/// Facilitate lowering to fir::loop
class LoopBuilder : public OpBuilderWrapper {
public:
  LoopBuilder(mlir::OpBuilder &b, mlir::Location l) : OpBuilderWrapper{b, l} {}
  LoopBuilder(OpBuilderWrapper &b) : OpBuilderWrapper{b} {}

  // In createLoop functions, lb, ub, and count arguments must have integer
  // types. They will be automatically converted the IndexType if needed.

  using BodyGenerator = std::function<void(OpBuilderWrapper &, mlir::Value)>;

  /// Build loop [\p lb, \p ub) with step \p step.
  /// If \p is an empty value, 1 is used for the step.
  void createLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [\p lb,  \p ub) with step 1.
  void createLoop(mlir::Value lb, mlir::Value ub,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [0, \p count) with step 1.
  void createLoop(mlir::Value count, const BodyGenerator &bodyGenerator);

private:
  mlir::Type getIndexType();
  mlir::Value convertToIndexType(mlir::Value integer);
};

/// Facilitate lowering of CHARACTER operation
class CharacterOpsBuilder : public OpBuilderWrapper {
public:
  CharacterOpsBuilder(mlir::OpBuilder &b, mlir::Location l)
      : OpBuilderWrapper{b, l} {}
  CharacterOpsBuilder(OpBuilderWrapper &b) : OpBuilderWrapper{b} {}
  /// Interchange format to avoid inserting unbox/embox everywhere while
  /// evaluating character expressions.
  struct CharValue {
    fir::ReferenceType getReferenceType();
    fir::CharacterType getCharacterType();

    mlir::Value reference;
    mlir::Value len;
  };

  /// Copy the \p count first characters of \p src into \p dest.
  void createCopy(CharValue &dest, CharValue &src, mlir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  void createPadding(CharValue &str, mlir::Value lower, mlir::Value upper);

  /// Allocate storage (on the stack) for character given the kind and length.
  CharValue createTemp(fir::CharacterType type, mlir::Value len);

private:
  mlir::Value createBlankConstant(fir::CharacterType type);
};

/// Provide helper to generate Complex manipulations in FIR.
class ComplexOpsBuilder : public OpBuilderWrapper {
public:
  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  ComplexOpsBuilder(mlir::OpBuilder &b, mlir::Location l)
      : OpBuilderWrapper{b, l} {}
  ComplexOpsBuilder(OpBuilderWrapper &bw) : OpBuilderWrapper{bw} {}

  // Type helper. They do not create MLIR operations.
  mlir::Type getComplexPartType(mlir::Value cplx);
  mlir::Type getComplexPartType(mlir::Type complexType);
  mlir::Type getComplexPartType(fir::KindTy complexKind);

  // Complex operation creation helper. They create MLIR operations.
  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag);
  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart);
  mlir::Value insertComplexPart(mlir::Value cplx, mlir::Value part,
                                bool isImagPart);

  template <Part partId>
  mlir::Value extract(mlir::Value cplx);
  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part);

  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq);

private:
  template <Part partId>
  mlir::Value createPartId();
};

/// Get the current Module
inline mlir::ModuleOp getModule(mlir::OpBuilder *bldr) {
  return bldr->getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
}

/// Get the current Function
inline mlir::FuncOp getFunction(mlir::OpBuilder *bldr) {
  return bldr->getBlock()->getParent()->getParentOfType<mlir::FuncOp>();
}

/// Get the entry block of the current Function
inline mlir::Block *getEntryBlock(mlir::OpBuilder *bldr) {
  return &getFunction(bldr).front();
}

/// Create a new basic block
inline mlir::Block *createBlock(mlir::OpBuilder *bldr, mlir::Region *region) {
  return bldr->createBlock(region, region->end());
}

inline mlir::Block *createBlock(mlir::OpBuilder *bldr) {
  return createBlock(bldr, bldr->getBlock()->getParent());
}

/// Get a function by name (or null)
mlir::FuncOp getNamedFunction(mlir::ModuleOp, llvm::StringRef name);

/// Create a new FuncOp
mlir::FuncOp createFunction(AbstractConverter &converter, llvm::StringRef name,
                            mlir::FunctionType funcTy);

/// Create a new FuncOp
/// The function is created with no Location information
mlir::FuncOp createFunction(mlir::ModuleOp module, llvm::StringRef name,
                            mlir::FunctionType funcTy);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_BUILDER_H_
