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

#ifndef FORTRAN_BURNSIDE_BUILDER_H_
#define FORTRAN_BURNSIDE_BUILDER_H_

#include "../semantics/symbol.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include <string>

namespace llvm {
class StringRef;
}

namespace Fortran {
namespace parser {
class CookedSource;
}

namespace evaluate {
struct ProcedureDesignator;
}

namespace burnside {

/// Miscellaneous helper routines for building MLIR
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

class AbstractConverter;

class SymMap {
  llvm::DenseMap<const semantics::Symbol *, mlir::Value *> symbolMap;
  std::vector<std::pair<const semantics::Symbol *, mlir::Value *>> shadowStack;

public:
  void addSymbol(semantics::SymbolRef symbol, mlir::Value *value);

  mlir::Value *lookupSymbol(semantics::SymbolRef symbol);

  void pushShadowSymbol(semantics::SymbolRef symbol, mlir::Value *value);
  void popShadowSymbol() { shadowStack.pop_back(); }

  void clear() {
    symbolMap.clear();
    shadowStack.clear();
  }
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
mlir::FuncOp createFunction(
    mlir::ModuleOp module, llvm::StringRef name, mlir::FunctionType funcTy);

}  // burnside
}  // Fortran

#endif  // FORTRAN_BURNSIDE_BUILDER_H_
