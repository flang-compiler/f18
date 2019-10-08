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

#include "builder.h"
#include "bridge.h"
#include "fe-helper.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"

namespace B = Fortran::burnside;

using namespace Fortran;
using namespace B;

// This will need to be extended to consider the type of what is being mangled
std::string B::applyNameMangling(llvm::StringRef parserName) {
  // FIXME: this is fake for now, add type info, etc.
  return "_Qp_"s + parserName.str();
}

mlir::FuncOp B::createFunction(
    mlir::ModuleOp module, llvm::StringRef name, mlir::FunctionType funcTy) {
  mlir::MLIRContext *ctxt{module.getContext()};
  auto func{mlir::FuncOp::create(dummyLoc(ctxt), name, funcTy)};
  module.push_back(func);
  return func;
}

mlir::FuncOp B::getNamedFunction(mlir::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<mlir::FuncOp>(name);
}

void B::SymMap::addSymbol(const semantics::Symbol *symbol, mlir::Value *value) {
  symbolMap.try_emplace(symbol, value);
}

mlir::Value *B::SymMap::lookupSymbol(const semantics::Symbol *symbol) {
  auto iter{symbolMap.find(symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}
