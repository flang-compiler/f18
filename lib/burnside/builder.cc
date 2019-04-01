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

namespace Br = Fortran::burnside;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

// This will need to be extended to consider the type of what is being mangled
std::string Br::applyNameMangling(llvm::StringRef parserName) {
  // FIXME: this is fake for now, add type info, etc.
  return "_Qp_"s + parserName.str();
}

M::FuncOp Br::createFunction(
    M::ModuleOp module, const std::string &name, M::FunctionType funcTy) {
  M::MLIRContext *ctxt{module.getContext()};
  auto func{M::FuncOp::create(dummyLoc(ctxt), name, funcTy)};
  module.push_back(func);
  return func;
}

M::FuncOp Br::getNamedFunction(llvm::StringRef name) {
  return getBridge().getManager().lookupSymbol<M::FuncOp>(name);
}

// symbol map: {Symbol* -> Value*}

void Br::SymMap::addSymbol(const Se::Symbol *symbol, M::Value *value) {
  sMap.try_emplace(symbol, value);
}

M::Value *Br::SymMap::lookupSymbol(const Se::Symbol *symbol) {
  auto iter{sMap.find(symbol)};
  return (iter == sMap.end()) ? nullptr : iter->second;
}
