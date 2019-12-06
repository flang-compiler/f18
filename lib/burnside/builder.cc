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
#include "convert-type.h"
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

std::string B::applyNameMangling(const evaluate::ProcedureDesignator &proc) {
  if (const auto *symbol{proc.GetSymbol()}) {
    return applyNameMangling(*symbol);
  } else {
    // Do not mangle intrinsic for now
    assert(proc.GetSpecificIntrinsic() &&
        "expected intrinsic procedure in designator");
    return proc.GetName();
  }
}

std::string B::applyNameMangling(semantics::SymbolRef symbol) {
  // FIXME: this is fake for now, add type info, etc.
  // For now, only works for external procedures
  // TODO: apply binding
  // TODO: determine if procedure are:
  //   - external, internal or  module
  // TODO: Apply proposed mangling with _Qp_ ....
  return std::visit(
      common::visitors{
          [&](const semantics::MainProgramDetails &) { return "MAIN_"s; },
          [&](const semantics::SubprogramDetails &) {
            return symbol->name().ToString() + "_";
          },
          [&](const semantics::ProcEntityDetails &) {
            return symbol->name().ToString() + "_";
          },
          [&](const semantics::SubprogramNameDetails &) {
            assert(false &&
                "SubprogramNameDetails not expected after semantic analysis");
            return ""s;
          },
          [&](const auto &) {
            assert(false && "Symbol mangling TODO");
            return ""s;
          },
      },
      symbol->details());
}

mlir::FuncOp B::createFunction(mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType funcTy, parser::CookedSource const *cooked,
    parser::CharBlock const *cb) {
  mlir::MLIRContext *ctxt{module.getContext()};
  mlir::Location loc{
      (cooked && cb) ? parserPosToLoc(*ctxt, cooked, *cb) : dummyLoc(*ctxt)};
  auto func{mlir::FuncOp::create(loc, name, funcTy)};
  module.push_back(func);
  return func;
}

mlir::FuncOp B::getNamedFunction(mlir::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<mlir::FuncOp>(name);
}

void B::SymMap::addSymbol(semantics::SymbolRef symbol, mlir::Value *value) {
  symbolMap.try_emplace(&*symbol, value);
}

mlir::Value *B::SymMap::lookupSymbol(semantics::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void B::SymMap::pushShadowSymbol(
    semantics::SymbolRef symbol, mlir::Value *value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const semantics::Symbol *sym{nullptr};
  mlir::Value *val{nullptr};
  // if mapping exists, save it on the shadow stack
  if (iter != symbolMap.end()) {
    sym = iter->first;
    val = iter->second;
    symbolMap.erase(iter);
  }
  shadowStack.emplace_back(sym, val);
  // insert new shadow mapping
  auto r{symbolMap.try_emplace(&*symbol, value)};
  assert(r.second && "unexpected insertion failure");
  (void)r;
}
