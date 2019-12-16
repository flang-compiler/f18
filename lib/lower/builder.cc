//===-- lib/lower/builder.cc ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "builder.h"
#include "bridge.h"
#include "convert-type.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "optimizer/FIROpsSupport.h"
#include "llvm/ADT/StringRef.h"

namespace B = Fortran::lower;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::lower;

M::FuncOp B::createFunction(B::AbstractConverter &converter,
                            llvm::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(converter.getCurrentLocation(),
                           converter.getModuleOp(), name, funcTy);
}

M::FuncOp B::createFunction(M::ModuleOp module, llvm::StringRef name,
                            M::FunctionType funcTy) {
  return fir::createFuncOp(M::UnknownLoc::get(module.getContext()), module,
                           name, funcTy);
}

M::FuncOp B::getNamedFunction(M::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<M::FuncOp>(name);
}

void B::SymMap::addSymbol(Se::SymbolRef symbol, M::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

M::Value B::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void B::SymMap::pushShadowSymbol(Se::SymbolRef symbol, M::Value value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const Se::Symbol *sym{nullptr};
  M::Value val;
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
