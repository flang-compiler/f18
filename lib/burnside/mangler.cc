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

#include "mangler.h"
#include "../common/reference.h"
#include "../semantics/tools.h"
#include "fir/InternalNames.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace L = llvm;
namespace Ma = Fortran::burnside::mangle;
namespace Se = Fortran::semantics;

using namespace Fortran;

namespace {

L::StringRef toStringRef(const parser::CharBlock &cb) {
  return L::StringRef{cb.begin(), cb.size()};
}

// recursively build the vector of module scopes
void moduleNames(
    const Se::Scope *scope, L::SmallVector<L::StringRef, 2> &result) {
  if (scope->kind() == Se::Scope::Kind::Global) {
    return;
  }
  moduleNames(&scope->parent(), result);
  if (scope->kind() == Se::Scope::Kind::Module)
    if (auto *symbol = scope->symbol())
      result.emplace_back(toStringRef(symbol->name()));
}

L::SmallVector<L::StringRef, 2> moduleNames(const Se::Scope *scope) {
  L::SmallVector<L::StringRef, 2> result;
  moduleNames(scope, result);
  return result;
}

L::Optional<L::StringRef> hostName(const Se::Scope *scope) {
  if (scope->kind() == Se::Scope::Kind::Subprogram) {
    auto &parent = scope->parent();
    if (parent.kind() == Se::Scope::Kind::Subprogram)
      if (auto *symbol = parent.symbol()) {
        return {toStringRef(symbol->name())};
      }
  }
  return {};
}

}  // namespace

// Mangle the name of `symbol` to make it unique within FIR's symbol table using
// the FIR name mangler, `mangler`
std::string Ma::mangleName(
    fir::NameUniquer &uniquer, const Se::SymbolRef symbol) {
  return std::visit(Co::visitors{
                        [&](const Se::MainProgramDetails &) {
                          return uniquer.doProgramEntry().str();
                        },
                        [&](const Se::SubprogramDetails &) {
                          auto &cb{symbol->name()};
                          auto modNames{moduleNames(symbol->scope())};
                          return uniquer.doProcedure(modNames,
                              hostName(symbol->scope()), toStringRef(cb));
                        },
                        [&](const Se::ProcEntityDetails &) {
                          auto &cb{symbol->name()};
                          auto modNames{moduleNames(symbol->scope())};
                          return uniquer.doProcedure(modNames,
                              hostName(symbol->scope()), toStringRef(cb));
                        },
                        [](const auto &) -> std::string {
                          assert(false);
                          return {};
                        },
                    },
      symbol->details());
}

std::string Ma::demangleName(L::StringRef name) {
  auto result{fir::NameUniquer::deconstruct(name)};
  return result.second.name;
}
