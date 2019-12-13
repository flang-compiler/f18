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

#include "fir/InternalNames.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"

namespace L = llvm;

namespace {

inline L::StringRef prefix() { return "_Q"; }

std::string doModules(L::ArrayRef<L::StringRef> mods) {
  std::string result;
  auto *token = "M";
  for (auto mod : mods) {
    L::Twine t = result + token + mod;
    result = t.str();
    token = "S";
  }
  return result;
}

std::string doModulesHost(L::ArrayRef<L::StringRef> mods,
                          L::Optional<L::StringRef> host) {
  auto result = doModules(mods);
  if (host.hasValue()) {
    L::Twine t = result + "F" + *host;
    result = t.str();
  }
  return result;
}

} // namespace

L::StringRef fir::NameUniquer::toLower(L::StringRef name) {
  auto lo = name.lower();
  if (name.equals(lo))
    return name;
  return cache.insert(lo).first->getKey();
}

L::StringRef fir::NameUniquer::addAsString(std::int64_t i) {
  return cache.insert(std::to_string(i)).first->getKey();
}

std::string fir::NameUniquer::doKind(std::int64_t kind) {
  if (kind < 0) {
    L::Twine result = "KN" + addAsString(-kind);
    return result.str();
  }
  L::Twine result = "K" + addAsString(kind);
  return result.str();
}

std::string fir::NameUniquer::doKinds(L::ArrayRef<std::int64_t> kinds) {
  L::Twine result;
  for (auto i : kinds)
    result.concat(doKind(i));
  return result.str();
}

std::string fir::NameUniquer::doCommonBlock(L::StringRef name) {
  L::Twine result = prefix() + "B" + toLower(name);
  return result.str();
}

std::string fir::NameUniquer::doConstant(L::ArrayRef<L::StringRef> modules,
                                         L::StringRef name) {
  L::Twine result = prefix() + doModules(modules) + "EC" + toLower(name);
  return result.str();
}

std::string fir::NameUniquer::doDispatchTable(L::ArrayRef<L::StringRef> modules,
                                              L::Optional<L::StringRef> host,
                                              L::StringRef name,
                                              L::ArrayRef<std::int64_t> kinds) {
  L::Twine result = prefix() + doModulesHost(modules, host) + "DT" +
                    toLower(name) + doKinds(kinds);
  return result.str();
}

std::string fir::NameUniquer::doGenerated(L::StringRef name) {
  L::Twine result = prefix() + "Q" + toLower(name);
  return result.str();
}

std::string fir::NameUniquer::doIntrinsicTypeDescriptor(
    L::ArrayRef<L::StringRef> modules, L::Optional<L::StringRef> host,
    IntrinsicType type, std::int64_t kind) {
  const char *name{nullptr};
  switch (type) {
  case IntrinsicType::CHARACTER:
    name = "character";
    break;
  case IntrinsicType::COMPLEX:
    name = "complex";
    break;
  case IntrinsicType::INTEGER:
    name = "integer";
    break;
  case IntrinsicType::LOGICAL:
    name = "logical";
    break;
  case IntrinsicType::REAL:
    name = "real";
    break;
  }
  L::Twine result =
      prefix() + doModulesHost(modules, host) + "C" + name + doKind(kind);
  return result.str();
}

std::string fir::NameUniquer::doProcedure(L::ArrayRef<L::StringRef> modules,
                                          L::Optional<L::StringRef> host,
                                          L::StringRef name) {
  L::Twine result =
      prefix() + doModulesHost(modules, host) + "P" + toLower(name);
  return result.str();
}

std::string fir::NameUniquer::doType(L::ArrayRef<L::StringRef> modules,
                                     L::Optional<L::StringRef> host,
                                     L::StringRef name,
                                     L::ArrayRef<std::int64_t> kinds) {
  L::Twine result = prefix() + doModulesHost(modules, host) + "T" +
                    toLower(name) + doKinds(kinds);
  return result.str();
}

std::string fir::NameUniquer::doTypeDescriptor(
    L::ArrayRef<L::StringRef> modules, L::Optional<L::StringRef> host,
    L::StringRef name, L::ArrayRef<std::int64_t> kinds) {
  L::Twine result = prefix() + doModulesHost(modules, host) + "CT" +
                    toLower(name) + doKinds(kinds);
  return result.str();
}

std::string fir::NameUniquer::doVariable(L::ArrayRef<L::StringRef> modules,
                                         L::StringRef name) {
  L::Twine result = prefix() + doModules(modules) + "E" + toLower(name);
  return result.str();
}

std::pair<fir::NameUniquer::NameKind, fir::NameUniquer::DeconstructedName>
fir::NameUniquer::deconstruct(L::StringRef uniquedName) {
  assert(false && "not yet implemented");
  return {};
}
