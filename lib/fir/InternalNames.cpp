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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace L = llvm;

namespace {

inline L::Twine prefix() { return "_Q"; }

L::Twine doModules(L::ArrayRef<L::StringRef> mods) {
  L::Twine result;
  auto *token = "M";
  for (auto mod : mods) {
    result.concat(token).concat(mod);
    token = "S";
  }
  return result;
}

L::Twine doModulesHost(L::ArrayRef<L::StringRef> mods,
                       L::Optional<L::StringRef> host) {
  L::Twine result = doModules(mods);
  if (host.hasValue())
    result.concat("F").concat(*host);
  return result;
}

} // namespace

L::Twine fir::NameMangler::toLower(L::StringRef name) {
  auto lo = name.lower();
  if (name.equals(lo))
    return name;
  return cache.insert(lo).first->getKey();
}

L::Twine fir::NameMangler::addAsString(std::int64_t i) {
  return cache.insert(std::to_string(i)).first->getKey();
}

L::Twine fir::NameMangler::doKind(std::int64_t kind) {
  if (kind < 0)
    return "KN" + addAsString(-kind);
  return "K" + addAsString(kind);
}

L::Twine fir::NameMangler::doKinds(L::ArrayRef<std::int64_t> kinds) {
  L::Twine result;
  for (auto i : kinds)
    result.concat(doKind(i));
  return result;
}

L::Twine fir::NameMangler::doCommonBlock(L::StringRef name) {
  return prefix() + "B" + toLower(name);
}

L::Twine fir::NameMangler::doConstant(L::ArrayRef<L::StringRef> modules,
                                      L::StringRef name) {
  return prefix() + doModules(modules) + "EC" + toLower(name);
}

L::Twine fir::NameMangler::doDispatchTable(L::ArrayRef<L::StringRef> modules,
                                           L::Optional<L::StringRef> host,
                                           L::StringRef name,
                                           L::ArrayRef<std::int64_t> kinds) {
  return prefix() + doModulesHost(modules, host) + "DT" + toLower(name) +
         doKinds(kinds);
}

L::Twine fir::NameMangler::doGenerated(L::StringRef name) {
  return prefix() + "Q" + toLower(name);
}

L::Twine fir::NameMangler::doIntrinsicTypeDescriptor(
    L::ArrayRef<L::StringRef> modules, L::Optional<L::StringRef> host,
    IntrinsicType type, std::int64_t kind) {
  char const *name;
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
  return prefix() + doModulesHost(modules, host) + "C" + name + doKind(kind);
}

L::Twine fir::NameMangler::doProcedure(L::ArrayRef<L::StringRef> modules,
                                       L::Optional<L::StringRef> host,
                                       L::StringRef name) {
  return prefix() + doModulesHost(modules, host) + "P" + toLower(name);
}

L::Twine fir::NameMangler::doType(L::ArrayRef<L::StringRef> modules,
                                  L::Optional<L::StringRef> host,
                                  L::StringRef name,
                                  L::ArrayRef<std::int64_t> kinds) {
  return prefix() + doModulesHost(modules, host) + "T" + toLower(name) +
         doKinds(kinds);
}

L::Twine fir::NameMangler::doTypeDescriptor(L::ArrayRef<L::StringRef> modules,
                                            L::Optional<L::StringRef> host,
                                            L::StringRef name,
                                            L::ArrayRef<std::int64_t> kinds) {
  return prefix() + doModulesHost(modules, host) + "CT" + toLower(name) +
         doKinds(kinds);
}

L::Twine fir::NameMangler::doVariable(L::ArrayRef<L::StringRef> modules,
                                      L::StringRef name) {
  return prefix() + doModules(modules) + "E" + toLower(name);
}
