//===-- lib/fir/InternalNames.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "optimizer/InternalNames.h"
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

inline L::SmallVector<L::StringRef, 2>
convertToStringRef(L::ArrayRef<std::string> from) {
  return {from.begin(), from.end()};
}

inline L::Optional<L::StringRef>
convertToStringRef(const L::Optional<std::string> &from) {
  L::Optional<L::StringRef> to;
  if (from.hasValue())
    to = from.getValue();
  return to;
}

std::string readName(L::StringRef uniq, std::size_t &i, std::size_t init,
                     std::size_t end) {
  for (i = init; i < end && uniq[i] >= 'a' && uniq[i] <= 'z'; ++i) {
    // do nothing
  }
  return uniq.substr(init, i);
}

std::int64_t readInt(L::StringRef uniq, std::size_t &i, std::size_t init,
                     std::size_t end) {
  for (i = init; i < end && uniq[i] >= '0' && uniq[i] <= '9'; ++i) {
    // do nothing
  }
  std::int64_t result;
  uniq.substr(init, i).getAsInteger(10, result);
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

std::string fir::NameUniquer::doTypeDescriptor(
    L::ArrayRef<std::string> modules, L::Optional<std::string> host,
    L::StringRef name, L::ArrayRef<std::int64_t> kinds) {
  auto rmodules = convertToStringRef(modules);
  auto rhost = convertToStringRef(host);
  return doTypeDescriptor(rmodules, rhost, name, kinds);
}

std::string fir::NameUniquer::doVariable(L::ArrayRef<L::StringRef> modules,
                                         L::StringRef name) {
  L::Twine result = prefix() + doModules(modules) + "E" + toLower(name);
  return result.str();
}

std::pair<fir::NameUniquer::NameKind, fir::NameUniquer::DeconstructedName>
fir::NameUniquer::deconstruct(L::StringRef uniq) {
  if (uniq.startswith("_Q")) {
    L::SmallVector<std::string, 4> modules;
    L::Optional<std::string> host;
    std::string name;
    L::SmallVector<std::int64_t, 8> kinds;
    NameKind nk = NameKind::NOT_UNIQUED;
    for (std::size_t i = 2, end = uniq.size(); i != end;) {
      switch (uniq[i]) {
      case 'B':
        nk = NameKind::COMMON;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'C':
        if (uniq[i + 1] == 'T') {
          nk = NameKind::TYPE_DESC;
          name = readName(uniq, i, i + 2, end);
        } else {
          nk = NameKind::INTRINSIC_TYPE_DESC;
          name = readName(uniq, i, i + 1, end);
        }
        break;
      case 'D':
        nk = NameKind::DISPATCH_TABLE;
        assert(uniq[i + 1] == 'T');
        name = readName(uniq, i, i + 2, end);
        break;
      case 'E':
        if (uniq[i + 1] == 'C') {
          nk = NameKind::CONSTANT;
          name = readName(uniq, i, i + 2, end);
        } else {
          nk = NameKind::VARIABLE;
          name = readName(uniq, i, i + 1, end);
        }
        break;
      case 'P':
        nk = NameKind::PROCEDURE;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'Q':
        nk = NameKind::GENERATED;
        name = readName(uniq, i, i + 1, end);
        break;
      case 'T':
        nk = NameKind::DERIVED_TYPE;
        name = readName(uniq, i, i + 1, end);
        break;

      case 'M':
      case 'S':
        modules.push_back(readName(uniq, i, i + 1, end));
        break;
      case 'F':
        host = readName(uniq, i, i + 1, end);
        break;
      case 'K':
        if (uniq[i + 1] == 'N')
          kinds.push_back(-readInt(uniq, i, i + 2, end));
        else
          kinds.push_back(readInt(uniq, i, i + 1, end));
        break;

      default:
        assert(false && "unknown uniquing code");
        break;
      }
    }
    return {nk, DeconstructedName(modules, host, name, kinds)};
  }
  return {NameKind::NOT_UNIQUED, DeconstructedName(uniq)};
}
