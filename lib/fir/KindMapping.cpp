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

#include "fir/KindMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"

/// Allow the user to set the FIR intrinsic type kind value to LLVM type
/// mappings.  Note that these are not mappings from kind values to any
/// other MLIR dialect, only to LLVM IR. The default values follow the f18
/// front-end kind mappings.

namespace M = mlir;

using namespace fir;
using namespace llvm;

using Bitsize = KindMapping::Bitsize;
using KindTy = KindMapping::KindTy;
using LLVMTypeID = KindMapping::LLVMTypeID;
using MatchResult = KindMapping::MatchResult;

static cl::opt<std::string>
    ClKindMapping("kind-mapping",
                  cl::desc("kind mapping string to set kind precision"),
                  cl::value_desc("kind-mapping-string"), cl::init(""));

namespace {

/// Integral types default to the kind value being the size of the value in
/// bytes. The default is to scale from bytes to bits.
Bitsize defaultScalingKind(KindTy kind) {
  const unsigned BITS_IN_BYTE = 8;
  return kind * BITS_IN_BYTE;
}

/// Floating-point types default to the kind value being the size of the value
/// in bytes. The default is to translate kinds of 2, 4, 8, 10, and 16 to a
/// valid llvm::Type::TypeID value. Otherwise, the default is FloatTyID.
LLVMTypeID defaultRealKind(KindTy kind) {
  switch (kind) {
  case 2:
    return LLVMTypeID::HalfTyID;
  case 4:
    return LLVMTypeID::FloatTyID;
  case 8:
    return LLVMTypeID::DoubleTyID;
  case 10:
    return LLVMTypeID::X86_FP80TyID;
  case 16:
    return LLVMTypeID::FP128TyID;
  default:
    return LLVMTypeID::FloatTyID;
  }
}

template <typename RT, char KEY>
RT doLookup(std::function<RT(KindTy)> def,
            std::map<char, std::map<KindTy, RT>> const &map, KindTy kind) {
  auto iter = map.find(KEY);
  if (iter != map.end()) {
    auto iter2 = iter->second.find(kind);
    if (iter2 != iter->second.end())
      return iter2->second;
  }
  return def(kind);
}

template <char KEY, typename MAP>
Bitsize getIntegerLikeBitsize(KindTy kind, MAP const &map) {
  return doLookup<Bitsize, KEY>(defaultScalingKind, map, kind);
}

template <char KEY, typename MAP>
LLVMTypeID getFloatLikeTypeID(KindTy kind, MAP const &map) {
  return doLookup<LLVMTypeID, KEY>(defaultRealKind, map, kind);
}

MatchResult parseCode(char &code, char const *&ptr) {
  if (*ptr != 'a' && *ptr != 'c' && *ptr != 'i' && *ptr != 'l' && *ptr != 'r')
    return {};
  code = *ptr++;
  return {true};
}

template <char ch>
MatchResult parseSingleChar(char const *&ptr) {
  if (*ptr != ch)
    return {};
  ++ptr;
  return {true};
}

MatchResult parseColon(char const *&ptr) { return parseSingleChar<':'>(ptr); }

MatchResult parseComma(char const *&ptr) { return parseSingleChar<','>(ptr); }

MatchResult parseInt(unsigned &result, char const *&ptr) {
  char const *beg = ptr;
  while (*ptr >= '0' && *ptr <= '9')
    ptr++;
  if (beg == ptr)
    return {};
  StringRef ref(beg, ptr - beg);
  int temp;
  if (ref.consumeInteger(10, temp))
    return {};
  result = temp;
  return {true};
}

bool matchString(char const *&ptr, StringRef literal) {
  StringRef s(ptr);
  if (s.startswith(literal)) {
    ptr += literal.size();
    return true;
  }
  return false;
}

MatchResult parseTypeID(LLVMTypeID &result, char const *&ptr) {
  if (matchString(ptr, "Half")) {
    result = LLVMTypeID::HalfTyID;
    return {true};
  }
  if (matchString(ptr, "Float")) {
    result = LLVMTypeID::FloatTyID;
    return {true};
  }
  if (matchString(ptr, "Double")) {
    result = LLVMTypeID::DoubleTyID;
    return {true};
  }
  if (matchString(ptr, "X86_FP80")) {
    result = LLVMTypeID::X86_FP80TyID;
    return {true};
  }
  if (matchString(ptr, "FP128")) {
    result = LLVMTypeID::FP128TyID;
    return {true};
  }
  return {};
}

} // namespace

fir::KindMapping::KindMapping(mlir::MLIRContext *context, StringRef map)
    : context{context} {
  parse(map);
}

fir::KindMapping::KindMapping(mlir::MLIRContext *context)
    : KindMapping{context, ClKindMapping} {}

MatchResult fir::KindMapping::badMapString(Twine const &ptr) {
  auto unknown = mlir::UnknownLoc::get(context);
  mlir::emitError(unknown, ptr);
  return {};
}

MatchResult fir::KindMapping::parse(StringRef kindMap) {
  if (kindMap.empty())
    return {true};
  char const *srcPtr = kindMap.begin();
  while (true) {
    char code;
    KindTy kind;
    if (parseCode(code, srcPtr) || parseInt(kind, srcPtr))
      return badMapString(srcPtr);
    if (code == 'a' || code == 'i' || code == 'l') {
      Bitsize bits;
      if (parseColon(srcPtr) || parseInt(bits, srcPtr))
        return badMapString(srcPtr);
      intMap[code][kind] = bits;
    } else if (code == 'r' || code == 'c') {
      LLVMTypeID id;
      if (parseColon(srcPtr) || parseTypeID(id, srcPtr))
        return badMapString(srcPtr);
      floatMap[code][kind] = id;
    } else {
      return badMapString(srcPtr);
    }
    if (parseComma(srcPtr))
      break;
  }
  if (*srcPtr)
    return badMapString(srcPtr);
  return {true};
}

Bitsize fir::KindMapping::getCharacterBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'a'>(kind, intMap);
}

Bitsize fir::KindMapping::getIntegerBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'i'>(kind, intMap);
}

Bitsize fir::KindMapping::getLogicalBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'l'>(kind, intMap);
}

LLVMTypeID fir::KindMapping::getRealTypeID(KindTy kind) {
  return getFloatLikeTypeID<'r'>(kind, floatMap);
}

LLVMTypeID fir::KindMapping::getComplexTypeID(KindTy kind) {
  return getFloatLikeTypeID<'c'>(kind, floatMap);
}
