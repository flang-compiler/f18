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

#ifndef FIR_KINDMAPPING_H
#define FIR_KINDMAPPING_H

#include "llvm/IR/Type.h"
#include <map>

namespace llvm {
template <typename>
class Optional;
} // namespace llvm

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace fir {

class KindMapping {
public:
  using KindTy = unsigned;
  using Bitsize = unsigned;
  using LLVMTypeID = llvm::Type::TypeID;
  using MatchResult = llvm::Optional<bool>;

  explicit KindMapping(mlir::MLIRContext *context);
  explicit KindMapping(mlir::MLIRContext *context, llvm::StringRef map);

  /// Get the size in bits of !fir.char<kind>
  Bitsize getCharacterBitsize(KindTy kind);

  /// Get the size in bits of !fir.int<kind>
  Bitsize getIntegerBitsize(KindTy kind);

  /// Get the size in bits of !fir.logical<kind>
  Bitsize getLogicalBitsize(KindTy kind);

  /// Get the LLVM Type::TypeID of !fir.real<kind>
  LLVMTypeID getRealTypeID(KindTy kind);

  /// Get the LLVM Type::TypeID of !fir.complex<kind>
  LLVMTypeID getComplexTypeID(KindTy kind);

private:
  MatchResult badMapString(llvm::Twine const &ptr);
  MatchResult parse(llvm::StringRef kindMap);

  mlir::MLIRContext *context;
  std::map<char, std::map<KindTy, Bitsize>> intMap;
  std::map<char, std::map<KindTy, LLVMTypeID>> floatMap;
};

} // namespace fir

#endif // FIR_KINDMAPPING_H
