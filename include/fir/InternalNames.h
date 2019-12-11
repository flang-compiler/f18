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

#ifndef FIR_INTERNAL_NAMES_H
#define FIR_INTERNAL_NAMES_H

#include "llvm/ADT/StringSet.h"
#include <cstdint>

namespace llvm {
template <typename>
class ArrayRef;
template <typename>
class Optional;
class Twine;
} // namespace llvm

namespace fir {

/// Internal name mangling of identifiers
struct NameMangler {
  enum class IntrinsicType { CHARACTER, COMPLEX, INTEGER, LOGICAL, REAL };

  NameMangler() = default;

  /// Mangle a common block name
  std::string doCommonBlock(llvm::StringRef name);

  /// Mangle a (global) constant name
  std::string doConstant(llvm::ArrayRef<llvm::StringRef> modules,
                         llvm::StringRef name);

  /// Mangle a dispatch table name
  std::string doDispatchTable(llvm::ArrayRef<llvm::StringRef> modules,
                              llvm::Optional<llvm::StringRef> host,
                              llvm::StringRef name,
                              llvm::ArrayRef<std::int64_t> kinds);

  /// Mangle a compiler generated name
  std::string doGenerated(llvm::StringRef name);

  /// Mangle an intrinsic type descriptor
  std::string doIntrinsicTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                                        llvm::Optional<llvm::StringRef> host,
                                        IntrinsicType type, std::int64_t kind);

  /// Mangle a procedure name
  std::string doProcedure(llvm::ArrayRef<llvm::StringRef> modules,
                          llvm::Optional<llvm::StringRef> host,
                          llvm::StringRef name);

  /// Mangle a derived type name
  std::string doType(llvm::ArrayRef<llvm::StringRef> modules,
                     llvm::Optional<llvm::StringRef> host, llvm::StringRef name,
                     llvm::ArrayRef<std::int64_t> kinds);

  /// Mangle a (derived) type descriptor name
  std::string doTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                               llvm::Optional<llvm::StringRef> host,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::int64_t> kinds);

  /// Mangle a (global) variable name
  std::string doVariable(llvm::ArrayRef<llvm::StringRef> modules,
                         llvm::StringRef name);

  /// Entry point for the PROGRAM (called by the runtime)
  constexpr static llvm::StringRef doProgramEntry() {
    return "MAIN_";
  }

private:
  llvm::StringRef addAsString(std::int64_t i);
  std::string doKind(std::int64_t kind);
  std::string doKinds(llvm::ArrayRef<std::int64_t> kinds);
  llvm::StringRef toLower(llvm::StringRef name);

  llvm::StringSet<> cache;
};

} // namespace fir

#endif // FIR_INTERNAL_NAMES_H
