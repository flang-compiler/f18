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
///
/// In order to generate symbolically referencable artifacts in a ModuleOp,
/// it is required that those symbols be uniqued.  This is a simple interface
/// for converting Fortran symbols into unique names.
///
/// This is intentionally bijective. Given a symbol's parse name, type, and
/// scope-like information, we can generate a uniqued (mangled) name.  Given a
/// uniqued name, we can return the symbol parse name, type of the symbol, and
/// any scope-like information for that symbol.
struct NameUniquer {
  enum class IntrinsicType { CHARACTER, COMPLEX, INTEGER, LOGICAL, REAL };

  enum class NameKind {
    NOT_UNIQUED,
    COMMON,
    CONSTANT,
    DERIVED_TYPE,
    DISPATCH_TABLE,
    GENERATED,
    INTRINSIC_TYPE_DESC,
    PROCEDURE,
    TYPE_DESC,
    VARIABLE
  };

  struct DeconstructedName {
    llvm::SmallVector<std::string, 2> modules;
    llvm::Optional<std::string> host;
    std::string name;
    llvm::SmallVector<std::int64_t, 4> kinds;
  };

  NameUniquer() = default;

  /// Unique a common block name
  std::string doCommonBlock(llvm::StringRef name);

  /// Unique a (global) constant name
  std::string doConstant(llvm::ArrayRef<llvm::StringRef> modules,
                         llvm::StringRef name);

  /// Unique a dispatch table name
  std::string doDispatchTable(llvm::ArrayRef<llvm::StringRef> modules,
                              llvm::Optional<llvm::StringRef> host,
                              llvm::StringRef name,
                              llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a compiler generated name
  std::string doGenerated(llvm::StringRef name);

  /// Unique an intrinsic type descriptor
  std::string doIntrinsicTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                                        llvm::Optional<llvm::StringRef> host,
                                        IntrinsicType type, std::int64_t kind);

  /// Unique a procedure name
  std::string doProcedure(llvm::ArrayRef<llvm::StringRef> modules,
                          llvm::Optional<llvm::StringRef> host,
                          llvm::StringRef name);

  /// Unique a derived type name
  std::string doType(llvm::ArrayRef<llvm::StringRef> modules,
                     llvm::Optional<llvm::StringRef> host, llvm::StringRef name,
                     llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a (derived) type descriptor name
  std::string doTypeDescriptor(llvm::ArrayRef<llvm::StringRef> modules,
                               llvm::Optional<llvm::StringRef> host,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::int64_t> kinds);

  /// Unique a (global) variable name
  std::string doVariable(llvm::ArrayRef<llvm::StringRef> modules,
                         llvm::StringRef name);

  /// Entry point for the PROGRAM (called by the runtime)
  constexpr static llvm::StringRef doProgramEntry() { return "MAIN_"; }

  /// Decompose `uniquedName` into the parse name, symbol type, and scope info
  static std::pair<NameKind, DeconstructedName>
  deconstruct(llvm::StringRef uniquedName);

private:
  llvm::StringRef addAsString(std::int64_t i);
  std::string doKind(std::int64_t kind);
  std::string doKinds(llvm::ArrayRef<std::int64_t> kinds);
  llvm::StringRef toLower(llvm::StringRef name);

  llvm::StringSet<> cache;
};

} // namespace fir

#endif // FIR_INTERNAL_NAMES_H
