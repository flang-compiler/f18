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

#ifndef FORTRAN_BURNSIDE_INTRINSICS_H_
#define FORTRAN_BURNSIDE_INTRINSICS_H_

#include "mlir/Dialect/StandardOps/Ops.h"
#include <optional>
#include <unordered_map>
#include <utility>

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace Fortran::burnside {

/// IntrinsicLibrary holds the runtime description of intrinsics. It aims
/// at abstracting which library version is used to implement Fortran
/// numerical intrinsics while lowering expressions.
/// It can be probed for a certain intrinsic and will return an mlir::FuncOp
/// that matches the targeted library implementation.
class IntrinsicLibrary {
public:
  /// Available intrinsic library versions.
  enum class Version { PgmathFast, PgmathRelaxed, PgmathPrecise, LLVM };
  using Key = std::pair<std::string, mlir::Type>;
  // Need a custom hash for this kind of keys. LLVM provides it.
  struct Hash {
    size_t operator()(const Key &k) const { return llvm::hash_value(k); }
  };
  /// Internal structure to describe the runtime function. An intrinsic function
  /// is not declared in mlir until the IntrinsicLibrary needs to return it.
  /// This is to avoid polluting the LLVM IR with useless declarations.
  /// This structure allows generating mlir::FuncOp on the fly.
  struct IntrinsicImplementation {
    IntrinsicImplementation(const std::string &n, mlir::FunctionType t)
      : symbol{n}, type{t} {};
    std::string symbol;
    mlir::FunctionType type;
  };
  using Map = std::unordered_map<Key, IntrinsicImplementation, Hash>;

  /// Probe the intrinsic library for a certain intrinsic and get/build the
  /// related mlir::FuncOp if a runtime description is found.
  /// Also add an unit attribute "fir.intrinsic" to the function so that later
  /// it is possible to quickly know what function are intrinsics vs users.
  std::optional<mlir::FuncOp> getFunction(
      const std::string &, const mlir::Type &, mlir::OpBuilder &) const;

  /// Create the runtime description for the targeted library version.
  static IntrinsicLibrary create(Version, mlir::MLIRContext &);

private:
  IntrinsicLibrary(Map &&l) : lib{std::move(l)} {};
  /// Holds the intrinsic runtime description to be probed.
  Map lib;
};

}

#endif  // FORTRAN_BURNSIDE_INTRINSICS_H_
