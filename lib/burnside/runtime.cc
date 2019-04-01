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

#include "runtime.h"
#include "fir/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include <cassert>

namespace Br = Fortran::burnside;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

using FuncPointer = llvm::SmallVector<mlir::Type, 4> (*)(
    mlir::MLIRContext *, int kind);

// FIXME: these are wrong and just serving as temporary placeholders
using Cplx = mlir::IntegerType;
using Intopt = mlir::IntegerType;
using TypeDesc = mlir::IntegerType;

template<typename A, typename... B>
void consType(
    llvm::SmallVector<mlir::Type, 4> &types, mlir::MLIRContext *ctx, int kind) {
  if constexpr (std::is_same_v<A, mlir::IntegerType>) {
    types.emplace_back(A::get(kind, ctx));
  } else if constexpr (std::is_same_v<A, mlir::ComplexType>) {
    // FIXME: this is wrong
    assert(false);
    types.emplace_back(mlir::IntegerType::get(kind, ctx));
  } else {
    types.emplace_back(A::get(ctx, kind));
  }
  if constexpr (sizeof...(B) > 0) {
    consType<B...>(types, ctx, kind);
  }
}

template<typename... A>
llvm::SmallVector<mlir::Type, 4> consType(mlir::MLIRContext *ctx, int kind) {
  llvm::SmallVector<mlir::Type, 4> types;
  if constexpr (sizeof...(A) > 0) {
    consType<A...>(types, ctx, kind);
  } else {
    (void)ctx;
    (void)kind;
  }
  return types;
}

#define DEFINE_RUNTIME_ENTRY(A, B, C, D) "Fortran_" B,
char const *const RuntimeEntryNames[FIRT_LAST_ENTRY_CODE] = {
#include "runtime.def"
};

#define UNPACK_TYPES(...) __VA_ARGS__
#define DEFINE_RUNTIME_ENTRY(A, B, C, D) consType<UNPACK_TYPES C>,
FuncPointer RuntimeEntryInputType[FIRT_LAST_ENTRY_CODE] = {
#include "runtime.def"
};

#define DEFINE_RUNTIME_ENTRY(A, B, C, D) consType<UNPACK_TYPES D>,
FuncPointer RuntimeEntryResultType[FIRT_LAST_ENTRY_CODE] = {
#include "runtime.def"
};
#undef UNPACK_TYPES

}  // namespace

llvm::StringRef Br::getRuntimeEntryName(RuntimeEntryCode code) {
  assert(code < FIRT_LAST_ENTRY_CODE);
  return {RuntimeEntryNames[code]};
}

mlir::FunctionType Br::getRuntimeEntryType(
    RuntimeEntryCode code, mlir::MLIRContext &mlirContext, int kind) {
  assert(code < FIRT_LAST_ENTRY_CODE);
  return mlir::FunctionType::get(
      RuntimeEntryInputType[code](&mlirContext, kind),
      RuntimeEntryResultType[code](&mlirContext, kind), &mlirContext);
}

mlir::FunctionType Br::getRuntimeEntryType(RuntimeEntryCode code,
    mlir::MLIRContext &mlirContext, int inpKind, int resKind) {
  assert(code < FIRT_LAST_ENTRY_CODE);
  return mlir::FunctionType::get(
      RuntimeEntryInputType[code](&mlirContext, inpKind),
      RuntimeEntryResultType[code](&mlirContext, resKind), &mlirContext);
}
