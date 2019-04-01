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

#ifndef FORTRAN_BURNSIDE_RUNTIME_H_
#define FORTRAN_BURNSIDE_RUNTIME_H_

#include <string>

namespace llvm {
class StringRef;
}
namespace mlir {
class FunctionType;
class MLIRContext;
}

namespace Fortran::burnside {

// In the Fortran::burnside namespace, the code will default follow the
// LLVM/MLIR coding standards

#define DEFINE_RUNTIME_ENTRY(A, B, C, D) FIRT_##A,
enum RuntimeEntryCode {
#include "runtime.def"
  FIRT_LAST_ENTRY_CODE
};

llvm::StringRef getRuntimeEntryName(RuntimeEntryCode code);

mlir::FunctionType getRuntimeEntryType(
    RuntimeEntryCode code, mlir::MLIRContext &mlirContext, int kind);

mlir::FunctionType getRuntimeEntryType(RuntimeEntryCode code,
    mlir::MLIRContext &mlirContext, int inpKind, int resKind);

}  // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_RUNTIME_H_
