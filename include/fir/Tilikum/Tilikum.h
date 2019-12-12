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

#ifndef FIR_TILIKUM_TILIKUM_H
#define FIR_TILIKUM_TILIKUM_H

#include <memory>

namespace llvm {
class StringRef;
}
namespace mlir {
class Pass;
}

namespace fir {

struct NameUniquer;

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<mlir::Pass> createFIRToLLVMPass(NameUniquer &uniquer);

/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<mlir::Pass>
createLLVMDialectToLLVMPass(llvm::StringRef output);

} // namespace fir

#endif // FIR_TILIKUM_TILIKUM_H
