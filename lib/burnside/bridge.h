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

#ifndef FORTRAN_BURNSIDE_BRIDGE_H_
#define FORTRAN_BURNSIDE_BRIDGE_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include <memory>

// Implement the burnside bridge from Fortran to MLIR
// https://github.com/tensorflow/mlir

namespace Fortran::common {
class IntrinsicTypeDefaultKinds;
}

namespace Fortran::parser {
struct Program;
}

namespace llvm {
class Module;
class SourceMgr;
}

namespace Fortran::burnside {

/// An instance of BurnsideBridge is a singleton that owns the state of the
/// bridge
class BurnsideBridge {
public:
  static std::unique_ptr<BurnsideBridge> create(
      const common::IntrinsicTypeDefaultKinds &defaultKinds) {
    BurnsideBridge *p = new BurnsideBridge{defaultKinds};
    return std::unique_ptr<BurnsideBridge>{p};
  }

  mlir::MLIRContext &getMLIRContext() { return *context_.get(); }
  mlir::ModuleManager &getManager() { return *manager_.get(); }
  mlir::ModuleOp getModule() { return module_.get(); }

  void parseSourceFile(llvm::SourceMgr &);

  const common::IntrinsicTypeDefaultKinds &getDefaultKinds() {
    return defaultKinds_;
  }

  bool validModule() { return getModule(); }

private:
  explicit BurnsideBridge(
      const common::IntrinsicTypeDefaultKinds &defaultKinds);
  BurnsideBridge() = delete;
  BurnsideBridge(const BurnsideBridge &) = delete;

  const common::IntrinsicTypeDefaultKinds &defaultKinds_;
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningModuleRef module_;
  std::unique_ptr<mlir::ModuleManager> manager_;
};

/// Cross the bridge from the Fortran parse-tree, etc. to FIR+OpenMP+MLIR
void crossBurnsideBridge(
    BurnsideBridge &bridge, const parser::Program &program);

/// Bridge from MLIR to LLVM-IR
std::unique_ptr<llvm::Module> LLVMBridge(mlir::ModuleOp &module);

/// instantiate the BURNSIDE bridge singleton
void instantiateBurnsideBridge(
    const common::IntrinsicTypeDefaultKinds &defaultKinds);

/// access to the default kinds class (for MLIR bridge)
const common::IntrinsicTypeDefaultKinds &getDefaultKinds();

/// get the burnside bridge singleton
BurnsideBridge &getBridge();

}  // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_BRIDGE_H_
