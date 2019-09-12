//===- FrontendAction.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTEND_FRONTENDACTION_H
#define LLVM_FLANG_FRONTEND_FRONTENDACTION_H

#include "flang/Frontend/FrontendOptions.h"

#include "llvm/Support/Error.h"

namespace clang {

class CompilerInstance;

class FrontendAction {
public:
  virtual ~FrontendAction() = default;

  virtual bool isModelParsingAction() const { return false; }

  virtual bool PrepareToExecute(CompilerInstance &CI) { return true; }
  virtual bool BeginSourceFile(CompilerInstance &CI, const FrontendInputFile &Input);
  virtual llvm::Error Execute();
  virtual void EndSourceFile();
};

}

#endif