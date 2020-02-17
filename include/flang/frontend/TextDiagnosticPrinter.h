//===--- TextDiagnosticPrinter.h - Text Diagnostic Client -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a dummy implemenation of concrete diagnostic client for flang.
// TODO: Print diagnostics to standard error.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

class TextDiagnosticPrinter : public clang::DiagnosticConsumer {
  llvm::raw_ostream &OS;
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts;

  unsigned OwnsOutputStream : 1;

public:
  TextDiagnosticPrinter(llvm::raw_ostream &os, clang::DiagnosticOptions *diags,
      bool OwnsOutputStream = false);
  ~TextDiagnosticPrinter() override;
};
