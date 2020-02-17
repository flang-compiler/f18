//===--- TextDiagnosticPrinter.cpp - Diagnostic Printer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client ATM only creates and destroys the object.
// TODO: Diagnostic client should print the diagnostic messages for flang.
//
//===----------------------------------------------------------------------===//

#include "../../include/flang/frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/raw_ostream.h"

TextDiagnosticPrinter::TextDiagnosticPrinter(llvm::raw_ostream &os,
    clang::DiagnosticOptions *diags, bool _OwnsOutputStream)
  : OS(os), DiagOpts(diags), OwnsOutputStream(_OwnsOutputStream) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {
  if (OwnsOutputStream) delete &OS;
}
