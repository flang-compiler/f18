//===-- main.cpp - Flang e Driver -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the flang driver; it is a thin wrapper
// for functionality in the Driver flang library.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "../../include/flang/frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"

std::string GetExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void *)(intptr_t)GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance
static clang::DiagnosticOptions *CreateAndPopulateDiagOpts(
    llvm::ArrayRef<const char *> argv) {
  auto *DiagOpts = new clang::DiagnosticOptions;
  return DiagOpts;
}

int main(int argc_, const char **argv_) {

  llvm::InitLLVM X(argc_, argv_);
  llvm::SmallVector<const char *, 256> argv(argv_, argv_ + argc_);

  clang::driver::ParsedClangName TargetandMode("flang", "--driver-mode=flang");
  std::string Path = GetExecutablePath(argv[0]);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts = CreateAndPopulateDiagOpts(argv);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  TextDiagnosticPrinter *DiagClient = new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  clang::DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  clang::driver::Driver TheDriver(
      Path, llvm::sys::getDefaultTargetTriple(), Diags);
  TheDriver.setTargetAndMode(TargetandMode);
  std::unique_ptr<clang::driver::Compilation> C(
      TheDriver.BuildCompilation(argv));
  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
      FailingCommands;

  return TheDriver.ExecuteCompilation(*C, FailingCommands);
}
