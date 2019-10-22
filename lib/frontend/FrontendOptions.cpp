//===- FrontendOptions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendOptions.h"
// #include "clang/Basic/LangStandard.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

// InputKind FrontendOptions::getInputKindForExtension(StringRef Extension) {
//   return llvm::StringSwitch<InputKind>(Extension)
//       .Case("fortran", Language::Fortran)
//       .Cases("ll", "bc", Language::LLVM_IR)
//       .Default(Language::Unknown);
// }
