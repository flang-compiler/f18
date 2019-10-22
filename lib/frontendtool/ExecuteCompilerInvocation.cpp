//===--- ExecuteCompilerInvocation.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds ExecuteCompilerInvocation(). It is split into its own file to
// minimize the impact of pulling in essentially everything else in Clang.
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/FrontendDiagnostic.h"
// #include "flang/Frontend/Utils.h"
// #include "flang/FrontendTool/Utils.h"

#include "clang/Driver/Options.h"

#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace llvm::opt;

namespace clang {

static std::unique_ptr<FrontendAction>
CreateFrontendBaseAction(CompilerInstance &CI) {
  using namespace clang::frontend;
  StringRef Action("unknown");
  (void)Action;

  ActionKind AK = CI.getFrontendOpts().ProgramAction;
  switch (AK) {
    default:
      break;
  //case ASTDeclList:            return std::make_unique<ASTDeclListAction>();
  //case ASTDump:                return std::make_unique<ASTDumpAction>();
  //case ASTPrint:               return std::make_unique<ASTPrintAction>();
  //case ASTView:                return std::make_unique<ASTViewAction>();
  //case DumpCompilerOptions:    return std::make_unique<DumpCompilerOptionsAction>();
  //case DumpRawTokens:          return std::make_unique<DumpRawTokensAction>();
  //case DumpTokens:             return std::make_unique<DumpTokensAction>();
  //case EmitAssembly:           return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitBC:                 return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitHTML:               return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitLLVM:               return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitLLVMOnly:           return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitCodeGenOnly:        return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case EmitObj:                return nullptr; // TODO(peterwaller-arm): fortran codegen
  //case InitOnly:               return std::make_unique<InitOnlyAction>();
  //case ParseSyntaxOnly:        return std::make_unique<SyntaxOnlyAction>();
  //case PrintPreprocessedInput: return std::make_unique<PrintPreprocessedAction>();
  //case RunPreprocessorOnly:    return std::make_unique<PreprocessOnlyAction>();
  }

  CI.getDiagnostics().Report(diag::err_fe_action_not_available) <<
    getActionKindName(AK);
  return 0;
}


std::unique_ptr<FrontendAction>
CreateFrontendAction(CompilerInstance &CI) {
  // Create the underlying action.
  std::unique_ptr<FrontendAction> Act = CreateFrontendBaseAction(CI);
  if (!Act)
    return nullptr;

  return Act;
}


bool ExecuteCompilerInvocation(CompilerInstance *Clang) {
  // Honor -help.
  if (Clang->getFrontendOpts().ShowHelp) {
    driver::getDriverOptTable().PrintHelp(
        llvm::outs(), "clang -cc1 [options] file...",
        "LLVM 'Clang' Compiler: http://clang.llvm.org",
        /*Include=*/driver::options::CC1Option,
        /*Exclude=*/0, /*ShowAllAliases=*/false);
    return true;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Clang->getFrontendOpts().ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }


  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  // This should happen AFTER plugins have been loaded!
  if (!Clang->getFrontendOpts().LLVMArgs.empty()) {
    unsigned NumArgs = Clang->getFrontendOpts().LLVMArgs.size();
    auto Args = std::make_unique<const char*[]>(NumArgs + 2);
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Clang->getFrontendOpts().LLVMArgs[i].c_str();
    Args[NumArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args.get());
  }

  // If there were errors in processing arguments, don't do anything else.
  if (Clang->getDiagnostics().hasErrorOccurred())
    return false;

  // Create and execute the frontend action.
  std::unique_ptr<FrontendAction> Act(CreateFrontendAction(*Clang));
  if (!Act)
    return false;
  bool Success = Clang->ExecuteAction(*Act);
  if (Clang->getFrontendOpts().DisableFree)
    llvm::BuryPointer(std::move(Act));

  return Success;
}


} // namespace clang
