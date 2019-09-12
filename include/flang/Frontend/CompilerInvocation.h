//===- CompilerInvocation.h - Compiler Invocation Helper Data ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H
#define LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H

#include "flang/Frontend/FrontendOptions.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <string>

namespace llvm {

class Triple;

namespace opt {

class ArgList;

} // namespace opt

namespace vfs {

class FileSystem;

} // namespace vfs

} // namespace llvm

namespace clang {

class DiagnosticsEngine;
class TargetOptions;

/// Fill out Opts based on the options given in Args.
///
/// Args must have been created from the OptTable returned by
/// createCC1OptTable().
///
/// When errors are encountered, return false and, if Diags is non-null,
/// report the error(s).
bool ParseDiagnosticArgs(DiagnosticOptions &Opts, llvm::opt::ArgList &Args,
                         DiagnosticsEngine *Diags = nullptr,
                         bool DefaultDiagColor = true,
                         bool DefaultShowOpt = true);

class CompilerInvocationBase {
public:
  /// Options controlling the target.
  std::shared_ptr<TargetOptions> TargetOpts;

  /// Options controlling the diagnostic engine.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagnosticOpts;

  CompilerInvocationBase();
  CompilerInvocationBase(const CompilerInvocationBase &X);
  CompilerInvocationBase &operator=(const CompilerInvocationBase &) = delete;
  ~CompilerInvocationBase();

  DiagnosticOptions &getDiagnosticOpts() { return *DiagnosticOpts.get(); }
  const DiagnosticOptions &getDiagnosticOpts() const { return *DiagnosticOpts.get(); }

  TargetOptions &getTargetOpts() { return *TargetOpts.get(); }
  const TargetOptions &getTargetOpts() const { return *TargetOpts.get(); }
};

/// Helper class for holding the data necessary to invoke the compiler.
///
/// This class is designed to represent an abstract "invocation" of the
/// compiler, including data such as the include paths, the code generation
/// options, the warning flags, and so on.
class CompilerInvocation : public CompilerInvocationBase {
  /// Options controlling file system operations.
  FileSystemOptions FileSystemOpts;

  /// Options controlling the frontend itself.
  FrontendOptions FrontendOpts;

public:
  CompilerInvocation() {}

  /// @name Utility Methods
  /// @{

  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  ///
  /// \returns false if an error was encountered while parsing the arguments
  /// and attempts to recover and continue parsing the rest of the arguments.
  /// The recovery is best-effort and only guarantees that \p Res will end up in
  /// one of the vaild-to-access (albeit arbitrary) states.
  ///
  /// \param [out] Res - The resulting invocation.
  static bool CreateFromArgs(CompilerInvocation &Res,
                             ArrayRef<const char *> CommandLineArgs,
                             DiagnosticsEngine &Diags);

  /// Get the directory where the compiler headers
  /// reside, relative to the compiler binary (found by the passed in
  /// arguments).
  ///
  /// \param Argv0 - The program path (from argv[0]), for finding the builtin
  /// compiler path.
  /// \param MainAddr - The address of main (or some other function in the main
  /// executable), for finding the builtin compiler path.
  static std::string GetResourcesPath(const char *Argv0, void *MainAddr);

  /// @}
  /// @name Option Subgroups
  /// @{

  FileSystemOptions &getFileSystemOpts() { return FileSystemOpts; }

  const FileSystemOptions &getFileSystemOpts() const {
    return FileSystemOpts;
  }

  FrontendOptions &getFrontendOpts() { return FrontendOpts; }
  const FrontendOptions &getFrontendOpts() const { return FrontendOpts; }

  /// @}
};

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromCompilerInvocation(const CompilerInvocation &CI,
                                DiagnosticsEngine &Diags);

IntrusiveRefCntPtr<llvm::vfs::FileSystem> createVFSFromCompilerInvocation(
    const CompilerInvocation &CI, DiagnosticsEngine &Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS);

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H
