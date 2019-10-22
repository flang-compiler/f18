//===- Utils.h - Misc utilities for the front-end ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_UTILS_H
#define LLVM_CLANG_FRONTEND_UTILS_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace llvm {

class Triple;

namespace opt {

class ArgList;

} // namespace opt

} // namespace llvm

namespace clang {

class ASTReader;
class CompilerInstance;
class CompilerInvocation;
class DiagnosticsEngine;
class ExternalSemaSource;
class FrontendOptions;
class HeaderSearch;
class HeaderSearchOptions;
class LangOptions;
class PCHContainerReader;
class Preprocessor;
class PreprocessorOptions;
class PreprocessorOutputOptions;

/// Apply the header search options to get given HeaderSearch object.
void ApplyHeaderSearchOptions(HeaderSearch &HS,
                              const HeaderSearchOptions &HSOpts,
                              const LangOptions &Lang,
                              const llvm::Triple &triple);

/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file.
void InitializePreprocessor(Preprocessor &PP, const PreprocessorOptions &PPOpts,
                            const PCHContainerReader &PCHContainerRdr,
                            const FrontendOptions &FEOpts);

/// DoPrintPreprocessedInput - Implement -E mode.
void DoPrintPreprocessedInput(Preprocessor &PP, raw_ostream *OS,
                              const PreprocessorOutputOptions &Opts);

/// The ChainedIncludesSource class converts headers to chained PCHs in
/// memory, mainly for testing.
IntrusiveRefCntPtr<ExternalSemaSource>
createChainedIncludesSource(CompilerInstance &CI,
                            IntrusiveRefCntPtr<ExternalSemaSource> &Reader);

/// createInvocationFromCommandLine - Construct a compiler invocation object for
/// a command line argument vector.
///
/// \param ShouldRecoverOnErrors - whether we should attempt to return a
/// non-null (and possibly incorrect) CompilerInvocation if any errors were
/// encountered. When this flag is false, always return null on errors.
///
/// \return A CompilerInvocation, or 0 if none was built for the given
/// argument vector.
std::unique_ptr<CompilerInvocation> createInvocationFromCommandLine(
    ArrayRef<const char *> Args,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        IntrusiveRefCntPtr<DiagnosticsEngine>(),
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr,
    bool ShouldRecoverOnErrors = false);

/// Return the value of the last argument as an integer, or a default. If Diags
/// is non-null, emits an error if the argument is given, but non-integral.
int getLastArgIntValue(const llvm::opt::ArgList &Args,
                       llvm::opt::OptSpecifier Id, int Default,
                       DiagnosticsEngine *Diags = nullptr);

inline int getLastArgIntValue(const llvm::opt::ArgList &Args,
                              llvm::opt::OptSpecifier Id, int Default,
                              DiagnosticsEngine &Diags) {
  return getLastArgIntValue(Args, Id, Default, &Diags);
}

uint64_t getLastArgUInt64Value(const llvm::opt::ArgList &Args,
                               llvm::opt::OptSpecifier Id, uint64_t Default,
                               DiagnosticsEngine *Diags = nullptr);

inline uint64_t getLastArgUInt64Value(const llvm::opt::ArgList &Args,
                                      llvm::opt::OptSpecifier Id,
                                      uint64_t Default,
                                      DiagnosticsEngine &Diags) {
  return getLastArgUInt64Value(Args, Id, Default, &Diags);
}

// Frontend timing utils

/// If the user specifies the -ftime-report argument on an Clang command line
/// then the value of this boolean will be true, otherwise false.
extern bool FrontendTimesIsEnabled;

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_UTILS_H
