//===- FrontendOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H

#include "flang/Frontend/CommandLineSourceLoc.h"

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {

class MemoryBuffer;

} // namespace llvm

namespace clang {

namespace frontend {

enum ActionKind {
  /// Parse ASTs and print them.
  ASTPrint,

  /// Dump the compiler configuration.
  DumpCompilerOptions,

  /// Emit a .s file.
  EmitAssembly,

  /// Emit a .bc file.
  EmitBC,

  /// Translate input source into HTML.
  EmitHTML,

  /// Emit a .ll file.
  EmitLLVM,

  /// Generate LLVM IR, but do not emit anything.
  EmitLLVMOnly,

  /// Generate machine code, but don't emit anything.
  EmitCodeGenOnly,

  /// Emit a .o file.
  EmitObj,

  /// Only execute frontend initialization.
  InitOnly,

  /// Parse and perform semantic analysis.
  ParseSyntaxOnly,

  /// -E mode.
  PrintPreprocessedInput,

  /// Just lex, no output.
  RunPreprocessorOnly
};

inline const char* getActionKindName(const ActionKind ak) {
  switch (ak) {
  case ASTPrint: return "ASTPrint";
  case DumpCompilerOptions: return "DumpCompilerOptions";
  case EmitAssembly: return "EmitAssembly";
  case EmitBC: return "EmitBC";
  case EmitHTML: return "EmitHTML";
  case EmitLLVM: return "EmitLLVM";
  case EmitLLVMOnly: return "EmitLLVMOnly";
  case EmitCodeGenOnly: return "EmitCodeGenOnly";
  case EmitObj: return "EmitObj";
  case InitOnly: return "InitOnly";
  case ParseSyntaxOnly: return "ParseSyntaxOnly";
  case PrintPreprocessedInput: return "PrintPreprocessedInput";
  case RunPreprocessorOnly: return "RunPreprocessorOnly";
  default:
    return "<unknown ActionKind>";
  }
}

} // namespace frontend

enum class Language : uint8_t {
  Unknown,

  /// LLVM IR: we accept this so that we can run the optimizer on it,
  /// and compile it to assembly or object code.
  LLVM_IR,

  ///@{ Languages that the frontend can parse and compile.
  Fortran,
  ///@}
};

/// The kind of a file that we've been handed as an input.
class InputKind {
private:
  Language Lang;
  unsigned Fmt : 3;
  unsigned Preprocessed : 1;

public:
  /// The input file format.
  enum Format {
    Source,
    ModuleMap,
    Precompiled
  };

  constexpr InputKind(Language L = Language::Unknown, Format F = Source,
                      bool PP = false)
      : Lang(L), Fmt(F), Preprocessed(PP) {}

  Language getLanguage() const { return static_cast<Language>(Lang); }
  Format getFormat() const { return static_cast<Format>(Fmt); }
  bool isPreprocessed() const { return Preprocessed; }

  /// Is the input kind fully-unknown?
  bool isUnknown() const { return Lang == Language::Unknown && Fmt == Source; }

  InputKind getPreprocessed() const {
    return InputKind(getLanguage(), getFormat(), true);
  }

  InputKind withFormat(Format F) const {
    return InputKind(getLanguage(), F, isPreprocessed());
  }
};

/// An input file for the front end.
class FrontendInputFile {
  /// The file name, or "-" to read from standard input.
  std::string File;

  /// The input, if it comes from a buffer rather than a file. This object
  /// does not own the buffer, and the caller is responsible for ensuring
  /// that it outlives any users.
  const llvm::MemoryBuffer *Buffer = nullptr;

  /// Whether we're dealing with a 'system' input (vs. a 'user' input).
  bool IsSystem = false;

public:
  FrontendInputFile() = default;
  FrontendInputFile(StringRef File, bool IsSystem = false)
      : File(File.str()), IsSystem(IsSystem) {}
  FrontendInputFile(const llvm::MemoryBuffer *Buffer,
                    bool IsSystem = false)
      : Buffer(Buffer), IsSystem(IsSystem) {}

  bool isSystem() const { return IsSystem; }

  bool isEmpty() const { return File.empty() && Buffer == nullptr; }
  bool isFile() const { return !isBuffer(); }
  bool isBuffer() const { return Buffer != nullptr; }

  StringRef getFile() const {
    assert(isFile());
    return File;
  }

  const llvm::MemoryBuffer *getBuffer() const {
    assert(isBuffer());
    return Buffer;
  }
};

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  /// Disable memory freeing on exit.
  unsigned DisableFree : 1;

  /// Show the -help text.
  unsigned ShowHelp : 1;

  /// Show the -version text.
  unsigned ShowVersion : 1;

  /// print the supported cpus for the current target
  unsigned PrintSupportedCPUs : 1;

  /// The frontend action to perform.
  frontend::ActionKind ProgramAction = frontend::ParseSyntaxOnly;

  /// A list of arguments to forward to LLVM's option processing; this
  /// should only be used for debugging and experimental features.
  std::vector<std::string> LLVMArgs;

  /// The input files and their types.
  std::vector<FrontendInputFile> Inputs;

  /// The output file, if any.
  std::string OutputFile;

public:
  FrontendOptions()
      : DisableFree(false), ShowHelp(false), ShowVersion(false) {}
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
