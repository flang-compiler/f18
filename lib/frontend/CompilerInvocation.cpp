//===- CompilerInvocation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/CommandLineSourceLoc.h"
#include "flang/Frontend/FrontendDiagnostic.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/Utils.h"

#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/Visibility.h"
#include "clang/Config/config.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace clang;
using namespace driver;
using namespace options;
using namespace llvm::opt;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

CompilerInvocationBase::CompilerInvocationBase()
    : TargetOpts(new TargetOptions()),
      DiagnosticOpts(new DiagnosticOptions()) {}

CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase &X)
    : TargetOpts(new TargetOptions(X.getTargetOpts())),
      DiagnosticOpts(new DiagnosticOptions(X.getDiagnosticOpts())) {}

CompilerInvocationBase::~CompilerInvocationBase() = default;


static void addDiagnosticArgs(ArgList &Args, OptSpecifier Group,
                              OptSpecifier GroupWithValue,
                              std::vector<std::string> &Diagnostics) {
  for (auto *A : Args.filtered(Group)) {
    if (A->getOption().getKind() == Option::FlagClass) {
      // The argument is a pure flag (such as OPT_Wall or OPT_Wdeprecated). Add
      // its name (minus the "W" or "R" at the beginning) to the warning list.
      Diagnostics.push_back(A->getOption().getName().drop_front(1));
    } else if (A->getOption().matches(GroupWithValue)) {
      // This is -Wfoo= or -Rfoo=, where foo is the name of the diagnostic group.
      Diagnostics.push_back(A->getOption().getName().drop_front(1).rtrim("=-"));
    } else {
      // Otherwise, add its value (for OPT_W_Joined and similar).
      for (const auto *Arg : A->getValues())
        Diagnostics.emplace_back(Arg);
    }
  }
}

static bool parseShowColorsArgs(const ArgList &Args, bool DefaultColor) {
  // Color diagnostics default to auto ("on" if terminal supports) in the driver
  // but default to off in cc1, needing an explicit OPT_fdiagnostics_color.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum {
    Colors_On,
    Colors_Off,
    Colors_Auto
  } ShowColors = DefaultColor ? Colors_Auto : Colors_Off;
  for (auto *A : Args) {
    const Option &O = A->getOption();
    if (O.matches(options::OPT_fcolor_diagnostics) ||
        O.matches(options::OPT_fdiagnostics_color)) {
      ShowColors = Colors_On;
    } else if (O.matches(options::OPT_fno_color_diagnostics) ||
               O.matches(options::OPT_fno_diagnostics_color)) {
      ShowColors = Colors_Off;
    } else if (O.matches(options::OPT_fdiagnostics_color_EQ)) {
      StringRef Value(A->getValue());
      if (Value == "always")
        ShowColors = Colors_On;
      else if (Value == "never")
        ShowColors = Colors_Off;
      else if (Value == "auto")
        ShowColors = Colors_Auto;
    }
  }
  return ShowColors == Colors_On ||
         (ShowColors == Colors_Auto &&
          llvm::sys::Process::StandardErrHasColors());
}

static bool checkVerifyPrefixes(const std::vector<std::string> &VerifyPrefixes,
                                DiagnosticsEngine *Diags) {
  bool Success = true;
  for (const auto &Prefix : VerifyPrefixes) {
    // Every prefix must start with a letter and contain only alphanumeric
    // characters, hyphens, and underscores.
    auto BadChar = llvm::find_if(Prefix, [](char C) {
      return !isAlphanumeric(C) && C != '-' && C != '_';
    });
    if (BadChar != Prefix.end() || !isLetter(Prefix[0])) {
      Success = false;
      if (Diags) {
        Diags->Report(diag::err_drv_invalid_value) << "-verify=" << Prefix;
        Diags->Report(diag::note_drv_verify_prefix_spelling);
      }
    }
  }
  return Success;
}

bool clang::ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args,
                                DiagnosticsEngine *Diags,
                                bool DefaultDiagColor, bool DefaultShowOpt) {
  bool Success = true;

  Opts.DiagnosticLogFile = Args.getLastArgValue(OPT_diagnostic_log_file);
  if (Arg *A =
          Args.getLastArg(OPT_diagnostic_serialized_file, OPT__serialize_diags))
    Opts.DiagnosticSerializationFile = A->getValue();
  Opts.IgnoreWarnings = Args.hasArg(OPT_w);
  Opts.NoRewriteMacros = Args.hasArg(OPT_Wno_rewrite_macros);
  Opts.Pedantic = Args.hasArg(OPT_pedantic);
  Opts.PedanticErrors = Args.hasArg(OPT_pedantic_errors);
  Opts.ShowCarets = !Args.hasArg(OPT_fno_caret_diagnostics);
  Opts.ShowColors = parseShowColorsArgs(Args, DefaultDiagColor);
  Opts.ShowColumn = Args.hasFlag(OPT_fshow_column,
                                 OPT_fno_show_column,
                                 /*Default=*/true);
  Opts.ShowFixits = !Args.hasArg(OPT_fno_diagnostics_fixit_info);
  Opts.ShowLocation = !Args.hasArg(OPT_fno_show_source_location);
  Opts.AbsolutePath = Args.hasArg(OPT_fdiagnostics_absolute_paths);
  Opts.ShowOptionNames =
      Args.hasFlag(OPT_fdiagnostics_show_option,
                   OPT_fno_diagnostics_show_option, DefaultShowOpt);

  llvm::sys::Process::UseANSIEscapeCodes(Args.hasArg(OPT_fansi_escape_codes));

  // Default behavior is to not to show note include stacks.
  Opts.ShowNoteIncludeStack = false;
  if (Arg *A = Args.getLastArg(OPT_fdiagnostics_show_note_include_stack,
                               OPT_fno_diagnostics_show_note_include_stack))
    if (A->getOption().matches(OPT_fdiagnostics_show_note_include_stack))
      Opts.ShowNoteIncludeStack = true;

  StringRef ShowOverloads =
    Args.getLastArgValue(OPT_fshow_overloads_EQ, "all");
  if (ShowOverloads == "best")
    Opts.setShowOverloads(Ovl_Best);
  else if (ShowOverloads == "all")
    Opts.setShowOverloads(Ovl_All);
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fshow_overloads_EQ)->getAsString(Args)
      << ShowOverloads;
  }

  StringRef ShowCategory =
    Args.getLastArgValue(OPT_fdiagnostics_show_category, "none");
  if (ShowCategory == "none")
    Opts.ShowCategories = 0;
  else if (ShowCategory == "id")
    Opts.ShowCategories = 1;
  else if (ShowCategory == "name")
    Opts.ShowCategories = 2;
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_show_category)->getAsString(Args)
      << ShowCategory;
  }

  StringRef Format =
    Args.getLastArgValue(OPT_fdiagnostics_format, "clang");
  if (Format == "clang")
    Opts.setFormat(DiagnosticOptions::Clang);
  else if (Format == "msvc")
    Opts.setFormat(DiagnosticOptions::MSVC);
  else if (Format == "msvc-fallback") {
    Opts.setFormat(DiagnosticOptions::MSVC);
    Opts.CLFallbackMode = true;
  } else if (Format == "vi")
    Opts.setFormat(DiagnosticOptions::Vi);
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_format)->getAsString(Args)
      << Format;
  }

  Opts.ShowSourceRanges = Args.hasArg(OPT_fdiagnostics_print_source_range_info);
  Opts.ShowParseableFixits = Args.hasArg(OPT_fdiagnostics_parseable_fixits);
  Opts.ShowPresumedLoc = !Args.hasArg(OPT_fno_diagnostics_use_presumed_location);
  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify) || Args.hasArg(OPT_verify_EQ);
  Opts.VerifyPrefixes = Args.getAllArgValues(OPT_verify_EQ);
  if (Args.hasArg(OPT_verify))
    Opts.VerifyPrefixes.push_back("expected");
  // Keep VerifyPrefixes in its original order for the sake of diagnostics, and
  // then sort it to prepare for fast lookup using std::binary_search.
  if (!checkVerifyPrefixes(Opts.VerifyPrefixes, Diags)) {
    Opts.VerifyDiagnostics = false;
    Success = false;
  }
  else
    llvm::sort(Opts.VerifyPrefixes);
  DiagnosticLevelMask DiagMask = DiagnosticLevelMask::None;

  if (Args.hasArg(OPT_verify_ignore_unexpected))
    DiagMask = DiagnosticLevelMask::All;
  Opts.setVerifyIgnoreUnexpected(DiagMask);
  Opts.ElideType = !Args.hasArg(OPT_fno_elide_type);
  Opts.ShowTemplateTree = Args.hasArg(OPT_fdiagnostics_show_template_tree);
  Opts.ErrorLimit = getLastArgIntValue(Args, OPT_ferror_limit, 0, Diags);
  Opts.MacroBacktraceLimit =
      getLastArgIntValue(Args, OPT_fmacro_backtrace_limit,
                         DiagnosticOptions::DefaultMacroBacktraceLimit, Diags);
  Opts.TemplateBacktraceLimit = getLastArgIntValue(
      Args, OPT_ftemplate_backtrace_limit,
      DiagnosticOptions::DefaultTemplateBacktraceLimit, Diags);
  Opts.ConstexprBacktraceLimit = getLastArgIntValue(
      Args, OPT_fconstexpr_backtrace_limit,
      DiagnosticOptions::DefaultConstexprBacktraceLimit, Diags);
  Opts.SpellCheckingLimit = getLastArgIntValue(
      Args, OPT_fspell_checking_limit,
      DiagnosticOptions::DefaultSpellCheckingLimit, Diags);
  Opts.SnippetLineLimit = getLastArgIntValue(
      Args, OPT_fcaret_diagnostics_max_lines,
      DiagnosticOptions::DefaultSnippetLineLimit, Diags);
  Opts.TabStop = getLastArgIntValue(Args, OPT_ftabstop,
                                    DiagnosticOptions::DefaultTabStop, Diags);
  if (Opts.TabStop == 0 || Opts.TabStop > DiagnosticOptions::MaxTabStop) {
    Opts.TabStop = DiagnosticOptions::DefaultTabStop;
    if (Diags)
      Diags->Report(diag::warn_ignoring_ftabstop_value)
      << Opts.TabStop << DiagnosticOptions::DefaultTabStop;
  }
  Opts.MessageLength = getLastArgIntValue(Args, OPT_fmessage_length, 0, Diags);
  addDiagnosticArgs(Args, OPT_W_Group, OPT_W_value_Group, Opts.Warnings);
  addDiagnosticArgs(Args, OPT_R_Group, OPT_R_value_Group, Opts.Remarks);

  return Success;
}

static void ParseFileSystemArgs(FileSystemOptions &Opts, ArgList &Args) {
  Opts.WorkingDir = Args.getLastArgValue(OPT_working_directory);
}

static InputKind ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args,
                                   DiagnosticsEngine &Diags) {
  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("Invalid option in group!");
    case OPT_ast_print:
      Opts.ProgramAction = frontend::ASTPrint; break;
    case OPT_compiler_options_dump:
      Opts.ProgramAction = frontend::DumpCompilerOptions; break;
    case OPT_S:
      Opts.ProgramAction = frontend::EmitAssembly; break;
    case OPT_emit_llvm_bc:
      Opts.ProgramAction = frontend::EmitBC; break;
    case OPT_emit_html:
      Opts.ProgramAction = frontend::EmitHTML; break;
    case OPT_emit_llvm:
      Opts.ProgramAction = frontend::EmitLLVM; break;
    case OPT_emit_llvm_only:
      Opts.ProgramAction = frontend::EmitLLVMOnly; break;
    case OPT_emit_codegen_only:
      Opts.ProgramAction = frontend::EmitCodeGenOnly; break;
    case OPT_emit_obj:
      Opts.ProgramAction = frontend::EmitObj; break;
    case OPT_init_only:
      Opts.ProgramAction = frontend::InitOnly; break;
    case OPT_fsyntax_only:
      Opts.ProgramAction = frontend::ParseSyntaxOnly; break;
    case OPT_module_file_info:
    case OPT_E:
      Opts.ProgramAction = frontend::PrintPreprocessedInput; break;
    case OPT_Eonly:
      Opts.ProgramAction = frontend::RunPreprocessorOnly; break;
    }
  }

  Opts.DisableFree = Args.hasArg(OPT_disable_free);

  Opts.OutputFile = Args.getLastArgValue(OPT_o);
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowVersion = Args.hasArg(OPT_version);
  Opts.LLVMArgs = Args.getAllArgValues(OPT_mllvm);

  InputKind DashX(Language::Unknown);
  if (const Arg *A = Args.getLastArg(OPT_x)) {
    StringRef XValue = A->getValue();

    // Principal languages.
    DashX = llvm::StringSwitch<InputKind>(XValue)
                .Case("f90", Language::Fortran)
                .Default(Language::Unknown);

    // Some special cases cannot be combined with suffixes.
    if (DashX.isUnknown())
      DashX = llvm::StringSwitch<InputKind>(XValue)
                  .Case("ir", Language::LLVM_IR)
                  .Default(Language::Unknown);

    if (DashX.isUnknown())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
  }

  return DashX;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  std::string ClangExecutable =
      llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  return Driver::GetResourcesPath(ClangExecutable, CLANG_RESOURCE_DIR);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args,
                            DiagnosticsEngine &Diags) {
  Opts.ABI = Args.getLastArgValue(OPT_target_abi);
  if (Arg *A = Args.getLastArg(OPT_meabi)) {
    StringRef Value = A->getValue();
    llvm::EABI EABIVersion = llvm::StringSwitch<llvm::EABI>(Value)
                                 .Case("default", llvm::EABI::Default)
                                 .Case("4", llvm::EABI::EABI4)
                                 .Case("5", llvm::EABI::EABI5)
                                 .Case("gnu", llvm::EABI::GNU)
                                 .Default(llvm::EABI::Unknown);
    if (EABIVersion == llvm::EABI::Unknown)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << Value;
    else
      Opts.EABIVersion = EABIVersion;
  }
  Opts.CPU = Args.getLastArgValue(OPT_target_cpu);
  Opts.FPMath = Args.getLastArgValue(OPT_mfpmath);
  Opts.LinkerVersion = Args.getLastArgValue(OPT_target_linker_version);
  Opts.Triple = Args.getLastArgValue(OPT_triple);
  // Use the default target triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getDefaultTargetTriple();
  Opts.Triple = llvm::Triple::normalize(Opts.Triple);
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        ArrayRef<const char *> CommandLineArgs,
                                        DiagnosticsEngine &Diags) {
  bool Success = true;

  // Parse the arguments.
  const OptTable &Opts = getDriverOptTable();
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = Opts.ParseArgs(CommandLineArgs, MissingArgIndex,
                                     MissingArgCount, IncludedFlagsBitmask);

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (const auto *A : Args.filtered(OPT_UNKNOWN)) {
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (Opts.findNearest(ArgString, Nearest, IncludedFlagsBitmask) > 1)
      Diags.Report(diag::err_drv_unknown_argument) << ArgString;
    else
      Diags.Report(diag::err_drv_unknown_argument_with_suggestion)
          << ArgString << Nearest;
    Success = false;
  }

  Success &=
      ParseDiagnosticArgs(Res.getDiagnosticOpts(), Args, &Diags,
                          false /*DefaultDiagColor*/, false /*DefaultShowOpt*/);

  ParseFileSystemArgs(Res.getFileSystemOpts(), Args);

  InputKind DashX = ParseFrontendArgs(Res.getFrontendOpts(), Args, Diags);
  (void)DashX; // TODO(pwaller).
  ParseTargetArgs(Res.getTargetOpts(), Args, Diags);

  return Success;
}

template<typename IntTy>
static IntTy getLastArgIntValueImpl(const ArgList &Args, OptSpecifier Id,
                                    IntTy Default,
                                    DiagnosticsEngine *Diags) {
  IntTy Res = Default;
  if (Arg *A = Args.getLastArg(Id)) {
    if (StringRef(A->getValue()).getAsInteger(10, Res)) {
      if (Diags)
        Diags->Report(diag::err_drv_invalid_int_value) << A->getAsString(Args)
                                                       << A->getValue();
    }
  }
  return Res;
}

namespace clang {

// Declared in flang/Frontend/Utils.h.
int getLastArgIntValue(const ArgList &Args, OptSpecifier Id, int Default,
                       DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<int>(Args, Id, Default, Diags);
}

uint64_t getLastArgUInt64Value(const ArgList &Args, OptSpecifier Id,
                               uint64_t Default,
                               DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<uint64_t>(Args, Id, Default, Diags);
}

} // namespace clang
