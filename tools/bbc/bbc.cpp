//===- bbc.cpp - Burnside Bridge Compiler -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a tool for translating Fortran sources to the FIR dialect of MLIR.
///
//===----------------------------------------------------------------------===//

#include "../../lib/common/Fortran-features.h"
#include "../../lib/common/default-kinds.h"
#include "../../lib/lower/bridge.h"
#include "../../lib/lower/convert-expr.h"
#include "../../lib/parser/characters.h"
#include "../../lib/parser/dump-parse-tree.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/parsing.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/unparse.h"
#include "../../lib/semantics/expression.h"
#include "../../lib/semantics/semantics.h"
#include "../../lib/semantics/unparse-with-symbols.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "optimizer/CodeGen/CodeGen.h"
#include "optimizer/FIRDialect.h"
#include "optimizer/InternalNames.h"
#include "optimizer/KindMapping.h"
#include "optimizer/Transforms/Passes.h"
#include "optimizer/Transforms/StdConverter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

namespace Br = Fortran::lower;

using namespace llvm;

namespace {

// Some basic command-line options
cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                   cl::desc("<input file>"));

cl::opt<std::string> OutputFilename("o",
                                    cl::desc("Specify the output filename"),
                                    cl::value_desc("filename"),
                                    cl::init("a.mlir"));

cl::list<std::string> IncludeDirs("I", cl::desc("include search paths"));

cl::list<std::string> ModuleDirs("module", cl::desc("module search paths"));

cl::opt<std::string> ModuleSuffix("module-suffix",
                                  cl::desc("module file suffix override"),
                                  cl::init(".mod"));

cl::opt<bool> EmitLLVM("emit-llvm", cl::desc("emit LLVM IR"), cl::init(false));

// vestigal struct that should be deleted
struct DriverOptions {
  bool forcedForm{false};             // -Mfixed or -Mfree appeared
  bool warnOnNonstandardUsage{false}; // -Mstandard
  bool warningsAreErrors{false};      // -Werror
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::UTF_8};
  std::string prefix;
};

int exitStatus{EXIT_SUCCESS};

// Convert Fortran input to MLIR (target is FIR dialect)
void convertFortranSourceToMLIR(
    std::string path, Fortran::parser::Options options, DriverOptions &driver,
    Fortran::semantics::SemanticsContext &semanticsContext) {
  if (!driver.forcedForm) {
    auto dot{path.rfind(".")};
    if (dot != std::string::npos) {
      std::string suffix{path.substr(dot + 1)};
      options.isFixedForm = suffix == "f" || suffix == "F" || suffix == "ff";
    }
  }
  options.searchDirectories = IncludeDirs;
  Fortran::parser::Parsing parsing{semanticsContext.allSources()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() &&
      (driver.warningsAreErrors || parsing.messages().AnyFatalError())) {
    errs() << driver.prefix << "could not scan " << path << '\n';
    parsing.messages().Emit(std::cerr, parsing.cooked());
    exitStatus = EXIT_FAILURE;
    return;
  }
  parsing.Parse(&std::cout);
  parsing.messages().Emit(std::cerr, parsing.cooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(std::cerr, parsing.finalRestingPlace(),
                        "parser FAIL (final position)");
    exitStatus = EXIT_FAILURE;
    return;
  }
  if ((!parsing.messages().empty() &&
       (driver.warningsAreErrors || parsing.messages().AnyFatalError())) ||
      !parsing.parseTree().has_value()) {
    errs() << driver.prefix << "could not parse " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return;
  }
  auto &parseTree{*parsing.parseTree()};
  Fortran::semantics::Semantics semantics{semanticsContext, parseTree,
                                          parsing.cooked()};
  semantics.Perform();
  semantics.EmitMessages(std::cerr);
  if (semantics.AnyFatalError()) {
    errs() << driver.prefix << "semantic errors in " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return;
  }

  // MLIR+FIR
  fir::NameUniquer nameUniquer;
  auto burnside = Br::BurnsideBridge::create(semanticsContext.defaultKinds(),
                                             &parsing.cooked());
  fir::KindMapping kindMap{&burnside.getMLIRContext()};
  burnside.lower(parseTree, nameUniquer);
  mlir::ModuleOp mlirModule = burnside.getModule();
  mlir::PassManager pm = mlirModule.getContext();

  pm.addPass(fir::createMemToRegPass());
  pm.addPass(fir::createCSEPass());
  pm.addPass(fir::createLowerToLoopPass());
  pm.addPass(fir::createFIRToStdPass(kindMap));
  pm.addPass(mlir::createLowerToCFGPass());

  if (EmitLLVM) {
    pm.addPass(fir::createFIRToLLVMPass(nameUniquer));
    pm.addPass(fir::createLLVMDialectToLLVMPass(OutputFilename + ".ll"));
  }

  if (mlir::succeeded(pm.run(mlirModule))) {
    std::error_code ec;
    raw_fd_ostream out(OutputFilename, ec);
    if (!ec)
      mlirModule.print(out);
  } else {
    errs() << "oops, pass manager reported failure\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  [[maybe_unused]] InitLLVM y(argc, argv);

  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "bbc\n");

  DriverOptions driver;
  driver.prefix = argv[0] + ": "s;

  if (IncludeDirs.size() == 0) {
    IncludeDirs.push_back(".");
  }
  if (ModuleDirs.size() == 0) {
    ModuleDirs.push_back(".");
  }

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__F18", "1");
  options.predefinitions.emplace_back("__F18_MAJOR__", "1");
  options.predefinitions.emplace_back("__F18_MINOR__", "1");
  options.predefinitions.emplace_back("__F18_PATCHLEVEL__", "1");
#if __x86_64__
  options.predefinitions.emplace_back("__x86_64__", "1");
#endif

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;
  Fortran::parser::AllSources allSources;
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, options.features, allSources};
  semanticsContext.set_moduleDirectory(ModuleDirs.front())
      .set_moduleFileSuffix(ModuleSuffix)
      .set_searchDirectories(IncludeDirs)
      .set_warnOnNonstandardUsage(driver.warnOnNonstandardUsage)
      .set_warningsAreErrors(driver.warningsAreErrors);

  convertFortranSourceToMLIR(InputFilename, options, driver, semanticsContext);
  return exitStatus;
}
