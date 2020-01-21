//===- tco.cpp - Tilikum Crossing Opt ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is to be like LLVM's opt program, only for FIR.  Such a program is
// required for roundtrip testing, etc.
//
//===----------------------------------------------------------------------===//

#include "fir/Dialect/FIRDialect.h"
#include "flang/optimizer/CodeGen/CodeGen.h"
#include "flang/optimizer/InternalNames.h"
#include "flang/optimizer/KindMapping.h"
#include "flang/optimizer/Transforms/Passes.h"
#include "flang/optimizer/Transforms/StdConverter.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

namespace {

namespace Cl = llvm::cl;

Cl::opt<std::string> inputFilename(Cl::Positional, Cl::desc("<input file>"),
                                   Cl::init("-"));

Cl::opt<std::string> outputFilename("o", Cl::desc("Specify output filename"),
                                    Cl::value_desc("filename"), Cl::init("-"));

Cl::opt<bool> emitFir("emit-fir", Cl::desc("Parse and pretty-print the input"),
                      Cl::init(false));

void printModuleBody(mlir::ModuleOp mod) {
  // don't output the terminator bogo-op
  auto e{--mod.end()};
  for (auto i{mod.begin()}; i != e; ++i) {
    i->print(llvm::outs());
    llvm::outs() << '\n';
  }
}

// compile a .fir file
int compileFIR() {
  // check that there is a file to load
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open file: " << EC.message() << '\n';
    return 1;
  }

  // load the file into a module
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto context = std::make_unique<mlir::MLIRContext>();
  auto owningRef = mlir::parseSourceFile(sourceMgr, context.get());

  if (!owningRef) {
    llvm::errs() << "Error can't load file " << inputFilename << '\n';
    return 2;
  }
  if (mlir::failed(owningRef->verify())) {
    llvm::errs() << "Error verifying FIR module\n";
    return 4;
  }

  std::error_code ec;
  llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);

  // run passes
  fir::NameUniquer uniquer;
  fir::KindMapping kindMap{context.get()};
  mlir::PassManager pm{context.get()};
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
  } else {
    pm.addPass(fir::createMemToRegPass());
    pm.addPass(fir::createCSEPass());
    // convert fir dialect to affine
    pm.addPass(fir::createPromoteToAffinePass());
    // convert fir dialect to loop
    pm.addPass(fir::createLowerToLoopPass());
    pm.addPass(fir::createFIRToStdPass(kindMap));
    // convert loop dialect to standard
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(fir::createFIRToLLVMPass(uniquer));
    pm.addPass(fir::createLLVMDialectToLLVMPass(out.os()));
  }
  if (mlir::succeeded(pm.run(*owningRef))) {
    if (emitFir)
      printModuleBody(*owningRef);
    return 0;
  }
  llvm::errs() << "FAILED: " << inputFilename << '\n';
  return 8;
}
} // namespace

int main(int argc, char **argv) {
  [[maybe_unused]] llvm::InitLLVM y(argc, argv);
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  Cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Opt\n");
  return compileFIR();
}
