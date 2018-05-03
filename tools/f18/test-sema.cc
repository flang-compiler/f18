// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../../lib/parser/parsing.h"
#include "../../lib/semantics/attr.h"
#include "../../lib/semantics/type.h"

#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <stddef.h>
#include <string>

#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/ADT/SmallString.h>

#define TEST_FORTRAN_TARGET_INFO 0

#if TEST_FORTRAN_TARGET_INFO
#include "../../lib/basic/TargetInfo.h"
// and include some cpp files that are not yet handled by cmake
#include "../../lib/basic/TargetInfo.cc"
#include "../../lib/basic/targets/TargetInfoFromClang.cc"
#endif

using namespace Fortran;
using namespace parser;

extern void DoSemanticAnalysis(const CookedSource &, const Program &);

//static void visitProgramUnit(const ProgramUnit &unit);

// Catch the diagnostics emited while building the clang::TargetInfo.
// They should eventually be forwarded to our own DiagnosticConsumer.
class MyDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  virtual void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                               const clang::Diagnostic &info) {
    clang::SmallString<256> out;
    info.FormatDiagnostic(out);
    std::cerr << "ERROR: "  << out.c_str() << "\n";
  }
};

static const char * float_name(const llvm::fltSemantics *sem) {
  if ( !sem ) {
    return "NULL";
  } else if ( sem == &llvm::APFloat::IEEEhalf() ) {
    return "IEEEhalf";
  } else if ( sem == &llvm::APFloat::IEEEsingle() ) {
    return "IEEEsingle";
  } else if ( sem == &llvm::APFloat::IEEEdouble() ) {
    return "IEEEdouble";
  } else if ( sem == &llvm::APFloat::IEEEquad() ) {
    return "IEEEquad";
  } else if ( sem == &llvm::APFloat::PPCDoubleDouble() ) {
    return "PPCDoubleDouble";
  } else if ( sem == &llvm::APFloat::x87DoubleExtended() ) {
    return "x87DoubleExtended";
  } else {
    return "Unknown";
  } 
}

void test_clang_target()
{
  std::string tp = llvm::sys::getDefaultTargetTriple();
  if (getenv("TARGET")) {
    tp=getenv("TARGET");
  }
  std::cout << "Using target " << tp << "\n";
  llvm::Triple triple(tp);
  std::cout << " Arch = " << triple.getArchName().str() << " ("<< triple.getArch() << ")\n";
  std::cout << " Vendor = " << triple.getVendorName().str() << " (" << triple.getVendor()<< ")\n";
  std::cout << " OS = " << triple.getOSName().str() << " (" << triple.getOS()<< ")\n";
  std::string err;

#if 0
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(tp,err);
  std::cout << target << " err=" << err << "\n";
#endif
  
  //////////// Clang Target::Info

  // code taken from https://github.com/loarabia/Clang-tutorial/blob/master/tutorial4.cpp
  //
  // See also
  //    https://github.com/llvm-mirror/clang/blob/master/include/clang/Basic/TargetInfo.h
  
  clang::DiagnosticOptions diagnosticOptions;

  clang::DiagnosticConsumer *pTextDiagnosticPrinter = new MyDiagnosticConsumer();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> pDiagIDs;

  clang::DiagnosticsEngine *pDiagnosticsEngine =
    new clang::DiagnosticsEngine(pDiagIDs, 
                                 &diagnosticOptions,                                 
                                 pTextDiagnosticPrinter);
  
  const std::shared_ptr<clang::TargetOptions> targetOptions
    = std::make_shared<clang::TargetOptions>();

  targetOptions->Triple = tp;
  
  clang::TargetInfo * targetInfo =
    clang::TargetInfo::CreateTargetInfo( *pDiagnosticsEngine,
                                         targetOptions) ;

  if (targetInfo) {

#define DUMP(x) std::cout << #x << " = " << targetInfo->x << "\n" 
#define DUMP_V(name,v) std::cout << name << " = " << v << "\n" 
#define DUMP_INT_TYPE(t) DUMP(getTypeWidth(t)) ; DUMP(getTypeAlign(t))
    auto ptrdiff_type = targetInfo->getPtrDiffType(0);    
    auto intptr_type = targetInfo->getIntPtrType();
    DUMP_INT_TYPE(clang::TargetInfo::SignedChar) ; 
    DUMP_INT_TYPE(clang::TargetInfo::SignedShort) ; 
    DUMP_INT_TYPE(clang::TargetInfo::SignedInt) ; 
    DUMP_INT_TYPE(clang::TargetInfo::SignedLong) ; 
    DUMP_INT_TYPE(clang::TargetInfo::SignedLongLong) ; 
    DUMP_INT_TYPE(ptrdiff_type) ; 
    DUMP_INT_TYPE(intptr_type) ;
    DUMP(hasInt128Type());
    DUMP(hasFloat128Type());

    //    DUMP(hasLegalHalfType());
    DUMP(getHalfWidth());
    DUMP(getHalfAlign());
    DUMP_V("geHalfFormat()",float_name( &targetInfo->getHalfFormat()) );

    DUMP(getFloatWidth());
    DUMP(getFloatAlign());
    DUMP_V("geFloatFormat()",float_name( &targetInfo->getFloatFormat()) );

    DUMP(getDoubleWidth());
    DUMP(getDoubleAlign());
    DUMP_V("getDoubleFormat()",float_name( &targetInfo->getDoubleFormat()) );

    DUMP(getLongDoubleWidth());
    DUMP(getLongDoubleAlign());
    DUMP_V("getLongDoubleFormat()",float_name( &targetInfo->getLongDoubleFormat()) );

    DUMP(getFloat128Width());
    DUMP(getFloat128Align());
    DUMP_V("getFloat128Format()",float_name( &targetInfo->getFloat128Format()) );

#undef DUMP
#undef DUMP_INT_TYPE
  } else {
    std::cerr << "No clang::TargetInfo for " << tp << "\n"; 
  }

  
#if TEST_FORTRAN_TARGET_INFO
  Fortran::TargetInfo * ftarget = Fortran::TargetInfo::Create(triple);
  if (ftarget) {
    ftarget->dump(std::cout);
  } else {
    std::cerr << "No Fortran::TargetInfo for " << tp << "\n"; 
  }
#endif


}

int main(int argc, char *const argv[]) {
  test_clang_target() ;
  
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  Parsing parsing;
  if (parsing.ForTesting(path, std::cerr)) {
    DoSemanticAnalysis(parsing.cooked(), *parsing.parseTree());
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
