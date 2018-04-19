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

using namespace Fortran;
using namespace parser;

extern void DoSemanticAnalysis(const CookedSource &, const Program &);

class MyDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  virtual void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                               const clang::Diagnostic &info) {
    clang::SmallString<256> out;
    info.FormatDiagnostic(out);
    std::cerr << "ERROR: "  << out.c_str() << "\n";
  }
};


namespace Fortran {
  
//
// The Fortran version of TargeInfo
//
class TargetInfo {
public:

  // A few type aliases to clarify the meaning
  
  typedef int integer_k;   // An int representing a INTEGER kind
  typedef int real_k;      // An int representing a REAL or COMPLEX kind
  typedef int character_k; // An int representing a CHARACTER kind
  typedef int logical_k;   // An int representing a LOGICAL kind

  typedef int integer_s;  // An int representing a INTEGER size (as in INTEGER*4)
  typedef int real_s;     // An int representing a REAL size (as in REAL*4)
  typedef int logical_s;  // An int representing a LOGICAL size (as in LOGICAL*4)
  
protected:
  
  TargetInfo() {} ; 
  
  struct IntegerInfo {
    int size;  // size in bits
    int align; // alignment in bits
  };

  struct LogicalInfo {
    int size;  // size in bits
    int align; // alignment in bits
  };

  struct CharacterType {
    int size;  // size in bits
    int align; // alignment in bits
  };
  
  struct RealInfo {
    int size;
    int align;
    const llvm::fltSemantics *format;
  };
  
  LogicalInfo * getLogicalInfo(int kind) {
    auto it = logical_types_.find(kind);
    if ( it != logical_types_.end() ) {
      return &(it->second); 
    }         
    return 0;
  }
  
  IntegerInfo * getIntegerInfo(int kind) {
    auto it = integer_types_.find(kind);
    if ( it != integer_types_.end() ) {
      return &(it->second); 
    }         
    return 0;
  }
  
  RealInfo * getRealInfo(int kind){
    auto it = real_types_.find(kind);
    if ( it != real_types_.end() ) {
      return &(it->second); 
    }         
    return 0;
  }
  
  CharacterType * getCharacterInfo(int kind) {
    auto it = character_types_.find(kind);
    if ( it != character_types_.end() ) {
      return &(it->second); 
    }         
    return 0;
  }
  
  std::map<logical_k,LogicalInfo> logical_types_;
  std::map<integer_k,IntegerInfo> integer_types_;
  std::map<real_k,RealInfo>       real_types_;
  std::map<character_k,CharacterType>  character_types_;

  // All supported kinds for a given type.
  // The kind values are sorted in increasing order.
  // Warning: those vector are filled during the finalize() call
  std::vector<integer_k> integer_kinds_ ;
  std::vector<real_k> real_kinds_ ;
  std::vector<logical_k> logical_kinds_ ;
  std::vector<character_k> character_kinds_ ;
  
  //
  // Provide the mapping between the  old syntax 'TYPE*size'
  // to the new kind 'TYPE(kind)'. 
  //
  
  std::map<integer_s,integer_k> integer_size_to_kind_;
  std::map<real_s,real_k> real_size_to_kind_;
  std::map<logical_s,logical_k> logical_size_to_kind_;  

  integer_k  default_integer_kind_ =0;
  real_k default_real_kind_ = 0;
  real_k default_double_kind_ = 0;
  logical_k default_logical_kind_ =0;
  character_k default_character_kind_ =0;

  // Named constants for ISO_C_BINDING
  struct IsoCBinding {
    integer_k c_int_ = 0; 
    integer_k c_short_ = 0; 
    integer_k c_long_ = 0; 
    integer_k c_long_long_ = 0; 
    integer_k c_signed_char_ = 0; 
    integer_k c_size_t_ = 0; 
    integer_k c_int8_t_ = 0; 
    integer_k c_int16_t_ = 0; 
    integer_k c_int32_t_ = 0; 
    integer_k c_int64_t_ = 0; 
    integer_k c_int128_t_ = 0; 
    integer_k c_int_least8_t_ = 0; 
    integer_k c_int_least16_t_ = 0; 
    integer_k c_int_least32_t_ = 0; 
    integer_k c_int_fast8_t_ = 0; 
    integer_k c_int_fast16_t_ = 0; 
    integer_k c_int_fast32_t_= 0; 
    integer_k c_int_fast64_t_= 0; 
    integer_k c_int_fast128_t_= 0; 
    integer_k c_intmax_t_ = 0; 
    integer_k c_intptr_t_ = 0; 
    integer_k c_ptrdiff_t_ = 0; 
    
    real_k c_float_ = 0;  // also used for C_FLOAT_COMPLEX
    real_k c_double_ = 0; // also used for C_DOUBLE_COMPLEX
    real_k c_long_double_ = 0; // also used for C_LONG_DOUBLE_COMPLEX
    real_k c_float128_ = 0; // also used for C_FLOAT128_COMPLEX
    
    logical_k c_bool_ = 0;
    
    character_k c_char_ = 0;
    
    int c_null_char_ = 0;  
    int c_alert_     = 0;  
    int c_backspace_ = 0;  
    int c_form_feed_ = 0;  
    int c_new_line_  = 0;   
    int c_carriage_return_ = 0;
    int c_horizontal_tab_ = 0;
    int c_vertical_tab_ = 0; 
  } icb ;

  // This enum provides some reasonnable values for
  // some fields of the IsoFortranEnv structure below.
  enum {
    // Indicate that this field shall be automatically resolved 

    IFE_CHARACTER_STORAGE_SIZE = 8 ,
    IFE_FILE_STORAGE_SIZE = 8,
    IFE_NUMERIC_STORAGE_SIZE = 8, 

    IFE_PARENT_TEAM = -111,  // TODO
    IFE_CURRENT_TEAM = -666, // TODO

    IFE_IOSTAT_END = -1,
    IFE_IOSTAT_EOR = -2,

    IFE_ERROR_UNIT = 0,
    IFE_INPUT_UNIT = 5,
    IFE_OUTPUT_UNIT = 6,

    // The values for those 'constants' are not precisely
    // defined by the standard but they should all be different.
    // (see 16.10.2.3).
    // Remark: Some values have a sign requirement which is made
    //         explicit here with '+' or '-'
    // Remark: Strangely, the standard does not define a 'stat'
    //         value for success. Is that supposed to be 0?
    IFE_IOSTAT_INQUIRE_INTERNAL_UNIT = 9999,  
    IFE_STAT_FAILED_IMAGE_no = -111,
    IFE_STAT_FAILED_IMAGE_yes = +111,
    IFE_STAT_LOCKED= 222,
    IFE_STAT_LOCKED_OTHER_IMAGE= 333,
    IFE_STAT_STOPPED_IMAGE= +444,
    IFE_STAT_UNLOCKED = 555,
    IFE_STAT_UNLOCKED_FAILED_IMAGE= 666,
  };

  // Information for ISO_FORTRAN_ENV.
  // The kind values are expected to be filled during constructor
  // or during finalize call. 
  struct IsoFortranEnv {  

    integer_k atomic_int_kind_ = 0;
    logical_k atomic_logical_kind_ = 0;

    int character_storage_size_= IFE_CHARACTER_STORAGE_SIZE;
    int file_storage_size_ = IFE_FILE_STORAGE_SIZE;
    int numeric_storage_size_ = IFE_NUMERIC_STORAGE_SIZE;
    int current_team_ = IFE_CURRENT_TEAM;  

    int error_unit_ = IFE_ERROR_UNIT; 
    int output_unit_ = IFE_OUTPUT_UNIT; 
    int input_unit_ = IFE_INPUT_UNIT; 

    integer_k int8_ = 0;
    integer_k int16_ = 0;
    integer_k int32_ = 0;
    integer_k int64_ = 0;

    real_k real32_ = 0;
    real_k real64_ = 0;
    real_k real128_ = 0;
    
    int iostat_end_ = IFE_IOSTAT_END;
    int iostat_eor_ = IFE_IOSTAT_EOR;

    int iostat_inquire_internal_unit_ = IFE_IOSTAT_INQUIRE_INTERNAL_UNIT;
    int stat_failed_image_ = IFE_STAT_FAILED_IMAGE_no ;
    int stat_locked_ = IFE_STAT_LOCKED ;
    int stat_locked_other_image_ = IFE_STAT_LOCKED_OTHER_IMAGE ;
    int stat_stopped_image_ = IFE_STAT_STOPPED_IMAGE  ;
    int stat_unlocked_ = IFE_STAT_UNLOCKED;
    int stat_unlocked_failed_image_ = IFE_STAT_UNLOCKED_FAILED_IMAGE ;
    
  } ife ;    
                 
public:

  const std::vector<int> & getLogicalKinds() const { return logical_kinds_; }
  const std::vector<int> & geIntegerKinds() const { return integer_kinds_; }
  const std::vector<int> & getRealKinds() const { return real_kinds_; }
  const std::vector<int> & getCharacterKinds() const { return character_kinds_; } 


  int getDefaultLogicalKind() const { return default_logical_kind_; }
  int getDefaultIntegerKind() const { return default_integer_kind_; }
  int getDefaultCharacterKind() const { return default_character_kind_; }
  int getDefaultRealKind() const { return default_real_kind_; }
  int getDoublePrecisionKind()  const { return default_double_kind_; }

  bool HasInteger(int kind) { return bool(getIntegerInfo(kind)); }
  bool HasLogical(int kind) { return bool(getLogicalInfo(kind)); }
  bool HasReal(int kind) { return bool(getRealInfo(kind)); }
  bool HasCharacter(int kind) { return bool(getCharacterInfo(kind)); }
  
  // Provide values for the ISO_C_BINDING module 
  
  // Create a Fortran target description 
  static TargetInfo *Create(const llvm::Triple &tp /*,options*/ ) ;

  bool valid() { return valid_; }
  
private:

  // Finalize a target.
  // Return true if the target is valid.
  bool finalize(void);
};

//
// Derive a Fortran::TargetInfo from clang::TargetInfo
//
// That class is internal!
//
class TargetInfoFromClang : public TargetInfo
{
public:
  
  TargetInfoFromClang(const llvm::Triple &tp /*, options */) ;
  
protected:

  clang::TargetInfo * CTarget=0;

  // Populate integer_types_ and default_integer_kind_
  virtual void PopulateIntegerInfos();

  // Populate logical_types_ and default_logical_kind_
  //
  // It can be assumed that integer types are fully
  // configured.
  // 
  virtual void PopulateLogicalInfos();

  // Populate real_types and default_real_kind_
  //
  // It can be assumed that integer and logical types are
  // fully configured.
  //   
  virtual void PopulateRealInfos();

  
  virtual bool AllowHalfFloat() { return false; }

};

// TargetInfoFromClang can be specialized for architectures with
// specific needs. 
class TargetInfoNvptx : public TargetInfoFromClang
{
public:
  TargetInfoNvptx(const llvm::Triple &tp /*, options */) :
    TargetInfoFromClang(tp)
  {    
  }

protected:   
  bool AllowHalfFloat() override { return false; }
};


// Some vendors may prefer to have full control over the TargetInfo implementation.
//
//
class TargetInfoPgi : public TargetInfo
{
public:
  TargetInfoPgi(const llvm::Triple &tp /*, options */) {
    // ...
  }
protected:
  
};


TargetInfo *
TargetInfo::Create(const llvm::Triple &tp /*, options */ )
{
  TargetInfo * target = 0 ;
  
  if ( tp.getVendorName() == "pgi" ) {
    // TODO: Allow other vendors can create custom TargetInfo
    target = new TargetInfoPgi(tp);
  } else if ( tp.getArch() == llvm::Triple::nvptx ||
       tp.getArch() == llvm::Triple::nvptx64 ) {
    // ...
    target = new TargetInfoNvptx(tp);
  } else {
    // Use Clang target info to create a sensible one for Fortran 
    target = new TargetInfoFromClang(tp);
  }

  if (! target->finalize() ) {
    delete target; 
  }

  return target; 
}

bool
TargetInfo::finalize()
{
  // TODO: Fill missing values 
  return true;
}
  

TargetInfoFromClang::TargetInfoFromClang(const llvm::Triple &tp /*, options */) ;
{
  
  // Get the clang::TargetInfo
  
  clang::DiagnosticOptions diag_opts;
  clang::DiagnosticConsumer *diag_printer = new MyDiagnosticConsumer();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids;
  
  auto *diag_eng =  new clang::DiagnosticsEngine(diag_ids, 
                                                 &diag_opts,
                                                 diag_printer);
  
  const auto ctarget_opts = std::make_shared<clang::TargetOptions>();
  
  ctarget_opts->Triple = tp;
  
  CTarget = clang::TargetInfo::CreateTargetInfo( *diag_eng, ctarget_opts) ;
  
  if (!CTarget) {      
    return ; 
  }

  if (!init()) {
    // Insure that the target content is invalid
    default_integer_kind = 0;    
  }
}

// Retreives and sort integer keys from a map.
template <typename T>
static std::vector<int>
get_sorted_keys(std::map<int, M> const& input) {
  std::vector<int> output;
  for (auto const& element : input) {
    out.push_back(element.first);
  }
  std::sort(output.begin(),output.end()) ;
  return output;
}

bool
TargetInfoFromClang::init()
{
  valid_ = false;
  
  PopulateIntegerInfos();
  
  if ( !HasInteger(integer_default_kind_) ) {    
    return; 
  }
      
  integer_kinds_= get_sorted_keys(integer_types_);
  
  if (!PopulateLogicalInfos())
    return;
  
  logical_kinds_= get_sorted_keys(logical_types_);
  
  if (!PopulateRealInfos())
    return;

  real_kinds_= get_sorted_keys(real_types_);

  valid_ = true; 
}

void TargetInfoFromClang::PopulateIntegerInfo()
{
  // The default behavior is to have the default Fortran integer
  // match the C 'int' type.
  int default_size = CTarget->getIntSize();

  // Consider the usual case of INTEGER of kinds 1, 2, 4, 8 and 16.  
  int max_int_kind = ConsiderInteger128() ? 16 : 8 ;  
  for (integer_k kind=1 ; kind <= max_int_kind ; kind*=2 ) {
    int size = kind*8 ;    
    auto t = CTarget->GetIntTypeByWidth(size,true);
    if ( t != clang::TargetInfo::NoInt ) {
      int align = CTarget->getTypeAlign(t);
      integer_types.try_emplace(kind,size,align);
      integer_size_to_kind_[size] = kind;             
      if ( size == default_size ) {
        default_integer_kind_ = kind ; 
      }
    }
  }
}

bool TargetInfoFromClang::PopulateLogicalInfo()
{
  // The default behavior is to use the 'bool'
  // TODO: Is that really what we want? Another
  //       possible behavior could be to use
  //       
  int default_size = CTarget->getBoolSize();

  // The default behavior is to create a Logical type
  // for each supported Integer type
  for ( int kind : getIntegerKinds() ) {
    int size = getIntegerSize(kind);
    
  }
}

bool TargetInfoFromClang::PopulateRealInfo()
{
  
}


} // of namespace Fortran

//static void visitProgramUnit(const ProgramUnit &unit);

void test_target()
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

  clang::DiagnosticConsumer *pTextDiagnosticPrinter 
#if 0
    = new clang::TextDiagnosticPrinter(llvm::outs(),
                                     &diagnosticOptions);
#elif 0
    = new clang::DiagnosticConsumer();
#else
    = new MyDiagnosticConsumer();
#endif
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

    DUMP(hasLegalHalfType());
    DUMP(getHalfWidth());
    DUMP(getHalfAlign());

    DUMP(getFloatWidth());
    DUMP(getFloatAlign());

    DUMP(getDoubleWidth());
    DUMP(getDoubleAlign());

    DUMP(getLongDoubleWidth());
    DUMP(getLongDoubleAlign());
    
    DUMP(getFloat128Width());
    DUMP(getFloat128Align());

  } else {
    std::cerr << "No clang::TargetInfo for " << tp << "\n"; 
  }
}

int main(int argc, char *const argv[]) {
  test_target() ;
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
