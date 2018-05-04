#ifndef FORTRAN_BASIC_TARGET_INFO_FROM_CLANG_H_
#define FORTRAN_BASIC_TARGET_INFO_FROM_CLANG_H_

#include "../target-info.h"

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

namespace Fortran {

//
// Derive a Fortran::TargetInfo from clang::TargetInfo
//
// That class is internal!
//
class TargetInfoFromClang : public TargetInfo {
public:
  TargetInfoFromClang(const llvm::Triple &tp /*, options */);

protected:
  clang::TargetInfo *CTarget = 0;

  // Populate integer_types_ and default_integer_kind_
  virtual void PopulateInteger();

  // Populate logical_types_ and default_logical_kind_
  //
  // It can be assumed that integer types are fully
  // configured.
  //
  virtual void PopulateLogical();

  // Populate real_types and default_real_kind_
  //
  // It can be assumed that integer and logical types are
  // fully configured.
  //
  virtual void PopulateReal();

  // Populate character_types and default_character_kind_
  //
  // It can be assumed that integer, logical and real types are
  // fully configured.
  //
  virtual void PopulateChar();

  virtual bool AllowHalfFloat() { return false; }
  virtual bool AllowInteger128() { return false; }

private:
  bool init();
};

}  // namespace Fortran

#endif  // FORTRAN_BASIC_TARGET_INFO_FROM_CLANG_H_
