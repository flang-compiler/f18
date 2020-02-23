//===-- lib/semantics/check-select-stmt.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_
#define FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_

#include "flang/parser/parse-tree.h"
#include "flang/semantics/semantics.h"

namespace Fortran::semantics {
class SelectStmtChecker : public virtual BaseChecker {
public:
  SelectStmtChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::CaseConstruct &);

private:
  SemanticsContext &context_;
  bool IsValidSelectCaseType(const evaluate::DynamicType &,
      const parser::ConstantExpr &, const parser::CharBlock &);
};
}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_
