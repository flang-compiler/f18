//===-- lib/semantics/check-select-stmt.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_
#define FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_

#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"

namespace Fortran::semantics {
class SelectConstructChecker : public virtual BaseChecker {
public:
  SelectConstructChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::SelectRankConstruct &);

private:
  const SomeExpr *ResolveSelector(const parser::Selector &);
  SemanticsContext &context_;
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_SELECT_STMT_H_
