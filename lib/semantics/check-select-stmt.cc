//===--- check-select-stmt.cc - Checker for select-case, select-rank, select-type ---------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------------------===

#include "check-select-stmt.h"
#include "tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

void SelectStmtChecker::Leave(const parser::SelectCaseStmt &selectcaseStmt) {
  const auto &parsedExpr{std::get<parser::Scalar<parser::Expr>>(selectcaseStmt.t).thing};
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (auto type{expr->GetType()}) {
      if (type->category() == TypeCategory::Real) {
        context_.Say(parsedExpr.source, // C1145
            "Select case expression shall be of type character, integer, or logical"_err_en_US);
      }
    }
  }
}

}  // namespace Fortran::semantics
