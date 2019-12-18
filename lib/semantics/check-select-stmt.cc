//= check-select-stmt.cc - Checker for select-case, select-rank, select-type ==
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=--------------------------------------------------------------------------==

#include "check-select-stmt.h"
#include "tools.h"
#include "../parser/message.h"

namespace Fortran::semantics {

void SelectStmtChecker::Leave(
    const parser::CaseConstruct &selectcaseConstruct) {
  const auto &selectcaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(
          selectcaseConstruct.t)};
  const auto &parsedExpr{
      std::get<parser::Scalar<parser::Expr>>(selectcaseStmt.statement.t).thing};
  TypeCategory selectcasestmttype;
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (auto type{expr->GetType()}) {
      selectcasestmttype = type->category();
      if ((selectcasestmttype != TypeCategory::Integer) &&
          (selectcasestmttype != TypeCategory::Character) &&
          (selectcasestmttype != TypeCategory::Logical)) {
        // C1145 case-expr shall be of type character, integer, or logical.
        context_.Say(parsedExpr.source,
            "Select case expression must be of type character, integer, or logical"_err_en_US);
      }
    }
  }

  const auto &caselist{
      std::get<std::list<parser::CaseConstruct::Case>>(selectcaseConstruct.t)};
  bool defaultcasefound = false;
  for (const auto &cases : caselist) {
    const auto &casestmt{
        std::get<parser::Statement<parser::CaseStmt>>(cases.t)};
    const auto &caseselector{
        std::get<parser::CaseSelector>(casestmt.statement.t)};
    if (std::holds_alternative<parser::Default>(caseselector.u)) {
      if (!defaultcasefound) {
        defaultcasefound = true;
      } else {
        // C1146 (R1140) No more than one of the selectors of one of the CASE
        //       statements shall be DEFAULT.
        context_.Say(casestmt.source,
            "Not more than one of the selectors of case statements must be default"_err_en_US);
      }
    } else {
      const auto &casevaluerangelist{
          std::get<std::list<parser::CaseValueRange>>(caseselector.u)};
      for (const auto &casevalues : casevaluerangelist) {
        if (std::holds_alternative<parser::Scalar<parser::ConstantExpr>>(
                casevalues.u)) {
          const auto &constcase{
              std::get<parser::Scalar<parser::ConstantExpr>>(casevalues.u)
                  .thing};
          CheckSelectCaseType(selectcasestmttype, constcase, casestmt.source);
        } else {
          if (selectcasestmttype == TypeCategory::Logical) {
            // C1148  (R1140) A case-value-range using a colon shall not be used
            //        if case-expr is of type logical.
            context_.Say(casestmt.source,
                "Select case expression of type logical must not have case value range using colon"_err_en_US);
          }
          const auto &rangecase{
              std::get<parser::CaseValueRange::Range>(casevalues.u)};
          if (const auto &lower{rangecase.lower}) {
            CheckSelectCaseType(
                selectcasestmttype, lower.value().thing, casestmt.source);
          }
          if (const auto &upper{rangecase.upper}) {
            CheckSelectCaseType(
                selectcasestmttype, upper.value().thing, casestmt.source);
          }
        }
      }
    }
  }
}

void SelectStmtChecker::CheckSelectCaseType(const TypeCategory &expectedtype,
    const Fortran::parser::ConstantExpr &constcase,
    const Fortran::parser::CharBlock &src) {
  if (const auto *caseval{GetExpr(constcase)}) {
    if (auto type{caseval->GetType()}) {
      if (type->category() != expectedtype) {
        // C1147  (R1140) For a given case-construct, each case-value shall be
        //        of the same type as case-expr.
        //        For character type, the kind type parameters shall be the
        //        same; character length differences are allowed.
        context_.Say(src,
            "Select case value type must be same as select case expression type"_err_en_US);
      }
    }
  }
}

}  // namespace Fortran::semantics
