//= check-select-stmt.cc - Checker for select-case, select-rank, select-type ==
// TODO select-rank, select-type
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
    const parser::CaseConstruct &selectCaseConstruct) {
  const auto &selectCaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(
          selectCaseConstruct.t)};
  const auto &parsedExpr{
      std::get<parser::Scalar<parser::Expr>>(selectCaseStmt.statement.t).thing};
  std::optional<TypeCategory> selectCaseStmtType;
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (auto type{expr->GetType()}) {
      selectCaseStmtType = type->category();
    }
  }
  if (!selectCaseStmtType ||
      ((selectCaseStmtType.value() != TypeCategory::Integer) &&
          (selectCaseStmtType.value() != TypeCategory::Character) &&
          (selectCaseStmtType.value() != TypeCategory::Logical))) {  // C1145
    context_.Say(parsedExpr.source,
        "SELECT CASE expression must be of type character, integer, or logical"_err_en_US);
  }

  const auto &caseList{
      std::get<std::list<parser::CaseConstruct::Case>>(selectCaseConstruct.t)};
  bool defaultCaseFound{false};
  for (const auto &cases : caseList) {
    const auto &casestmt{
        std::get<parser::Statement<parser::CaseStmt>>(cases.t)};
    const auto &caseselector{
        std::get<parser::CaseSelector>(casestmt.statement.t)};
    if (std::holds_alternative<parser::Default>(caseselector.u)) {
      if (!defaultCaseFound) {
        defaultCaseFound = true;
      } else {  // C1146 (R1140)
        context_.Say(casestmt.source,
            "Not more than one of the selectors of case statements may be default"_err_en_US);
      }
    } else if (selectCaseStmtType) {
      const auto &caseValueRangeList{
          std::get<std::list<parser::CaseValueRange>>(caseselector.u)};
      for (const auto &caseValues : caseValueRangeList) {
        if (const auto *constCase{
                std::get_if<parser::Scalar<parser::ConstantExpr>>(
                    &caseValues.u)}) {
          CheckSelectCaseType(
              selectCaseStmtType.value(), constCase->thing, casestmt.source);
        } else {
          if (selectCaseStmtType.value() ==
              TypeCategory::Logical) {  // C1148 (R1140)
            context_.Say(casestmt.source,
                "SELECT CASE expression of type logical must not have case value range using colon"_err_en_US);
          }
          const auto &rangeCase{
              std::get<parser::CaseValueRange::Range>(caseValues.u)};
          if (const auto &lower{rangeCase.lower}) {
            CheckSelectCaseType(selectCaseStmtType.value(), lower.value().thing,
                casestmt.source);
          }
          if (const auto &upper{rangeCase.upper}) {
            CheckSelectCaseType(selectCaseStmtType.value(), upper.value().thing,
                casestmt.source);
          }
          // TODO C1149
        }
      }
    }
  }
}

void SelectStmtChecker::CheckSelectCaseType(const TypeCategory &expectedType,
    const parser::ConstantExpr &constCase, const parser::CharBlock &src) {
  if (const auto *caseval{GetExpr(constCase)}) {
    if (auto type{caseval->GetType()}) {
      if (type->category() != expectedType) {  // C1147 (R1140)
        context_.Say(src,
            "SELECT CASE value type must be same as SELECT CASE expression type"_err_en_US);
      }
    }
  }
}

}  // namespace Fortran::semantics
