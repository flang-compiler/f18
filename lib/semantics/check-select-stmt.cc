//===-- lib/semantics/check-select-stmt.cc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  bool validCaseStmtType{false};
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
        "SELECT CASE expression must be of type CHARACTER, INTERGER, OR LOGICAL"_err_en_US);
  } else {
    validCaseStmtType = true;
  }

  const auto &caseList{
      std::get<std::list<parser::CaseConstruct::Case>>(selectCaseConstruct.t)};
  bool defaultCaseFound{false};
  using var_t = std::variant<std::int64_t, bool, std::string>;
  using IntType = std::vector<
      std::pair<std::optional<std::int64_t>, std::optional<std::int64_t>>>;
  using StringType = std::vector<
      std::pair<std::optional<std::string>, std::optional<std::string>>>;
  std::set<var_t> caseVals;
  std::pair<std::set<var_t>::iterator, bool> ret;
  IntType caseRangeIntVals;
  IntType::iterator intIter;
  StringType caseRangeStringVals;
  StringType::iterator stringIter;

  for (const auto &cases : caseList) {
    const auto &caseStmt{
        std::get<parser::Statement<parser::CaseStmt>>(cases.t)};
    const auto &caseSelector{
        std::get<parser::CaseSelector>(caseStmt.statement.t)};
    if (std::holds_alternative<parser::Default>(caseSelector.u)) {
      if (!defaultCaseFound) {
        defaultCaseFound = true;
      } else {  // C1146 (R1140)
        context_.Say(caseStmt.source,
            "Not more than one of the selectors of case statements may be default"_err_en_US);
      }
    } else if (validCaseStmtType) {
      const auto &caseValueRangeList{
          std::get<std::list<parser::CaseValueRange>>(caseSelector.u)};
      for (const auto &caseValues : caseValueRangeList) {
        if (const auto *constCase{
                std::get_if<parser::Scalar<parser::ConstantExpr>>(
                    &caseValues.u)}) {
          if (!isValidSelectCaseType(selectCaseStmtType.value(),
                  constCase->thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value() == TypeCategory::Integer) {
            const auto intVal = GetIntValue(constCase->thing).value();
            intIter = std::find_if(caseRangeIntVals.begin(),
                caseRangeIntVals.end(), [&intVal](const auto &mem) {
                  return ((!mem.first || (intVal >= mem.first)) &&
                      (!mem.second || (intVal <= mem.second)));
                });
            ret = caseVals.insert(intVal);
            if ((intIter == caseRangeIntVals.end()) && ret.second) {
              continue;
            }
          } else if (selectCaseStmtType.value() == TypeCategory::Character) {
            const auto strVal = GetString(constCase->thing).value();
            stringIter = std::find_if(caseRangeStringVals.begin(),
                caseRangeStringVals.end(), [&strVal](const auto &mem) {
                  return (
                      (!mem.first || (strVal.compare(mem.first.value()) > 0)) &&
                      (!mem.second ||
                          (strVal.compare(mem.second.value()) < 0)));
                });
            ret = caseVals.insert(strVal);
            if ((stringIter == caseRangeStringVals.end()) && ret.second) {
              continue;
            }
          } else {  // TypeCategory::Logical
            ret = caseVals.insert(GetBoolValue(constCase->thing).value());
            if (ret.second) {
              continue;
            }
          }
          // C1149
          context_.Say(caseStmt.source,
              "SELECT CASE statement value must not match more than one case-value-range"_err_en_US);
        } else {
          if (selectCaseStmtType.value() ==
              TypeCategory::Logical) {  // C1148 (R1140)
            context_.Say(caseStmt.source,
                "SELECT CASE expression of type LOGICAL must not have range of case value"_err_en_US);
            continue;
          }
          const auto &rangeCase{
              std::get<parser::CaseValueRange::Range>(caseValues.u)};
          const auto &lower{rangeCase.lower};
          if (lower &&
              !isValidSelectCaseType(selectCaseStmtType.value(),
                  lower.value().thing, caseStmt.source)) {
            continue;
          }

          const auto &upper{rangeCase.upper};
          if (upper &&
              !isValidSelectCaseType(selectCaseStmtType.value(),
                  upper.value().thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value() == TypeCategory::Integer) {
            const auto &valPair{std::make_pair(lower
                    ? std::optional(GetIntValue(lower->thing).value())
                    : std::nullopt,
                upper ? std::optional(GetIntValue(upper->thing).value())
                      : std::nullopt)};
            if ((!lower ||
                    (caseVals.find(valPair.first.value()) == caseVals.end())) &&
                (!upper ||
                    (caseVals.find(valPair.second.value()) ==
                        caseVals.end()))) {
              intIter = std::find_if(caseRangeIntVals.begin(),
                  caseRangeIntVals.end(), [&valPair](const auto &mem) {
                    return ((!mem.first && !valPair.first) ||
                        (!mem.second && !valPair.second) ||
                        ((!mem.first ||
                             (valPair.first &&
                                 (valPair.first.value() >= mem.first))) &&
                            (!mem.second ||
                                (valPair.first &&
                                    (valPair.first.value() <= mem.second)))) ||
                        ((!mem.first ||
                             (valPair.second &&
                                 (valPair.second.value() >= mem.first))) &&
                            (!mem.second ||
                                (valPair.second &&
                                    (valPair.second.value() <= mem.second)))));
                  });
              if (intIter == caseRangeIntVals.end()) {
                caseRangeIntVals.push_back(valPair);
                continue;
              }
            }
          } else {  // TypeCategory::Character
            const auto &strPair{std::make_pair(lower
                    ? std::optional(GetString(lower->thing).value())
                    : std::nullopt,
                upper ? std::optional(GetString(upper->thing).value())
                      : std::nullopt)};
            if ((!lower ||
                    (caseVals.find(strPair.first.value()) == caseVals.end())) &&
                (!upper ||
                    (caseVals.find(strPair.second.value()) ==
                        caseVals.end()))) {
              stringIter = std::find_if(caseRangeStringVals.begin(),
                  caseRangeStringVals.end(), [&strPair](const auto &mem) {
                    return ((!mem.first && !strPair.first) ||
                        (!mem.second && !strPair.second) ||
                        ((!mem.first ||
                             (strPair.first &&
                                 ((strPair.first.value())
                                         .compare(mem.first.value()) > 0))) &&
                            (!mem.second ||
                                (strPair.first &&
                                    ((strPair.first.value())
                                            .compare(mem.second.value()) <
                                        0)))) ||
                        ((!mem.first ||
                             (strPair.second &&
                                 (strPair.second.value().compare(
                                      mem.first.value()) > 0))) &&
                            (!mem.second ||
                                (strPair.second &&
                                    (strPair.second.value().compare(
                                         mem.second.value()) < 0)))));
                  });

              if (stringIter == caseRangeStringVals.end()) {
                caseRangeStringVals.push_back(strPair);
                continue;
              }
            }
          }
          // C1149
          context_.Say(caseStmt.source,
              "SELECT CASE statement value must not match more than one case-value-range"_err_en_US);
        }
      }
    }
  }
}

bool SelectStmtChecker::isValidSelectCaseType(const TypeCategory &expectedType,
    const parser::ConstantExpr &constCase, const parser::CharBlock &src) {
  if (const auto *caseval{GetExpr(constCase)}) {
    if (auto type{caseval->GetType()}) {
      if (type->category() != expectedType) {  // C1147 (R1140)
        if (expectedType == TypeCategory::Integer) {
          context_.Say(
              src, "SELECT CASE value must be of type INTEGER"_err_en_US);
        } else if (expectedType == TypeCategory::Character) {
          context_.Say(
              src, "SELECT CASE value must be of type CHARACTER"_err_en_US);
        } else {  // TypeCategory::Logical
          context_.Say(
              src, "SELECT CASE value must be of type LOGICAL"_err_en_US);
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

}  // namespace Fortran::semantics
