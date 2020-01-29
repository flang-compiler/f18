//===-- lib/semantics/check-select-stmt.cc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-select-stmt.h"
#include "flang/parser/message.h"
#include "flang/semantics/tools.h"

namespace Fortran::semantics {

class SelectCaseHelper {

public:
  SelectCaseHelper(parser::Messages &messages) : messages_{messages} {}

  void ReportOverlapCaseValError(
      parser::CharBlock &source1, parser::CharBlock &source2) {
    // C1149
    messages_
        .Say(source2,
            "SELECT CASE statement value at this location overlaps with below location"_err_en_US)
        .Attach(source1, ""_err_en_US);
  }

  void Insert(std::int64_t intVal, parser::CharBlock source) {
    selectCaseIntRangeValsIter = std::find_if(selectCaseIntRangeVals.begin(),
        selectCaseIntRangeVals.end(), [&intVal](const auto &mem) {
          return ((!std::get<0>(mem) || (intVal >= std::get<0>(mem).value())) &&
              (!std::get<1>(mem) || (intVal <= std::get<1>(mem).value())));
        });
    if (selectCaseIntRangeValsIter != selectCaseIntRangeVals.end()) {
      ReportOverlapCaseValError(
          std::get<2>(*selectCaseIntRangeValsIter), source);
    }

    selectCaseValsIter = std::find_if(selectCaseVals.begin(),
        selectCaseVals.end(), [&intVal](const auto &mem) {
          return (intVal == std::get<std::int64_t>(mem.first));
        });
    if (selectCaseValsIter != selectCaseVals.end()) {
      ReportOverlapCaseValError(selectCaseValsIter->second, source);
    } else {
      selectCaseVals.push_back(std::make_pair(intVal, source));
    }
  }
  void Insert(std::string strVal, parser::CharBlock source) {
    selectCaseStringRangeValsIter =
        std::find_if(selectCaseStringRangeVals.begin(),
            selectCaseStringRangeVals.end(), [&strVal](const auto &mem) {
              return ((!std::get<0>(mem) ||
                          (strVal.compare(std::get<0>(mem).value()) >= 0)) &&
                  (!std::get<1>(mem) ||
                      (strVal.compare(std::get<1>(mem).value()) <= 0)));
            });
    if (selectCaseStringRangeValsIter != selectCaseStringRangeVals.end()) {
      ReportOverlapCaseValError(
          std::get<2>(*selectCaseStringRangeValsIter), source);
    }
    selectCaseValsIter = std::find_if(selectCaseVals.begin(),
        selectCaseVals.end(), [&strVal](const auto &mem) {
          return (!strVal.compare(std::get<std::string>(mem.first)));
        });
    if (selectCaseValsIter != selectCaseVals.end()) {
      ReportOverlapCaseValError(selectCaseValsIter->second, source);
    } else {
      selectCaseVals.push_back(std::make_pair(strVal, source));
    }
  }
  void Insert(bool logicalVal, parser::CharBlock source) {
    selectCaseValsIter = std::find_if(selectCaseVals.begin(),
        selectCaseVals.end(), [&logicalVal](const auto &mem) {
          return (logicalVal == std::get<bool>(mem.first));
        });
    if (selectCaseValsIter != selectCaseVals.end()) {
      ReportOverlapCaseValError(selectCaseValsIter->second, source);
    } else {
      selectCaseVals.push_back(std::make_pair(logicalVal, source));
    }
  }
  void Insert(std::optional<std::int64_t> lowerVal,
      std::optional<std::int64_t> upperVal, parser::CharBlock source) {
    const auto &intRangeValue{std::make_tuple(lowerVal, upperVal, source)};
    selectCaseValsIter = std::find_if(selectCaseVals.begin(),
        selectCaseVals.end(), [&intRangeValue](const auto &mem) {
          return ((!std::get<0>(intRangeValue) ||
                      (std::get<0>(intRangeValue).value() <=
                          std::get<std::int64_t>(mem.first))) &&
              (!std::get<1>(intRangeValue) ||
                  (std::get<1>(intRangeValue).value() >=
                      std::get<std::int64_t>(mem.first))));
        });
    if (selectCaseValsIter != selectCaseVals.end()) {
      ReportOverlapCaseValError(selectCaseValsIter->second, source);
    }

    selectCaseIntRangeValsIter = std::find_if(selectCaseIntRangeVals.begin(),
        selectCaseIntRangeVals.end(), [&intRangeValue](const auto &mem) {
          return ((!std::get<0>(mem) && !std::get<0>(intRangeValue)) ||
              (!std::get<1>(mem) && !std::get<1>(intRangeValue)) ||
              ((!std::get<0>(mem) ||
                   (std::get<0>(intRangeValue) &&
                       (std::get<0>(intRangeValue).value() >=
                           std::get<0>(mem)))) &&
                  (!std::get<1>(mem) ||
                      (std::get<0>(intRangeValue) &&
                          (std::get<0>(intRangeValue).value() <=
                              std::get<1>(mem))))) ||
              ((!std::get<0>(mem) ||
                   (std::get<1>(intRangeValue) &&
                       (std::get<1>(intRangeValue).value() >=
                           std::get<0>(mem)))) &&
                  (!std::get<1>(mem) ||
                      (std::get<1>(intRangeValue) &&
                          (std::get<1>(intRangeValue).value() <=
                              std::get<1>(mem))))));
        });
    if (selectCaseIntRangeValsIter == selectCaseIntRangeVals.end()) {
      selectCaseIntRangeVals.push_back(intRangeValue);
    } else {
      ReportOverlapCaseValError(
          std::get<2>(*selectCaseIntRangeValsIter), source);
    }
  }
  void Insert(std::optional<std::string> lowerVal,
      std::optional<std::string> upperVal, parser::CharBlock source) {
    const auto &stringRangeValue{std::make_tuple(lowerVal, upperVal, source)};
    selectCaseValsIter = std::find_if(selectCaseVals.begin(),
        selectCaseVals.end(), [&stringRangeValue](const auto &mem) {
          return (
              (!std::get<0>(stringRangeValue) ||
                  ((std::get<0>(stringRangeValue).value())
                          .compare(std::get<std::string>(mem.first)) >= 0)) &&
              (!std::get<1>(stringRangeValue) ||
                  ((std::get<1>(stringRangeValue).value())
                          .compare(std::get<std::string>(mem.first)) <= 0)));
        });
    if (selectCaseValsIter != selectCaseVals.end()) {
      ReportOverlapCaseValError(selectCaseValsIter->second, source);
    }

    selectCaseStringRangeValsIter = std::find_if(
        selectCaseStringRangeVals.begin(), selectCaseStringRangeVals.end(),
        [&stringRangeValue](const auto &mem) {
          return ((!std::get<0>(mem) && !std::get<0>(stringRangeValue)) ||
              (!std::get<1>(mem) && !std::get<1>(stringRangeValue)) ||
              ((!std::get<0>(mem) ||
                   (std::get<0>(stringRangeValue) &&
                       ((std::get<0>(stringRangeValue).value())
                               .compare(std::get<0>(mem).value()) >= 0))) &&
                  (!std::get<1>(mem) ||
                      (std::get<0>(stringRangeValue) &&
                          ((std::get<0>(stringRangeValue).value())
                                  .compare(std::get<1>(mem).value()) <= 0)))) ||
              ((!std::get<0>(mem) ||
                   (std::get<1>(stringRangeValue) &&
                       (std::get<1>(stringRangeValue)
                               .value()
                               .compare(std::get<0>(mem).value()) >= 0))) &&
                  (!std::get<1>(mem) ||
                      (std::get<1>(stringRangeValue) &&
                          (std::get<1>(stringRangeValue)
                                  .value()
                                  .compare(std::get<1>(mem).value()) <= 0)))));
        });

    if (selectCaseStringRangeValsIter == selectCaseStringRangeVals.end()) {
      selectCaseStringRangeVals.push_back(stringRangeValue);
    } else {
      ReportOverlapCaseValError(
          std::get<2>(*selectCaseStringRangeValsIter), source);
    }
  }

private:
  using selectCaseStmtTypes =
      std::vector<std::pair<std::variant<std::int64_t, bool, std::string>,
          parser::CharBlock>>;
  selectCaseStmtTypes selectCaseVals;
  selectCaseStmtTypes::iterator selectCaseValsIter;
  using selectCaseIntRangeType =
      std::vector<std::tuple<std::optional<std::int64_t>,
          std::optional<std::int64_t>, parser::CharBlock>>;
  selectCaseIntRangeType selectCaseIntRangeVals;
  selectCaseIntRangeType::iterator selectCaseIntRangeValsIter;
  using selectCaseStringRangeType =
      std::vector<std::tuple<std::optional<std::string>,
          std::optional<std::string>, parser::CharBlock>>;
  selectCaseStringRangeType selectCaseStringRangeVals;
  selectCaseStringRangeType::iterator selectCaseStringRangeValsIter;
  parser::Messages &messages_;
};

void SelectStmtChecker::Leave(
    const parser::CaseConstruct &selectCaseConstruct) {
  const auto &selectCaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(
          selectCaseConstruct.t)};
  const auto &parsedExpr{
      std::get<parser::Scalar<parser::Expr>>(selectCaseStmt.statement.t).thing};
  std::optional<TypeCategory> selectCaseStmtType;
  int selectCaseStmtKind;
  bool validCaseStmtType{false};
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (auto type{expr->GetType()}) {
      selectCaseStmtType = type->category();
      selectCaseStmtKind = type->kind();
    }
  }
  if (!selectCaseStmtType) {
    return;
  }
  if ((selectCaseStmtType.value() != TypeCategory::Integer) &&
      (selectCaseStmtType.value() != TypeCategory::Character) &&
      (selectCaseStmtType.value() != TypeCategory::Logical)) {  // C1145
    context_.Say(parsedExpr.source,
        "SELECT CASE expression must be of type CHARACTER, INTERGER, OR LOGICAL"_err_en_US);
  } else {
    validCaseStmtType = true;
  }

  const auto &caseList{
      std::get<std::list<parser::CaseConstruct::Case>>(selectCaseConstruct.t)};
  bool defaultCaseFound{false};
  SelectCaseHelper selectCaseStmts(context_.messages());

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
            "Not more than one of the selectors of SELECT CASE statement may be default"_err_en_US);
      }
    } else if (validCaseStmtType) {
      const auto &caseValueRangeList{
          std::get<std::list<parser::CaseValueRange>>(caseSelector.u)};
      for (const auto &caseValues : caseValueRangeList) {
        if (const auto *constCase{
                std::get_if<parser::Scalar<parser::ConstantExpr>>(
                    &caseValues.u)}) {
          if (!IsValidSelectCaseType(selectCaseStmtType.value(),
                  selectCaseStmtKind, constCase->thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value() == TypeCategory::Integer) {
            selectCaseStmts.Insert(
                GetIntValue(constCase->thing).value(), caseStmt.source);
          } else if (selectCaseStmtType.value() == TypeCategory::Character) {
            selectCaseStmts.Insert(
                GetStringValue(constCase->thing).value(), caseStmt.source);
          } else {  // TypeCategory::Logical
            selectCaseStmts.Insert(
                GetBoolValue(constCase->thing).value(), caseStmt.source);
          }
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
              !IsValidSelectCaseType(selectCaseStmtType.value(),
                  selectCaseStmtKind, lower.value().thing, caseStmt.source)) {
            continue;
          }

          const auto &upper{rangeCase.upper};
          if (upper &&
              !IsValidSelectCaseType(selectCaseStmtType.value(),
                  selectCaseStmtKind, upper.value().thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value() == TypeCategory::Integer) {
            const auto lowerVal = lower
                ? std::optional(GetIntValue(lower->thing).value())
                : std::nullopt;
            const auto upperVal = upper
                ? std::optional(GetIntValue(upper->thing).value())
                : std::nullopt;
            selectCaseStmts.Insert(lowerVal, upperVal, caseStmt.source);
          } else {  // TypeCategory::Character
            const auto lowerVal = lower
                ? std::optional(GetStringValue(lower->thing).value())
                : std::nullopt;
            const auto upperVal = upper
                ? std::optional(GetStringValue(upper->thing).value())
                : std::nullopt;
            selectCaseStmts.Insert(lowerVal, upperVal, caseStmt.source);
          }
        }
      }
    }
  }
}

bool SelectStmtChecker::IsValidSelectCaseType(const TypeCategory &expectedType,
    const int &selectCaseStmtKind, const parser::ConstantExpr &constCase,
    const parser::CharBlock &src) {
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
      } else if (expectedType == TypeCategory::Character) {
        if (selectCaseStmtKind != type->kind()) {
          context_.Say(src,
              "SELECT CASE value kind must be same as SELECT CASE expression kind"_err_en_US);
        } else {
          return true;
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

}  // namespace Fortran::semantics
