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
            "SELECT CASE statement value at this location overlaps with the below location"_err_en_US)
        .Attach(source1,
            "SELECT CASE statement value overlaps with the above location"_err_en_US);
  }

  void Insert(std::int64_t intVal, parser::CharBlock source) {
    auto selectCaseIntRangeValsIter =
        std::find_if(selectCaseIntRangeVals_.begin(),
            selectCaseIntRangeVals_.end(), [&intVal](const auto &mem) {
              return ((!mem.fromVal_ || (intVal >= mem.fromVal_.value())) &&
                  (!mem.toVal_ || (intVal <= mem.toVal_.value())));
            });
    if (selectCaseIntRangeValsIter != selectCaseIntRangeVals_.end()) {
      ReportOverlapCaseValError(selectCaseIntRangeValsIter->source_, source);
    }
    auto selectCaseIntValsInsertStatus =
        selectCaseIntVals_.insert(std::make_pair(intVal, source));
    if (selectCaseIntValsInsertStatus.second == false) {
      ReportOverlapCaseValError(
          selectCaseIntValsInsertStatus.first->second, source);
    }
  }
  void Insert(std::string strVal, parser::CharBlock source) {
    auto selectCaseStringRangeValsIter = std::find_if(
        selectCaseStringRangeVals_.begin(), selectCaseStringRangeVals_.end(),
        [&strVal](const auto &mem) {
          return (
              (!mem.fromVal_ || (strVal.compare(mem.fromVal_.value()) >= 0)) &&
              (!mem.toVal_ || (strVal.compare(mem.toVal_.value()) <= 0)));
        });
    if (selectCaseStringRangeValsIter != selectCaseStringRangeVals_.end()) {
      ReportOverlapCaseValError(selectCaseStringRangeValsIter->source_, source);
    }
    auto selectCaseStringValsInsertStatus =
        selectCaseStringVals_.insert(std::make_pair(strVal, source));
    if (selectCaseStringValsInsertStatus.second == false) {
      ReportOverlapCaseValError(
          selectCaseStringValsInsertStatus.first->second, source);
    }
  }
  void Insert(bool logicalVal, parser::CharBlock source) {
    auto selectCaseLogicalValsInsertStatus =
        selectCaseLogicalVals_.insert(std::make_pair(logicalVal, source));
    if (selectCaseLogicalValsInsertStatus.second == false) {
      ReportOverlapCaseValError(
          selectCaseLogicalValsInsertStatus.first->second, source);
    }
  }
  void Insert(std::optional<std::int64_t> lowerVal,
      std::optional<std::int64_t> upperVal, parser::CharBlock source) {
    SelectCaseRangeType<std::int64_t> intRangeValue(lowerVal, upperVal, source);
    auto selectCaseIntValsIter = std::find_if(selectCaseIntVals_.begin(),
        selectCaseIntVals_.end(), [&intRangeValue](const auto &mem) {
          return ((!intRangeValue.fromVal_ ||
                      (intRangeValue.fromVal_.value() <= mem.first)) &&
              (!intRangeValue.toVal_ ||
                  (intRangeValue.toVal_.value() >= mem.first)));
        });
    if (selectCaseIntValsIter != selectCaseIntVals_.end()) {
      ReportOverlapCaseValError(selectCaseIntValsIter->second, source);
    }
    auto selectCaseIntRangeValsIter = std::find_if(
        selectCaseIntRangeVals_.begin(), selectCaseIntRangeVals_.end(),
        [&intRangeValue](const auto &mem) {
          return ((!mem.fromVal_ && !intRangeValue.fromVal_) ||
              (!mem.toVal_ && !intRangeValue.toVal_) ||
              ((!mem.fromVal_ ||
                   (intRangeValue.fromVal_ &&
                       (intRangeValue.fromVal_.value() >= mem.fromVal_))) &&
                  (!mem.toVal_ ||
                      (intRangeValue.fromVal_ &&
                          (intRangeValue.fromVal_.value() <= mem.toVal_)))) ||
              ((!mem.fromVal_ ||
                   (intRangeValue.toVal_ &&
                       (intRangeValue.toVal_.value() >= mem.fromVal_))) &&
                  (!mem.toVal_ ||
                      (intRangeValue.toVal_ &&
                          (intRangeValue.toVal_.value() <= mem.toVal_)))));
        });
    if (selectCaseIntRangeValsIter == selectCaseIntRangeVals_.end()) {
      selectCaseIntRangeVals_.push_back(intRangeValue);
    } else {
      ReportOverlapCaseValError(selectCaseIntRangeValsIter->source_, source);
    }
  }
  void Insert(std::optional<std::string> lowerVal,
      std::optional<std::string> upperVal, parser::CharBlock source) {
    SelectCaseRangeType<std::string> stringRangeValue(
        lowerVal, upperVal, source);
    auto selectCaseStringValsIter = std::find_if(selectCaseStringVals_.begin(),
        selectCaseStringVals_.end(), [&stringRangeValue](const auto &mem) {
          return ((!stringRangeValue.fromVal_ ||
                      ((stringRangeValue.fromVal_.value()).compare(mem.first) >=
                          0)) &&
              (!stringRangeValue.toVal_ ||
                  ((stringRangeValue.toVal_.value()).compare(mem.first) <= 0)));
        });
    if (selectCaseStringValsIter != selectCaseStringVals_.end()) {
      ReportOverlapCaseValError(selectCaseStringValsIter->second, source);
    }
    auto selectCaseStringRangeValsIter = std::find_if(
        selectCaseStringRangeVals_.begin(), selectCaseStringRangeVals_.end(),
        [&stringRangeValue](const auto &mem) {
          return ((!mem.fromVal_ && !stringRangeValue.fromVal_) ||
              (!mem.toVal_ && !stringRangeValue.toVal_) ||
              ((!mem.fromVal_ ||
                   (stringRangeValue.fromVal_ &&
                       ((stringRangeValue.fromVal_.value())
                               .compare(mem.fromVal_.value()) >= 0))) &&
                  (!mem.toVal_ ||
                      (stringRangeValue.fromVal_ &&
                          ((stringRangeValue.fromVal_.value())
                                  .compare(mem.toVal_.value()) <= 0)))) ||
              ((!mem.fromVal_ ||
                   (stringRangeValue.toVal_ &&
                       (stringRangeValue.toVal_.value().compare(
                            mem.fromVal_.value()) >= 0))) &&
                  (!mem.toVal_ ||
                      (stringRangeValue.toVal_ &&
                          (stringRangeValue.toVal_.value().compare(
                               mem.toVal_.value()) <= 0)))));
        });
    if (selectCaseStringRangeValsIter == selectCaseStringRangeVals_.end()) {
      selectCaseStringRangeVals_.push_back(stringRangeValue);
    } else {
      ReportOverlapCaseValError(selectCaseStringRangeValsIter->source_, source);
    }
  }

private:
  template<typename T> struct SelectCaseRangeType {
    SelectCaseRangeType(std::optional<T> fromVal, std::optional<T> toVal,
        parser::CharBlock source)
      : fromVal_(fromVal), toVal_(toVal), source_(source) {}
    std::optional<T> fromVal_;
    std::optional<T> toVal_;
    parser::CharBlock source_;
  };

  std::map<bool, parser::CharBlock> selectCaseLogicalVals_;
  std::map<std::int64_t, parser::CharBlock> selectCaseIntVals_;
  std::map<std::string, parser::CharBlock> selectCaseStringVals_;
  std::vector<SelectCaseRangeType<std::int64_t>> selectCaseIntRangeVals_;
  std::vector<SelectCaseRangeType<std::string>> selectCaseStringRangeVals_;
  parser::Messages &messages_;
};

void SelectStmtChecker::Leave(
    const parser::CaseConstruct &selectCaseConstruct) {
  const auto &selectCaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(
          selectCaseConstruct.t)};
  const auto &parsedExpr{
      std::get<parser::Scalar<parser::Expr>>(selectCaseStmt.statement.t).thing};
  std::optional<evaluate::DynamicType> selectCaseStmtType;
  bool validCaseStmtType{false};
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (auto type{expr->GetType()}) {
      selectCaseStmtType = type;
    }
  }
  if (!selectCaseStmtType) {
    return;
  }
  if ((selectCaseStmtType.value().category() != TypeCategory::Integer) &&
      (selectCaseStmtType.value().category() != TypeCategory::Character) &&
      (selectCaseStmtType.value().category() !=
          TypeCategory::Logical)) {  // C1145
    context_.Say(parsedExpr.source,
        "SELECT CASE expression must be of type CHARACTER, INTEGER, OR LOGICAL"_err_en_US);
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
                  constCase->thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value().category() == TypeCategory::Integer) {
            selectCaseStmts.Insert(
                GetIntValue(constCase->thing).value(), caseStmt.source);
          } else if (selectCaseStmtType.value().category() ==
              TypeCategory::Character) {
            selectCaseStmts.Insert(
                GetStringValue(constCase->thing).value(), caseStmt.source);
          } else {  // TypeCategory::Logical
            selectCaseStmts.Insert(
                GetBoolValue(constCase->thing).value(), caseStmt.source);
          }
        } else {
          if (selectCaseStmtType.value().category() ==
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
                  lower.value().thing, caseStmt.source)) {
            continue;
          }

          const auto &upper{rangeCase.upper};
          if (upper &&
              !IsValidSelectCaseType(selectCaseStmtType.value(),
                  upper.value().thing, caseStmt.source)) {
            continue;
          }

          if (selectCaseStmtType.value().category() == TypeCategory::Integer) {
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

bool SelectStmtChecker::IsValidSelectCaseType(
    const evaluate::DynamicType &expectedType,
    const parser::ConstantExpr &constCase, const parser::CharBlock &src) {
  if (const auto *caseval{GetExpr(constCase)}) {
    if (auto type{caseval->GetType()}) {
      if (type->category() != expectedType.category()) {  // C1147 (R1140)
        if (expectedType.category() == TypeCategory::Integer) {
          context_.Say(
              src, "SELECT CASE value must be of type INTEGER"_err_en_US);
        } else if (expectedType.category() == TypeCategory::Character) {
          context_.Say(
              src, "SELECT CASE value must be of type CHARACTER"_err_en_US);
        } else {  // TypeCategory::Logical
          context_.Say(
              src, "SELECT CASE value must be of type LOGICAL"_err_en_US);
        }
      } else if (expectedType.category() == TypeCategory::Character) {
        if (expectedType.kind() != type->kind()) {
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
