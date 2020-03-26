//===-- lib/semantics/check-select-stmt.cc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-select-stmt.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

template<typename T> class SelectCaseHelper {

public:
  SelectCaseHelper(parser::Messages &messages, int kind, TypeCategory category)
    : kind_{kind}, category_(category), messages_{messages} {}
  TypeCategory GetCategory() { return category_; }

  void ReportOverlapCaseValError(parser::CharBlock currSource,
      parser::CharBlock prevSource, const char *val) {
    // C1149
    messages_
        .Say(currSource,
            "CASE value %s matches a previous CASE statement"_err_en_US, val)
        .Attach(
            prevSource, "Previous CASE statement matching %s"_err_en_US, val);
  }
  // Before inserting the value, create a range value val:val and check if
  // it overlaps any existing value
  void Insert(T val, parser::CharBlock source) {
    SelectCaseRangeType rangeValue(val, val, source);
    parser::CharBlock overlappingSource;
    if (!IsRangeValueOverlapping(rangeValue, overlappingSource)) {
      selectCaseRangeVals_.push_back(rangeValue);
    } else {
      ReportOverlapCaseValError(source, overlappingSource,
          Fortran::evaluate::GetValueAsString(val).c_str());
    }
  }
  void Insert(std::optional<T> lowerVal, std::optional<T> upperVal,
      parser::CharBlock source) {
    SelectCaseRangeType rangeValue(lowerVal, upperVal, source);

    parser::CharBlock overlappingSource;
    if (!IsRangeValueOverlapping(rangeValue, overlappingSource)) {
      selectCaseRangeVals_.push_back(rangeValue);
    } else {
      ReportOverlapCaseValError(source, overlappingSource,
          Fortran::evaluate::GetValuePairAsString(lowerVal, upperVal).c_str());
    }
  }
  template<typename U> void Insert(const U &x, parser::CharBlock source) {
    if constexpr (std::is_same_v<T, bool>) {
      if (auto val{GetBoolValue(x)}) {
        Insert(val.value(), source);
      }
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      if (auto val{GetIntValue(x)}) {
        Insert(val.value(), source);
      }
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (auto val{GetStringValue(x)}) {
        Insert(val.value(), source);
      }
    } else if constexpr (std::is_same_v<T, std::u16string>) {
      if (auto val{GetU16StringValue(x)}) {
        Insert(val.value(), source);
      }
    } else if constexpr (std::is_same_v<T, std::u32string>) {
      if (auto val{GetU32StringValue(x)}) {
        Insert(val.value(), source);
      }
    }
  }
  template<typename U>
  void Insert(const U &lower, const U &upper, parser::CharBlock source) {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      const auto lowerVal = lower ? GetIntValue(lower->thing) : std::nullopt;
      const auto upperVal = upper ? GetIntValue(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if constexpr (std::is_same_v<T, std::string>) {
      const auto lowerVal = lower ? GetStringValue(lower->thing) : std::nullopt;
      const auto upperVal = upper ? GetStringValue(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if constexpr (std::is_same_v<T, std::u16string>) {
      const auto lowerVal =
          lower ? GetU16StringValue(lower->thing) : std::nullopt;
      const auto upperVal =
          upper ? GetU16StringValue(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if constexpr (std::is_same_v<T, std::u32string>) {
      const auto lowerVal =
          lower ? GetU32StringValue(lower->thing) : std::nullopt;
      const auto upperVal =
          upper ? GetU32StringValue(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if constexpr (std::is_same_v<T, bool>) {
      messages_.Say(source,
          "SELECT CASE expression of type LOGICAL must "
          "not have range of case value"_err_en_US);
    }
  }
  bool IsValidSelectCaseType(
      const parser::ConstantExpr &constCase, parser::CharBlock src) {
    if (const auto *caseval{GetExpr(constCase)}) {
      if (auto type{caseval->GetType()}) {
        if (type->category() != category_) {  // C1147 (R1140)
          if (category_ == TypeCategory::Integer) {
            messages_.Say(src, "CASE value must be of type INTEGER"_err_en_US);
          } else if (category_ == TypeCategory::Character) {
            messages_.Say(
                src, "CASE value must be of type CHARACTER"_err_en_US);
          } else {  // TypeCategory::Logical
            messages_.Say(src, "CASE value must be of type LOGICAL"_err_en_US);
          }
        } else if (category_ == TypeCategory::Character) {
          if (kind_ != type->kind()) {
            messages_.Say(src,
                "Character kind type of case construct (=%d) mismatches "
                "with the kind type of case value (=%d)"_err_en_US,
                kind_, type->kind());
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

private:
  struct SelectCaseRangeType {
    SelectCaseRangeType(std::optional<T> fromVal, std::optional<T> toVal,
        parser::CharBlock source)
      : fromVal(fromVal), toVal(toVal), source(source) {}
    std::optional<T> fromVal;
    std::optional<T> toVal;
    parser::CharBlock source;
  };
  bool IsLessThanOrEqualTo(const T &val, const T &upperVal) {
    if constexpr (std::is_same_v<T, bool>) {
      return val == upperVal;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      return val <= upperVal;
    } else {
      return (val.compare(upperVal) <= 0);
    }
  }
  bool CheckOverlappingValue(
      std::optional<T> lowerVal, std::optional<T> upperVal) {
    return (!lowerVal || !upperVal ||
        IsLessThanOrEqualTo(lowerVal.value(), upperVal.value()));
  }
  // [a, b] & [c, d] overlaps if: (c<=b && a<=d)
  bool CheckOverlappingRangeValue(std::optional<T> memberFromVal,
      std::optional<T> memberToVal, std::optional<T> rangeFromVal,
      std::optional<T> rangeToVal) {
    return (CheckOverlappingValue(memberFromVal, rangeToVal) &&
        CheckOverlappingValue(rangeFromVal, memberToVal));
  }
  bool IsRangeValueOverlapping(
      SelectCaseRangeType &rangeValue, parser::CharBlock &overlappingSource) {
    auto selectCaseRangeValsIter{std::find_if(selectCaseRangeVals_.begin(),
        selectCaseRangeVals_.end(), [&rangeValue, this](const auto &mem) {
          return ((!mem.fromVal && !rangeValue.fromVal) ||
              (!mem.toVal && !rangeValue.toVal) ||
              this->CheckOverlappingRangeValue(mem.fromVal, mem.toVal,
                  rangeValue.fromVal, rangeValue.toVal));
        })};
    if (selectCaseRangeValsIter != selectCaseRangeVals_.end()) {
      overlappingSource = selectCaseRangeValsIter->source;
      return true;
    }
    return false;
  }

  std::vector<SelectCaseRangeType> selectCaseRangeVals_;
  int kind_;
  TypeCategory category_;
  parser::Messages &messages_;
};

template<typename T>
void ProcessSelectCaseList(
    const std::list<parser::CaseConstruct::Case> &caseList,
    SelectCaseHelper<T> &&selectCaseStmts, parser::Messages &messages) {
  bool defaultCaseFound{false};
  for (const auto &cases : caseList) {
    const auto &caseStmt{
        std::get<parser::Statement<parser::CaseStmt>>(cases.t)};
    const auto &caseSelector{
        std::get<parser::CaseSelector>(caseStmt.statement.t)};
    if (std::holds_alternative<parser::Default>(caseSelector.u)) {
      if (!defaultCaseFound) {
        defaultCaseFound = true;
      } else {  // C1146 (R1140)
        messages.Say(caseStmt.source,
            "Not more than one of the selectors of "
            "SELECT CASE statement may be DEFAULT"_err_en_US);
      }
    } else {
      const auto &caseValueRangeList{
          std::get<std::list<parser::CaseValueRange>>(caseSelector.u)};
      for (const auto &caseValues : caseValueRangeList) {
        if (const auto *constCase{
                std::get_if<parser::Scalar<parser::ConstantExpr>>(
                    &caseValues.u)}) {
          if (!selectCaseStmts.IsValidSelectCaseType(
                  constCase->thing, caseStmt.source)) {
            continue;
          }
          selectCaseStmts.Insert(constCase->thing, caseStmt.source);
        } else {
          if (selectCaseStmts.GetCategory() ==
              TypeCategory::Logical) {  // C1148 (R1140)
            messages.Say(caseStmt.source,
                "SELECT CASE expression of type LOGICAL must "
                "not have range of case value"_err_en_US);
            continue;
          }
          const auto &rangeCase{
              std::get<parser::CaseValueRange::Range>(caseValues.u)};
          const auto &lower{rangeCase.lower};
          if (lower &&
              !selectCaseStmts.IsValidSelectCaseType(
                  lower.value().thing, caseStmt.source)) {
            continue;
          }
          const auto &upper{rangeCase.upper};
          if (upper &&
              !selectCaseStmts.IsValidSelectCaseType(
                  upper.value().thing, caseStmt.source)) {
            continue;
          }
          selectCaseStmts.Insert(lower, upper, caseStmt.source);
        }
      }
    }
  }
}

void SelectStmtChecker::Leave(
    const parser::CaseConstruct &selectCaseConstruct) {
  const auto &selectCaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(
          selectCaseConstruct.t)};
  const auto &parsedExpr{
      std::get<parser::Scalar<parser::Expr>>(selectCaseStmt.statement.t).thing};
  std::optional<evaluate::DynamicType> selectCaseStmtType;
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
    return;
  }

  const auto &caseList{
      std::get<std::list<parser::CaseConstruct::Case>>(selectCaseConstruct.t)};

  if (selectCaseStmtType.value().category() == TypeCategory::Integer) {
    ProcessSelectCaseList(caseList,
        SelectCaseHelper<std::int64_t>(context_.messages(),
            selectCaseStmtType->kind(), selectCaseStmtType.value().category()),
        context_.messages());
  } else if (selectCaseStmtType.value().category() == TypeCategory::Logical) {
    ProcessSelectCaseList(caseList,
        SelectCaseHelper<bool>(context_.messages(), selectCaseStmtType->kind(),
            selectCaseStmtType.value().category()),
        context_.messages());
  } else if (selectCaseStmtType.value().category() == TypeCategory::Character) {
    if (selectCaseStmtType->kind() == 1) {
      ProcessSelectCaseList(caseList,
          SelectCaseHelper<std::string>(context_.messages(),
              selectCaseStmtType->kind(),
              selectCaseStmtType.value().category()),
          context_.messages());
    } else if (selectCaseStmtType->kind() == 2) {
      ProcessSelectCaseList(caseList,
          SelectCaseHelper<std::u16string>(context_.messages(),
              selectCaseStmtType->kind(),
              selectCaseStmtType.value().category()),
          context_.messages());
    } else if (selectCaseStmtType->kind() == 4) {
      ProcessSelectCaseList(caseList,
          SelectCaseHelper<std::u32string>(context_.messages(),
              selectCaseStmtType->kind(),
              selectCaseStmtType.value().category()),
          context_.messages());
    } else {
      context_.Say(parsedExpr.source,
          "Character kind type of case value (=%d) is not supported"_err_en_US,
          selectCaseStmtType->kind());
    }
  }
}
}  // namespace Fortran::semantics
