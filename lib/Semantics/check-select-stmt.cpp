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

class SelectCaseHelper {

public:
  SelectCaseHelper(parser::Messages &messages) : messages_{messages} {}
  void SetKind(int kind) { kind_ = kind; }
  void SetTypeCategory(TypeCategory category) { category_ = category; }

  template<typename T> std::string GetValueAsString(T val) {
    if constexpr (std::is_same_v<T, bool>) {
      return std::string(val ? "true" : "false");
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      return std::string(std::to_string(val));
    } else {
      return parser::QuoteCharacterLiteral(val);
    }
  }
  template<typename T> std::string GetValuePairAsString(T fromVal, T toVal) {
    if constexpr (std::is_same_v<T, std::optional<std::int64_t>>) {
      return std::string((fromVal ? std::to_string(*fromVal) : "") + ":" +
          (toVal ? std::to_string(*toVal) : ""));
    } else {
      return ((fromVal ? parser::QuoteCharacterLiteral(fromVal.value())
                       : std::string("")) +
          std::string(":") +
          (toVal ? parser::QuoteCharacterLiteral(toVal.value())
                 : std::string("")));
    }
  }
  void ReportOverlapCaseValError(parser::CharBlock currSource,
      parser::CharBlock prevSource, const char *val) {
    // C1149
    messages_
        .Say(currSource,
            "CASE value %s matches a previous CASE statement"_err_en_US, val)
        .Attach(
            prevSource, "Previous CASE statement matching %s"_err_en_US, val);
  }
  template<typename T> void Insert(T val, parser::CharBlock source) {
    auto &rangeValsHandle{GetRangeValHandle<T>()};
    auto selectCaseRangeValsIter{std::find_if(rangeValsHandle.begin(),
        rangeValsHandle.end(), [&val, this](const auto &mem) {
          return this->CheckOverlappingValue(val, mem.fromVal, mem.toVal);
        })};
    if (selectCaseRangeValsIter != rangeValsHandle.end()) {
      ReportOverlapCaseValError(source, selectCaseRangeValsIter->source,
          GetValueAsString(val).c_str());
    }
    auto selectCaseValsInsertStatus{GetValHandle<T>().emplace(val, source)};
    if (!selectCaseValsInsertStatus.second) {
      ReportOverlapCaseValError(source,
          selectCaseValsInsertStatus.first->second,
          GetValueAsString(val).c_str());
    }
  }
  void Insert(bool logicalVal, parser::CharBlock source) {
    auto selectCaseLogicalValsInsertStatus{
        selectCaseLogicalVals_.emplace(logicalVal, source)};
    if (!selectCaseLogicalValsInsertStatus.second) {
      ReportOverlapCaseValError(source,
          selectCaseLogicalValsInsertStatus.first->second,
          GetValueAsString(logicalVal).c_str());
    }
  }
  template<typename T>
  void Insert(std::optional<T> lowerVal, std::optional<T> upperVal,
      parser::CharBlock source) {
    SelectCaseRangeType<T> rangeValue(lowerVal, upperVal, source);
    auto &rangeValsHandle{GetRangeValHandle<T>()};
    auto &valsHandle{GetValHandle<T>()};
    auto selectCaseValsIter{std::find_if(valsHandle.begin(), valsHandle.end(),
        [&rangeValue, this](const auto &mem) {
          return this->CheckOverlappingValue(
              mem.first, rangeValue.fromVal, rangeValue.toVal);
        })};
    if (selectCaseValsIter != valsHandle.end()) {
      ReportOverlapCaseValError(source, selectCaseValsIter->second,
          GetValuePairAsString(lowerVal, upperVal).c_str());
    }
    auto selectCaseRangeValsIter{std::find_if(rangeValsHandle.begin(),
        rangeValsHandle.end(), [&rangeValue, this](const auto &mem) {
          return ((!mem.fromVal && !rangeValue.fromVal) ||
              (!mem.toVal && !rangeValue.toVal) ||
              this->CheckOverlappingRangeValue(mem.fromVal, mem.toVal,
                  rangeValue.fromVal, rangeValue.toVal));
        })};
    if (selectCaseRangeValsIter == rangeValsHandle.end()) {
      rangeValsHandle.push_back(rangeValue);
    } else {
      ReportOverlapCaseValError(source, selectCaseRangeValsIter->source,
          GetValuePairAsString(lowerVal, upperVal).c_str());
    }
  }
  template<typename T> void InsertString(const T &x, parser::CharBlock source) {
    if (kind_ == sizeof(std::string::value_type)) {
      if (auto val{GetStringValue(x)}) {
        Insert(val.value(), source);
      }
    } else if (kind_ == sizeof(std::u16string::value_type)) {
      if (auto val{GetStringValue<std::u16string>(x)}) {
        Insert(val.value(), source);
      }
    } else if (kind_ == sizeof(std::u32string::value_type)) {
      if (auto val{GetStringValue<std::u32string>(x)}) {
        Insert(val.value(), source);
      }
    } else {
      messages_.Say(source,
          "Character kind type of case value (=%d) is not supported"_err_en_US,
          kind_);
    }
  }
  template<typename T>
  void InsertString(const T &lower, const T &upper, parser::CharBlock source) {
    if (kind_ == sizeof(std::string::value_type)) {
      const auto lowerVal = lower ? GetStringValue(lower->thing) : std::nullopt;
      const auto upperVal = upper ? GetStringValue(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if (kind_ == sizeof(std::u16string::value_type)) {
      const auto lowerVal =
          lower ? GetStringValue<std::u16string>(lower->thing) : std::nullopt;
      const auto upperVal =
          upper ? GetStringValue<std::u16string>(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else if (kind_ == sizeof(std::u32string::value_type)) {
      const auto lowerVal =
          lower ? GetStringValue<std::u32string>(lower->thing) : std::nullopt;
      const auto upperVal =
          upper ? GetStringValue<std::u32string>(upper->thing) : std::nullopt;
      Insert(lowerVal, upperVal, source);
    } else {
      messages_.Say(source,
          "Character kind type of case value (=%d) is not supported"_err_en_US,
          kind_);
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
                "Character kind type of case construct (=%d) mismatches with the kind type of case value (=%d)"_err_en_US,
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
  template<typename T> struct SelectCaseRangeType {
    SelectCaseRangeType(std::optional<T> fromVal, std::optional<T> toVal,
        parser::CharBlock source)
      : fromVal(fromVal), toVal(toVal), source(source) {}
    std::optional<T> fromVal;
    std::optional<T> toVal;
    parser::CharBlock source;
  };
  template<typename T>
  bool IsMoreThanLowerVal(const T &strVal, const T &lowerStrVal) {
    return (strVal.compare(lowerStrVal) >= 0);
  }
  bool IsMoreThanLowerVal(
      const std::int64_t &intVal, const std::int64_t &lowerIntVal) {
    return intVal >= lowerIntVal;
  }
  template<typename T>
  bool IsLessThanUpperVal(const T &strVal, const T &upperStrVal) {
    return (strVal.compare(upperStrVal) <= 0);
  }
  bool IsLessThanUpperVal(
      const std::int64_t &intVal, const std::int64_t &upperIntVal) {
    return intVal <= upperIntVal;
  }
  template<typename T>
  bool CheckOverlappingValue(
      const T &val, std::optional<T> lowerVal, std::optional<T> upperVal) {
    return ((!lowerVal || IsMoreThanLowerVal(val, lowerVal.value())) &&
        (!upperVal || IsLessThanUpperVal(val, upperVal.value())));
  }
  template<typename T>
  bool CheckOverlappingRangeValue(std::optional<T> memberFromVal,
      std::optional<T> memberToVal, std::optional<T> rangeFromVal,
      std::optional<T> rangeToVal) {
    return ((memberFromVal &&
                CheckOverlappingValue(
                    memberFromVal.value(), rangeFromVal, rangeToVal)) ||
        (memberToVal &&
            CheckOverlappingValue(
                memberToVal.value(), rangeFromVal, rangeToVal)) ||
        (rangeFromVal &&
            CheckOverlappingValue(
                rangeFromVal.value(), memberFromVal, memberToVal)) ||
        (rangeToVal &&
            CheckOverlappingValue(
                rangeToVal.value(), memberFromVal, memberToVal)));
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::int64_t>::value,
      std::map<std::int64_t, parser::CharBlock> &>::type
  GetValHandle() {
    return selectCaseIntVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::string>::value,
      std::map<std::string, parser::CharBlock> &>::type
  GetValHandle() {
    return selectCaseStringVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::u16string>::value,
      std::map<std::u16string, parser::CharBlock> &>::type
  GetValHandle() {
    return selectCaseU16StringVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::u32string>::value,
      std::map<std::u32string, parser::CharBlock> &>::type
  GetValHandle() {
    return selectCaseU32StringVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::int64_t>::value,
      std::vector<SelectCaseRangeType<std::int64_t>> &>::type
  GetRangeValHandle() {
    return selectCaseIntRangeVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::string>::value,
      std::vector<SelectCaseRangeType<std::string>> &>::type
  GetRangeValHandle() {
    return selectCaseStringRangeVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::u16string>::value,
      std::vector<SelectCaseRangeType<std::u16string>> &>::type
  GetRangeValHandle() {
    return selectCaseU16StringRangeVals_;
  }
  template<typename T>
  typename std::enable_if<std::is_same<T, std::u32string>::value,
      std::vector<SelectCaseRangeType<std::u32string>> &>::type
  GetRangeValHandle() {
    return selectCaseU32StringRangeVals_;
  }

  std::map<bool, parser::CharBlock> selectCaseLogicalVals_;
  std::map<std::int64_t, parser::CharBlock> selectCaseIntVals_;
  std::map<std::string, parser::CharBlock> selectCaseStringVals_;
  std::map<std::u16string, parser::CharBlock> selectCaseU16StringVals_;
  std::map<std::u32string, parser::CharBlock> selectCaseU32StringVals_;
  std::vector<SelectCaseRangeType<std::int64_t>> selectCaseIntRangeVals_;
  std::vector<SelectCaseRangeType<std::string>> selectCaseStringRangeVals_;
  std::vector<SelectCaseRangeType<std::u16string>>
      selectCaseU16StringRangeVals_;
  std::vector<SelectCaseRangeType<std::u32string>>
      selectCaseU32StringRangeVals_;
  int kind_;
  TypeCategory category_;
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
  std::optional<TypeCategory> validCaseStmtTypeCategory;
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
    validCaseStmtTypeCategory = selectCaseStmtType.value().category();
  }

  const auto &caseList{
      std::get<std::list<parser::CaseConstruct::Case>>(selectCaseConstruct.t)};
  bool defaultCaseFound{false};
  SelectCaseHelper selectCaseStmts(context_.messages());
  if (validCaseStmtTypeCategory) {
    selectCaseStmts.SetKind(selectCaseStmtType->kind());
    selectCaseStmts.SetTypeCategory(*validCaseStmtTypeCategory);
  }

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
            "Not more than one of the selectors of SELECT CASE statement may be DEFAULT"_err_en_US);
      }
    } else if (validCaseStmtTypeCategory) {
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

          if (*validCaseStmtTypeCategory == TypeCategory::Integer) {
            selectCaseStmts.Insert(
                GetIntValue(constCase->thing).value(), caseStmt.source);
          } else if (*validCaseStmtTypeCategory == TypeCategory::Character) {
            selectCaseStmts.InsertString(constCase->thing, caseStmt.source);
          } else {  // TypeCategory::Logical
            selectCaseStmts.Insert(
                GetBoolValue(constCase->thing).value(), caseStmt.source);
          }
        } else {
          if (*validCaseStmtTypeCategory ==
              TypeCategory::Logical) {  // C1148 (R1140)
            context_.Say(caseStmt.source,
                "SELECT CASE expression of type LOGICAL must not have range of case value"_err_en_US);
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

          if (*validCaseStmtTypeCategory == TypeCategory::Integer) {
            const auto lowerVal =
                lower ? GetIntValue(lower->thing) : std::nullopt;
            const auto upperVal =
                upper ? GetIntValue(upper->thing) : std::nullopt;
            selectCaseStmts.Insert(lowerVal, upperVal, caseStmt.source);
          } else {  // TypeCategory::Character
            selectCaseStmts.InsertString(lower, upper, caseStmt.source);
          }
        }
      }
    }
  }
}
}  // namespace Fortran::semantics
