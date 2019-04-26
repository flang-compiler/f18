// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "check-io.h"
#include "expression.h"
#include "tools.h"
#include "../parser/tools.h"

namespace Fortran::semantics {

// TODO: C1234, C1235 -- defined I/O constraints

void IoChecker::Enter(const parser::ConnectSpec &spec) {
  // ConnectSpec context FileNameExpr
  if (std::get_if<parser::FileNameExpr>(&spec.u)) {
    SetSpecifier(SpecifierKind::File);
  }
}

void IoChecker::Enter(const parser::ConnectSpec::CharExpr &spec) {
  SpecifierKind specKind{};
  using ParseKind = parser::ConnectSpec::CharExpr::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Access: specKind = SpecifierKind::Access; break;
  case ParseKind::Action: specKind = SpecifierKind::Action; break;
  case ParseKind::Asynchronous: specKind = SpecifierKind::Asynchronous; break;
  case ParseKind::Blank: specKind = SpecifierKind::Blank; break;
  case ParseKind::Decimal: specKind = SpecifierKind::Decimal; break;
  case ParseKind::Delim: specKind = SpecifierKind::Delim; break;
  case ParseKind::Encoding: specKind = SpecifierKind::Encoding; break;
  case ParseKind::Form: specKind = SpecifierKind::Form; break;
  case ParseKind::Pad: specKind = SpecifierKind::Pad; break;
  case ParseKind::Position: specKind = SpecifierKind::Position; break;
  case ParseKind::Round: specKind = SpecifierKind::Round; break;
  case ParseKind::Sign: specKind = SpecifierKind::Sign; break;
  case ParseKind::Convert: specKind = SpecifierKind::Convert; break;
  case ParseKind::Dispose: specKind = SpecifierKind::Dispose; break;
  }
  SetSpecifier(specKind);
  if (const std::optional<std::string> charConst{GetConstExpr<std::string>(
          std::get<parser::ScalarDefaultCharExpr>(spec.t))}) {
    std::string s{parser::ToUpperCaseLetters(*charConst)};
    if (specKind == SpecifierKind::Access) {
      flag_.set(Flag::KnownAccess);
      flag_.set(Flag::AccessDirect, s == "DIRECT");
      flag_.set(Flag::AccessStream, s == "STREAM");
    }
    CheckStringValue(specKind, *charConst, parser::FindSourceLocation(spec));
  }
}

void IoChecker::Enter(const parser::ConnectSpec::Newunit &) {
  SetSpecifier(SpecifierKind::Newunit);
}

void IoChecker::Enter(const parser::ConnectSpec::Recl &spec) {
  SetSpecifier(SpecifierKind::Recl);
  if (const std::optional<std::int64_t> recl{
          GetConstExpr<std::int64_t>(spec)}) {
    if (*recl <= 0) {
      context_.Say(parser::FindSourceLocation(spec),
          "RECL value (%jd) must be positive"_err_en_US,
          std::move(static_cast<std::intmax_t>(*recl)));  // 12.5.6.15
    }
  }
}

void IoChecker::Enter(const parser::EndLabel &spec) {
  SetSpecifier(SpecifierKind::End);
}

void IoChecker::Enter(const parser::EorLabel &spec) {
  SetSpecifier(SpecifierKind::Eor);
}

void IoChecker::Enter(const parser::ErrLabel &spec) {
  SetSpecifier(SpecifierKind::Err);
}

void IoChecker::Enter(const parser::FileUnitNumber &spec) {
  SetSpecifier(SpecifierKind::Unit);
  flag_.set(Flag::NumberUnit);
}

void IoChecker::Enter(const parser::Format &spec) {
  SetSpecifier(SpecifierKind::Fmt);
  flag_.set(Flag::FmtOrNml);
  if (std::get_if<parser::Star>(&spec.u)) {
    flag_.set(Flag::StarFmt);
  } else if (std::get_if<parser::Label>(&spec.u)) {
    // Format statement format should be validated elsewhere.
    flag_.set(Flag::LabelFmt);
  } else {
    flag_.set(Flag::CharFmt);
    // TODO: validate compile-time constant format -- 12.6.2.2
  }
}

void IoChecker::Enter(const parser::IdExpr &spec) {
  SetSpecifier(SpecifierKind::Id);
}

void IoChecker::Enter(const parser::IdVariable &spec) {
  SetSpecifier(SpecifierKind::Id);
  auto expr{GetExpr(spec)};
  if (expr == nullptr) {
    return;
  }
  int kind{expr->GetType()->kind};
  int defaultKind{
      context_.defaultKinds().GetDefaultKind(TypeCategory::Integer)};
  if (kind < defaultKind) {
    context_.Say(
        "ID kind (%d) is smaller than default INTEGER kind (%d)"_err_en_US,
        std::move(kind), std::move(defaultKind));  // C1229
  }
}

void IoChecker::Enter(const parser::InputItem &spec) {
  flag_.set(Flag::DataList);
  if (const parser::Variable * var{std::get_if<parser::Variable>(&spec.u)}) {
    const parser::Name &name{GetLastName(*var)};
    if (auto *details{name.symbol->detailsIf<ObjectEntityDetails>()}) {
      // TODO: Determine if this check is needed at all, and if so, replace
      // the false subcondition with a check for a whole array.  Otherwise,
      // the check incorrectly flags array element and section references.
      if (details->IsAssumedSize() && false) {
        // This check may be superseded by C928 or C1002.
        context_.Say(name.source,
            "'%s' must not be a whole assumed size array"_err_en_US,
            name.ToString().c_str());  // C1231
      }
    }
  }
}

void IoChecker::Enter(const parser::InquireSpec &spec) {
  // InquireSpec context FileNameExpr
  if (std::get_if<parser::FileNameExpr>(&spec.u)) {
    SetSpecifier(SpecifierKind::File);
  }
}

void IoChecker::Enter(const parser::InquireSpec::CharVar &spec) {
  SpecifierKind specKind{};
  using ParseKind = parser::InquireSpec::CharVar::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Access: specKind = SpecifierKind::Access; break;
  case ParseKind::Action: specKind = SpecifierKind::Action; break;
  case ParseKind::Asynchronous: specKind = SpecifierKind::Asynchronous; break;
  case ParseKind::Blank: specKind = SpecifierKind::Blank; break;
  case ParseKind::Decimal: specKind = SpecifierKind::Decimal; break;
  case ParseKind::Delim: specKind = SpecifierKind::Delim; break;
  case ParseKind::Direct: specKind = SpecifierKind::Direct; break;
  case ParseKind::Encoding: specKind = SpecifierKind::Encoding; break;
  case ParseKind::Form: specKind = SpecifierKind::Form; break;
  case ParseKind::Formatted: specKind = SpecifierKind::Formatted; break;
  case ParseKind::Iomsg: specKind = SpecifierKind::Iomsg; break;
  case ParseKind::Name: specKind = SpecifierKind::Name; break;
  case ParseKind::Pad: specKind = SpecifierKind::Pad; break;
  case ParseKind::Position: specKind = SpecifierKind::Position; break;
  case ParseKind::Read: specKind = SpecifierKind::Read; break;
  case ParseKind::Readwrite: specKind = SpecifierKind::Readwrite; break;
  case ParseKind::Round: specKind = SpecifierKind::Round; break;
  case ParseKind::Sequential: specKind = SpecifierKind::Sequential; break;
  case ParseKind::Sign: specKind = SpecifierKind::Sign; break;
  case ParseKind::Status: specKind = SpecifierKind::Status; break;
  case ParseKind::Stream: specKind = SpecifierKind::Stream; break;
  case ParseKind::Unformatted: specKind = SpecifierKind::Unformatted; break;
  case ParseKind::Write: specKind = SpecifierKind::Write; break;
  case ParseKind::Convert: specKind = SpecifierKind::Convert; break;
  case ParseKind::Dispose: specKind = SpecifierKind::Dispose; break;
  }
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::InquireSpec::IntVar &spec) {
  SpecifierKind specKind{};
  using ParseKind = parser::InquireSpec::IntVar::Kind;
  switch (std::get<parser::InquireSpec::IntVar::Kind>(spec.t)) {
  case ParseKind::Iostat: specKind = SpecifierKind::Iostat; break;
  case ParseKind::Nextrec: specKind = SpecifierKind::Nextrec; break;
  case ParseKind::Number: specKind = SpecifierKind::Number; break;
  case ParseKind::Pos: specKind = SpecifierKind::Pos; break;
  case ParseKind::Recl: specKind = SpecifierKind::Recl; break;
  case ParseKind::Size: specKind = SpecifierKind::Size; break;
  }
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::InquireSpec::LogVar &spec) {
  SpecifierKind specKind{};
  using ParseKind = parser::InquireSpec::LogVar::Kind;
  switch (std::get<parser::InquireSpec::LogVar::Kind>(spec.t)) {
  case ParseKind::Exist: specKind = SpecifierKind::Exist; break;
  case ParseKind::Named: specKind = SpecifierKind::Named; break;
  case ParseKind::Opened: specKind = SpecifierKind::Opened; break;
  case ParseKind::Pending: specKind = SpecifierKind::Pending; break;
  }
  SetSpecifier(specKind);
}

void IoChecker::Enter(const parser::IoControlSpec &spec) {
  // IoControlSpec context Name
  flag_.set(Flag::IoControlList);
  if (const parser::Name * name{std::get_if<parser::Name>(&spec.u)}) {
    CHECK(name->symbol != nullptr && name->symbol->has<NamelistDetails>());
    SetSpecifier(SpecifierKind::Nml);
    flag_.set(Flag::FmtOrNml);
  }
}

void IoChecker::Enter(const parser::IoControlSpec::Asynchronous &spec) {
  SetSpecifier(SpecifierKind::Asynchronous);
  if (const std::optional<std::string> charConst{
          GetConstExpr<std::string>(spec)}) {
    flag_.set(
        Flag::AsynchronousYes, parser::ToUpperCaseLetters(*charConst) == "YES");
    CheckStringValue(SpecifierKind::Asynchronous, *charConst,
        parser::FindSourceLocation(spec));  // C1223
  }
}

void IoChecker::Enter(const parser::IoControlSpec::CharExpr &spec) {
  SpecifierKind specKind{};
  using ParseKind = parser::IoControlSpec::CharExpr::Kind;
  switch (std::get<ParseKind>(spec.t)) {
  case ParseKind::Advance: specKind = SpecifierKind::Advance; break;
  case ParseKind::Blank: specKind = SpecifierKind::Blank; break;
  case ParseKind::Decimal: specKind = SpecifierKind::Decimal; break;
  case ParseKind::Delim: specKind = SpecifierKind::Delim; break;
  case ParseKind::Pad: specKind = SpecifierKind::Pad; break;
  case ParseKind::Round: specKind = SpecifierKind::Round; break;
  case ParseKind::Sign: specKind = SpecifierKind::Sign; break;
  }
  SetSpecifier(specKind);
  if (const std::optional<std::string> charConst{GetConstExpr<std::string>(
          std::get<parser::ScalarDefaultCharExpr>(spec.t))}) {
    if (specKind == SpecifierKind::Advance) {
      flag_.set(
          Flag::AdvanceYes, parser::ToUpperCaseLetters(*charConst) == "YES");
    }
    CheckStringValue(specKind, *charConst, parser::FindSourceLocation(spec));
  }
}

void IoChecker::Enter(const parser::IoControlSpec::Pos &spec) {
  SetSpecifier(SpecifierKind::Pos);
}

void IoChecker::Enter(const parser::IoControlSpec::Rec &spec) {
  SetSpecifier(SpecifierKind::Rec);
}

void IoChecker::Enter(const parser::IoControlSpec::Size &spec) {
  SetSpecifier(SpecifierKind::Size);
}

void IoChecker::Enter(const parser::IoUnit &spec) {
  if (const parser::Variable * var{std::get_if<parser::Variable>(&spec.u)}) {
    // TODO: C1201 - internal file variable must not be an array section ...
    auto expr{GetExpr(*var)};
    int defaultKind{
        context_.defaultKinds().GetDefaultKind(TypeCategory::Character)};
    if (expr && expr->GetType() && expr->GetType()->kind != defaultKind) {
      // This may be too restrictive; other kinds may be valid.
      context_.Say(  // C1202
          "invalid character kind for an internal file variable"_err_en_US);
    }
    SetSpecifier(SpecifierKind::Unit);
    flag_.set(Flag::InternalUnit);
  } else if (std::get_if<parser::Star>(&spec.u)) {
    SetSpecifier(SpecifierKind::Unit);
    flag_.set(Flag::StarUnit);
  }
}

void IoChecker::Enter(const parser::MsgVariable &spec) {
  SetSpecifier(SpecifierKind::Iomsg);
}

void IoChecker::Enter(const parser::OutputItem &spec) {
  flag_.set(Flag::DataList);
  // TODO: C1233 - output item must not be a procedure pointer
}

void IoChecker::Enter(const parser::StatusExpr &spec) {
  SetSpecifier(SpecifierKind::Status);
  if (const std::optional<std::string> charConst{
          GetConstExpr<std::string>(spec)}) {
    // Status values for Open and Close are different.
    std::string s{parser::ToUpperCaseLetters(*charConst)};
    if (stmt_ == IoStmtKind::Open) {
      flag_.set(Flag::KnownStatus);
      flag_.set(Flag::StatusNew, s == "NEW");
      flag_.set(Flag::StatusReplace, s == "REPLACE");
      flag_.set(Flag::StatusScratch, s == "SCRATCH");
      // CheckStringValue compares for OPEN Status string values.
      CheckStringValue(
          SpecifierKind::Status, *charConst, parser::FindSourceLocation(spec));
      return;
    }
    CHECK(stmt_ == IoStmtKind::Close);
    if (s != "DELETE" && s != "KEEP") {
      context_.Say(parser::FindSourceLocation(spec),
          "invalid STATUS value '%s'"_err_en_US, (*charConst).c_str());
    }
  }
}

void IoChecker::Enter(const parser::StatVariable &spec) {
  SetSpecifier(SpecifierKind::Iostat);
}

void IoChecker::Leave(const parser::BackspaceStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1240
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::CloseStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1208
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::EndfileStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1240
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::FlushStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1243
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::InquireStmt &stmt) {
  if (std::get_if<std::list<parser::InquireSpec>>(&stmt.u)) {
    // Inquire by unit or by file (vs. by output list).
    CheckForRequiredSpecifier(
        flag_.test(Flag::NumberUnit) || specifierSet_.test(SpecifierKind::File),
        "UNIT number or FILE");  // C1246
    CheckForProhibitedSpecifier(SpecifierKind::File,
        SpecifierKind::Unit);  // C1246
    CheckForRequiredSpecifier(
        SpecifierKind::Id, SpecifierKind::Pending);  // C1248
  }
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::OpenStmt &stmt) {
  CheckForRequiredSpecifier(specifierSet_.test(SpecifierKind::Unit) ||
          specifierSet_.test(SpecifierKind::Newunit),
      "UNIT or NEWUNIT");  // C1204, C1205
  CheckForProhibitedSpecifier(
      SpecifierKind::Newunit, SpecifierKind::Unit);  // C1204, C1205
  CheckForRequiredSpecifier(flag_.test(Flag::StatusNew), "STATUS='NEW'",
      SpecifierKind::File);  // 12.5.6.10
  CheckForRequiredSpecifier(flag_.test(Flag::StatusReplace), "STATUS='REPLACE'",
      SpecifierKind::File);  // 12.5.6.10
  CheckForProhibitedSpecifier(flag_.test(Flag::StatusScratch),
      "STATUS='SCRATCH'", SpecifierKind::File);  // 12.5.6.10
  if (flag_.test(Flag::KnownStatus)) {
    CheckForRequiredSpecifier(SpecifierKind::Newunit,
        specifierSet_.test(SpecifierKind::File) ||
            flag_.test(Flag::StatusScratch),
        "FILE or STATUS='SCRATCH'");  // 12.5.6.12
  } else {
    CheckForRequiredSpecifier(SpecifierKind::Newunit,
        specifierSet_.test(SpecifierKind::File) ||
            specifierSet_.test(SpecifierKind::Status),
        "FILE or STATUS");  // 12.5.6.12
  }
  if (flag_.test(Flag::KnownAccess)) {
    CheckForRequiredSpecifier(flag_.test(Flag::AccessDirect), "ACCESS='DIRECT'",
        SpecifierKind::Recl);  // 12.5.6.15
    CheckForProhibitedSpecifier(flag_.test(Flag::AccessStream),
        "STATUS='STREAM'", SpecifierKind::Recl);  // 12.5.6.15
  }
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::ReadStmt &stmt) {
  if (!flag_.test(Flag::IoControlList)) {
    return;
  }
  LeaveReadWrite();
  CheckForProhibitedSpecifier(SpecifierKind::Delim);  // C1212
  CheckForProhibitedSpecifier(SpecifierKind::Sign);  // C1212
  CheckForProhibitedSpecifier(SpecifierKind::Rec, SpecifierKind::End);  // C1220
  CheckForRequiredSpecifier(SpecifierKind::Eor,
      specifierSet_.test(SpecifierKind::Advance) &&
          !flag_.test(Flag::AdvanceYes),
      "ADVANCE with value 'NO'");  // C1222 + 12.6.2.1p2
  CheckForRequiredSpecifier(
      SpecifierKind::Blank, flag_.test(Flag::FmtOrNml), "FMT or NML");  // C1227
  CheckForRequiredSpecifier(
      SpecifierKind::Pad, flag_.test(Flag::FmtOrNml), "FMT or NML");  // C1227
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::RewindStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1240
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::WaitStmt &stmt) {
  CheckForRequiredSpecifier(
      flag_.test(Flag::NumberUnit), "UNIT number");  // C1237
  stmt_ = IoStmtKind::None;
}

void IoChecker::Leave(const parser::WriteStmt &stmt) {
  LeaveReadWrite();
  CheckForProhibitedSpecifier(SpecifierKind::Blank);  // C1213
  CheckForProhibitedSpecifier(SpecifierKind::End);  // C1213
  CheckForProhibitedSpecifier(SpecifierKind::Eor);  // C1213
  CheckForProhibitedSpecifier(SpecifierKind::Pad);  // C1213
  CheckForProhibitedSpecifier(SpecifierKind::Size);  // C1213
  CheckForRequiredSpecifier(
      SpecifierKind::Sign, flag_.test(Flag::FmtOrNml), "FMT or NML");  // C1227
  CheckForRequiredSpecifier(SpecifierKind::Delim,
      flag_.test(Flag::StarFmt) || specifierSet_.test(SpecifierKind::Nml),
      "FMT=* or NML");  // C1228
  stmt_ = IoStmtKind::None;
}

void IoChecker::LeaveReadWrite() const {
  CheckForRequiredSpecifier(SpecifierKind::Unit);  // C1211
  CheckForProhibitedSpecifier(SpecifierKind::Nml, SpecifierKind::Rec);  // C1216
  CheckForProhibitedSpecifier(SpecifierKind::Nml, SpecifierKind::Fmt);  // C1216
  CheckForProhibitedSpecifier(
      SpecifierKind::Nml, flag_.test(Flag::DataList), "a data list");  // C1216
  CheckForProhibitedSpecifier(flag_.test(Flag::InternalUnit),
      "UNIT=internal-file", SpecifierKind::Pos);  // C1219
  CheckForProhibitedSpecifier(flag_.test(Flag::InternalUnit),
      "UNIT=internal-file", SpecifierKind::Rec);  // C1219
  CheckForProhibitedSpecifier(
      flag_.test(Flag::StarUnit), "UNIT=*", SpecifierKind::Pos);  // C1219
  CheckForProhibitedSpecifier(
      flag_.test(Flag::StarUnit), "UNIT=*", SpecifierKind::Rec);  // C1219
  CheckForProhibitedSpecifier(
      SpecifierKind::Rec, flag_.test(Flag::StarFmt), "FMT=*");  // C1220
  CheckForRequiredSpecifier(SpecifierKind::Advance,
      flag_.test(Flag::CharFmt) || flag_.test(Flag::LabelFmt),
      "an explicit format");  // C1221
  CheckForProhibitedSpecifier(SpecifierKind::Advance,
      flag_.test(Flag::InternalUnit), "UNIT=internal-file");  // C1221
  CheckForRequiredSpecifier(flag_.test(Flag::AsynchronousYes),
      "ASYNCHRONOUS='YES'", flag_.test(Flag::NumberUnit),
      "UNIT=number");  // C1224
  CheckForRequiredSpecifier(SpecifierKind::Id,
      flag_.test(Flag::AsynchronousYes), "ASYNCHRONOUS='YES'");  // C1225
  CheckForProhibitedSpecifier(SpecifierKind::Pos, SpecifierKind::Rec);  // C1226
  CheckForRequiredSpecifier(SpecifierKind::Decimal, flag_.test(Flag::FmtOrNml),
      "FMT or NML");  // C1227
  CheckForRequiredSpecifier(
      SpecifierKind::Round, flag_.test(Flag::FmtOrNml), "FMT or NML");  // C1227
}

void IoChecker::SetSpecifier(SpecifierKind specKind) {
  if (stmt_ == IoStmtKind::None) {
    // FMT may appear on PRINT statements, which don't have any checks.
    // [IO]MSG and [IO]STAT parse symbols are shared with non-I/O statements.
    return;
  }
  // C1203, C1207, C1210, C1236, C1239, C1242, C1245
  if (specifierSet_.test(specKind)) {
    context_.Say("duplicate %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str());
  }
  specifierSet_.set(specKind);
}

void IoChecker::CheckStringValue(SpecifierKind specKind,
    const std::string &value, const parser::CharBlock &source) const {
  static std::unordered_map<SpecifierKind, const std::set<std::string>>
      specValues{
          {SpecifierKind::Access, {"DIRECT", "SEQUENTIAL", "STREAM"}},
          {SpecifierKind::Action, {"READ", "READWRITE", "WRITE"}},
          {SpecifierKind::Advance, {"NO", "YES"}},
          {SpecifierKind::Asynchronous, {"NO", "YES"}},
          {SpecifierKind::Blank, {"NULL", "ZERO"}},
          {SpecifierKind::Decimal, {"COMMA", "POINT"}},
          {SpecifierKind::Delim, {"APOSTROPHE", "NONE", "QUOTE"}},
          {SpecifierKind::Encoding, {"DEFAULT", "UTF-8"}},
          {SpecifierKind::Form, {"FORMATTED", "UNFORMATTED"}},
          {SpecifierKind::Pad, {"NO", "YES"}},
          {SpecifierKind::Position, {"APPEND", "ASIS", "REWIND"}},
          {SpecifierKind::Round,
              {"COMPATIBLE", "DOWN", "NEAREST", "PROCESSOR_DEFINED", "UP",
                  "ZERO"}},
          {SpecifierKind::Sign, {"PLUS", "PROCESSOR_DEFINED", "SUPPRESS"}},
          {SpecifierKind::Status,
              // Open values; Close values are different ("DELETE", "KEEP").
              {"NEW", "OLD", "REPLACE", "SCRATCH", "UNKNOWN"}},
          {SpecifierKind::Convert, {"BIG_ENDIAN", "LITTLE_ENDIAN", "NATIVE"}},
          {SpecifierKind::Dispose, {"DELETE", "KEEP"}},
      };
  if (!specValues.at(specKind).count(parser::ToUpperCaseLetters(value))) {
    context_.Say(source, "invalid %s value '%s'"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str(),
        value.c_str());
  }
}

// CheckForRequiredSpecifier and CheckForProhibitedSpecifier functions
// need conditions to check, and string arguments to insert into a message.
// A SpecifierKind provides both an absence/presence condition and a string
// argument (its name).  A (condition, string) pair provides an arbitrary
// condition and an arbitrary string.

void IoChecker::CheckForRequiredSpecifier(SpecifierKind specKind) const {
  if (!specifierSet_.test(specKind)) {
    context_.Say("%s statement must have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(stmt_)).c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str());
  }
}

void IoChecker::CheckForRequiredSpecifier(
    bool condition, const std::string &s) const {
  if (!condition) {
    context_.Say("%s statement must have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(stmt_)).c_str(), s.c_str());
  }
}

void IoChecker::CheckForRequiredSpecifier(
    SpecifierKind specKind1, SpecifierKind specKind2) const {
  if (specifierSet_.test(specKind1) && !specifierSet_.test(specKind2)) {
    context_.Say("if %s appears, %s must also appear"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind1)).c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind2)).c_str());
  }
}

void IoChecker::CheckForRequiredSpecifier(
    SpecifierKind specKind, bool condition, const std::string &s) const {
  if (specifierSet_.test(specKind) && !condition) {
    context_.Say("if %s appears, %s must also appear"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str(), s.c_str());
  }
}

void IoChecker::CheckForRequiredSpecifier(
    bool condition, const std::string &s, SpecifierKind specKind) const {
  if (condition && !specifierSet_.test(specKind)) {
    context_.Say("if %s appears, %s must also appear"_err_en_US, s.c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str());
  }
}

void IoChecker::CheckForRequiredSpecifier(bool condition1,
    const std::string &s1, bool condition2, const std::string &s2) const {
  if (condition1 && !condition2) {
    context_.Say(
        "if %s appears, %s must also appear"_err_en_US, s1.c_str(), s2.c_str());
  }
}

void IoChecker::CheckForProhibitedSpecifier(SpecifierKind specKind) const {
  if (specifierSet_.test(specKind)) {
    context_.Say("%s statement must not have a %s specifier"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(stmt_)).c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str());
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    SpecifierKind specKind1, SpecifierKind specKind2) const {
  if (specifierSet_.test(specKind1) && specifierSet_.test(specKind2)) {
    context_.Say("if %s appears, %s must not appear"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind1)).c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind2)).c_str());
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    SpecifierKind specKind, bool condition, const std::string &s) const {
  if (specifierSet_.test(specKind) && condition) {
    context_.Say("if %s appears, %s must not appear"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str(), s.c_str());
  }
}

void IoChecker::CheckForProhibitedSpecifier(
    bool condition, const std::string &s, SpecifierKind specKind) const {
  if (condition && specifierSet_.test(specKind)) {
    context_.Say("if %s appears, %s must not appear"_err_en_US, s.c_str(),
        parser::ToUpperCaseLetters(EnumToString(specKind)).c_str());
  }
}

}  // namespace Fortran::semantics
