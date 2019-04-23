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

#ifndef FORTRAN_SEMANTICS_IO_H_
#define FORTRAN_SEMANTICS_IO_H_

#include "semantics.h"
#include "../common/enum-set.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

class IoChecker : public virtual BaseChecker {
public:
  explicit IoChecker(SemanticsContext &context) : context_{context} {}

  void Enter(const parser::BackspaceStmt &) { Init(IoStmtKind::Backspace); }
  void Enter(const parser::CloseStmt &) { Init(IoStmtKind::Close); }
  void Enter(const parser::EndfileStmt &) { Init(IoStmtKind::Endfile); }
  void Enter(const parser::FlushStmt &) { Init(IoStmtKind::Flush); }
  void Enter(const parser::InquireStmt &) { Init(IoStmtKind::Inquire); }
  void Enter(const parser::OpenStmt &) { Init(IoStmtKind::Open); }
  void Enter(const parser::ReadStmt &) { Init(IoStmtKind::Read); }
  void Enter(const parser::RewindStmt &) { Init(IoStmtKind::Rewind); }
  void Enter(const parser::WaitStmt &) { Init(IoStmtKind::Wait); }
  void Enter(const parser::WriteStmt &) { Init(IoStmtKind::Write); }

  void Enter(const parser::ConnectSpec &);
  void Enter(const parser::ConnectSpec::CharExpr &);
  void Enter(const parser::ConnectSpec::Newunit &);
  void Enter(const parser::ConnectSpec::Recl &);
  void Enter(const parser::EndLabel &);
  void Enter(const parser::EorLabel &);
  void Enter(const parser::ErrLabel &);
  void Enter(const parser::FileUnitNumber &);
  void Enter(const parser::Format &);
  void Enter(const parser::IdExpr &);
  void Enter(const parser::IdVariable &);
  void Enter(const parser::InputItem &);
  void Enter(const parser::InquireSpec &);
  void Enter(const parser::InquireSpec::CharVar &);
  void Enter(const parser::InquireSpec::IntVar &);
  void Enter(const parser::InquireSpec::LogVar &);
  void Enter(const parser::IoControlSpec &);
  void Enter(const parser::IoControlSpec::Asynchronous &);
  void Enter(const parser::IoControlSpec::CharExpr &);
  void Enter(const parser::IoControlSpec::Pos &);
  void Enter(const parser::IoControlSpec::Rec &);
  void Enter(const parser::IoControlSpec::Size &);
  void Enter(const parser::IoUnit &);
  void Enter(const parser::MsgVariable &);
  void Enter(const parser::OutputItem &);
  void Enter(const parser::StatusExpr &);
  void Enter(const parser::StatVariable &);

  void Leave(const parser::BackspaceStmt &);
  void Leave(const parser::CloseStmt &);
  void Leave(const parser::EndfileStmt &);
  void Leave(const parser::FlushStmt &);
  void Leave(const parser::InquireStmt &);
  void Leave(const parser::OpenStmt &);
  void Leave(const parser::ReadStmt &);
  void Leave(const parser::RewindStmt &);
  void Leave(const parser::WaitStmt &);
  void Leave(const parser::WriteStmt &);

private:
  // Nothing to check for Print, but include it anyway.
  ENUM_CLASS(IoStmtKind, None, Backspace, Close, Endfile, Flush, Inquire, Open,
      Print, Read, Rewind, Wait, Write);

  // Union of specifiers for all statements.
  ENUM_CLASS(SpecifierKind, Access, Action, Advance, Asynchronous, Blank,
      Decimal, Delim, Direct, Encoding, End, Eor, Err, Exist, File, Fmt, Form,
      Formatted, Id, Iomsg, Iostat, Name, Named, Newunit, Nextrec, Nml, Number,
      Opened, Pad, Pending, Pos, Position, Read, Readwrite, Rec, Recl, Round,
      Sequential, Sign, Size, Status, Stream, Unformatted, Unit, Write,
      Convert,  // nonstandard
      Dispose,  // nonstandard
  );

  void LeaveReadWrite() const;

  void SetSpecifier(SpecifierKind);

  void CheckStringValue(
      SpecifierKind, const std::string &, const parser::CharBlock &) const;

  void CheckForRequiredSpecifier(SpecifierKind) const;
  void CheckForRequiredSpecifier(bool, const std::string &) const;
  void CheckForRequiredSpecifier(SpecifierKind, SpecifierKind) const;
  void CheckForRequiredSpecifier(
      SpecifierKind, bool, const std::string &) const;
  void CheckForRequiredSpecifier(
      bool, const std::string &, SpecifierKind) const;
  void CheckForRequiredSpecifier(
      bool, const std::string &, bool, const std::string &) const;

  void CheckForProhibitedSpecifier(SpecifierKind) const;
  void CheckForProhibitedSpecifier(SpecifierKind, SpecifierKind) const;
  void CheckForProhibitedSpecifier(
      SpecifierKind, bool, const std::string &) const;
  void CheckForProhibitedSpecifier(
      bool, const std::string &, SpecifierKind) const;

  void Init(IoStmtKind s) {
    stmt_ = s;
    specifierSet_.reset();
    hasIoControlList_ = false;
    hasInternalUnit_ = hasNumberUnit_ = hasStarUnit_ = false;
    hasCharFormat_ = hasLabelFormat_ = hasStarFormat_ = false;
    hasKnownAccess_ = hasAccessDirect_ = hasAccessStream_ = false;
    hasAdvanceYes_ = false;
    hasAsynchronousYes_ = false;
    hasKnownStatus_ = hasStatusNew_ = hasStatusReplace_ = hasStatusScratch_ =
        false;
    hasDataList_ = false;
  }

  SemanticsContext &context_;
  IoStmtKind stmt_ = IoStmtKind::None;
  common::EnumSet<SpecifierKind, SpecifierKind_enumSize> specifierSet_;
  bool hasIoControlList_;
  bool hasInternalUnit_, hasNumberUnit_, hasStarUnit_;
  bool hasCharFormat_, hasLabelFormat_, hasStarFormat_;
  bool hasKnownAccess_, hasAccessDirect_, hasAccessStream_;
  bool hasAdvanceYes_;
  bool hasAsynchronousYes_;
  bool hasKnownStatus_, hasStatusNew_, hasStatusReplace_, hasStatusScratch_;
  bool hasDataList_;
};

}
#endif  // FORTRAN_SEMANTICS_IO_H_
