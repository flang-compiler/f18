// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_STMT_PARSER_H_
#define FORTRAN_PARSER_STMT_PARSER_H_

// Basic parsing of statements.

#include "basic-parsers.h"
#include "token-parsers.h"

namespace Fortran::parser {

// statement(p) parses Statement<P> for some statement type P that is the
// result type of the argument parser p, while also handling labels and
// end-of-statement markers.

// R611 label -> digit [digit]...
constexpr auto label{space >> digitString64 / spaceCheck};

template<typename PA> inline constexpr auto unterminatedStatement(const PA &p) {
  return skipStuffBeforeStatement >>
      sourced(construct<Statement<typename PA::resultType>>(
          maybe(label), space >> p));
}

constexpr auto endOfLine{
    "\n"_ch >> ok || fail("expected end of line"_err_en_US)};

constexpr auto semicolons{";"_ch >> skipMany(";"_tok) / space / maybe("\n"_ch)};
constexpr auto endOfStmt{
    space >> withMessage("expected end of statement"_err_en_US,
                 semicolons || endOfLine)};
constexpr auto forceEndOfStmt{recovery(endOfStmt, SkipPast<'\n'>{})};

template<typename PA> inline constexpr auto statement(const PA &p) {
  return unterminatedStatement(p) / endOfStmt;
}

// unlabeledStatement() is basically statement() for those few situations
// in Fortran where a statement cannot have a label.
template<typename PA> inline constexpr auto unlabeledStatement(const PA &p) {
  return skipStuffBeforeStatement >>
      sourced(
          construct<UnlabeledStatement<typename PA::resultType>>(space >> p)) /
      endOfStmt;
}

// This unambiguousStatement() variant of statement() provides better error
// recovery for contexts containing statements that might have trailing
// garbage, but it must be used only when no instance of the statement in
// question could also be a legal prefix of some other statement that might
// be valid at that point.  It only makes sense to use this within "some()"
// or "many()" so as to not end the list of statements.
template<typename PA> inline constexpr auto unambiguousStatement(const PA &p) {
  return unterminatedStatement(p) / forceEndOfStmt;
}

constexpr auto ignoredStatementPrefix{
    skipStuffBeforeStatement >> maybe(label) >> maybe(name / ":") >> space};

// Error recovery within a statement() call: skip *to* the end of the line,
// unless at an END or CONTAINS statement.
constexpr auto inStmtErrorRecovery{!"END"_tok >> !"CONTAINS"_tok >>
    SkipTo<'\n'>{} >> construct<ErrorRecovery>()};

// Error recovery within statement sequences: skip *past* the end of the line,
// but not over an END or CONTAINS statement.
constexpr auto skipStmtErrorRecovery{!"END"_tok >> !"CONTAINS"_tok >>
    SkipPast<'\n'>{} >> construct<ErrorRecovery>()};

// Error recovery across statements: skip the line, unless it looks
// like it might end the containing construct.
constexpr auto stmtErrorRecoveryStart{ignoredStatementPrefix};
constexpr auto skipBadLine{SkipPast<'\n'>{} >> construct<ErrorRecovery>()};
constexpr auto executionPartErrorRecovery{stmtErrorRecoveryStart >>
    !"END"_tok >> !"CONTAINS"_tok >> !"ELSE"_tok >> !"CASE"_tok >>
    !"TYPE IS"_tok >> !"CLASS"_tok >> !"RANK"_tok >>
    !("!$OMP "_sptok >> "END"_tok) >> skipBadLine};

// END statement error recovery
constexpr auto missingOptionalName{defaulted(cut >> maybe(name))};
constexpr auto noNameEnd{"END" >> missingOptionalName};
constexpr auto atEndOfStmt{space >>
    withMessage("expected end of statement"_err_en_US, lookAhead(";\n"_ch))};
constexpr auto bareEnd{noNameEnd / recovery(atEndOfStmt, SkipTo<'\n'>{})};

constexpr auto endStmtErrorRecovery{
    ("END"_tok >> SkipTo<'\n'>{} || ok) >> missingOptionalName};

constexpr auto progUnitEndStmtErrorRecovery{
    (many(!"END"_tok >> SkipPast<'\n'>{}) >>
        ("END"_tok >> SkipTo<'\n'>{} || consumedAllInput)) >>
    missingOptionalName};
}
#endif  // FORTRAN_PARSER_STMT_PARSER_H_
