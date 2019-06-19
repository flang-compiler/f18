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

#ifndef FORTRAN_COMMON_FORMAT_H_
#define FORTRAN_COMMON_FORMAT_H_

#include "Fortran.h"
#include <cstring>

// Define a FormatValidator class template to validate a format expression.
// To enable use in runtime library code as well as compiler code, the
// implementation does its own parsing without recourse to compiler parser
// machinery, and avoids features that require C++ runtime library support.
// A format expression is a pointer to a fixed size character of some kind,
// with an explicit length.  Class function Check analyzes the expression
// for syntax and semantic errors and warnings, and returns up to a caller-
// chosen number of messages in a caller-allocated array of FormatMessage
// structs.  If the context is a READ, WRITE, or PRINT statement, rather
// than a FORMAT statement, statement-specific checks are also done.

namespace Fortran::common {

struct FormatMessage {
  static constexpr int maxArgLength{25};

  const char *text;  // message text; may have one %s argument
  char arg[maxArgLength + 1];  // optional %s argument value
  int offset;  // offset to message marker
  int length;  // length of message marker
  bool error;  // vs. warning
};

template<typename CHAR = char> class FormatValidator {
public:
  explicit FormatValidator(const CHAR *format, size_t length, IoStmtKind stmt,
      bool warnOnNonstandardUsage, FormatMessage *messages, int maxMessages)
    : format_{format}, end_{format ? format + length : nullptr}, stmt_{stmt},
      warnOnNonstandardUsage_{warnOnNonstandardUsage}, messages_{messages},
      maxMessages_{maxMessages}, cursor_{format - 1} {}

  int Check();

private:
  enum class TokenKind {
    None,
    A,
    B,
    BN,
    BZ,
    D,
    DC,
    DP,
    DT,
    E,
    EN,
    ES,
    EX,
    F,
    G,
    I,
    L,
    O,
    P,
    RC,
    RD,
    RN,
    RP,
    RU,
    RZ,
    S,
    SP,
    SS,
    T,
    TL,
    TR,
    X,
    Z,
    Colon,
    Slash,
    Backslash,  // nonstandard: inhibit newline on output
    Dollar,  // nonstandard: inhibit newline on output on terminals
    Star,
    LParen,
    RParen,
    Comma,
    Point,
    Sign,
    UnsignedInteger,  // value in integerValue_
    String,  // char-literal-constant or Hollerith constant
  };

  struct Token {
    Token &set_kind(TokenKind kind) {
      kind_ = kind;
      return *this;
    }
    Token &set_offset(int offset) {
      offset_ = offset;
      return *this;
    }
    Token &set_length(int length) {
      length_ = length;
      return *this;
    }

    TokenKind kind() const { return kind_; }
    int offset() const { return offset_; }
    int length() const { return length_; }

    bool IsSet() { return kind_ != TokenKind::None; }

  private:
    TokenKind kind_{TokenKind::None};
    int offset_{0};
    int length_{1};
  };

  void AppendWarning(const char *text) { AppendWarning(text, token_); }
  void AppendError(const char *text) { AppendError(text, token_); }

  void AppendWarning(
      const char *text, Token &token, const char *arg = nullptr) {
    int messageIndex{messageCount_};
    AppendError(text, token, arg);
    if (messageIndex < messageCount_) {
      messages_[messageIndex].error = false;
    }
  }

  void AppendError(const char *text, Token &token, const char *arg = nullptr) {
    FormatMessage *message{
        messageCount_ ? messages_ + messageCount_ - 1 : nullptr};
    if (messageCount_ &&
        (suppressMessageCascade_ || messageCount_ >= maxMessages_ ||
            message->offset == token.offset())) {
      return;
    }
    message = messages_ + messageCount_;
    message->text = text;
    message->offset = token.offset();
    message->length = token.length();
    message->error = true;
    strncpy(message->arg, arg ? arg : argString_, FormatMessage::maxArgLength);
    CHECK(message->arg[FormatMessage::maxArgLength - 1] == 0);
    ++messageCount_;
    suppressMessageCascade_ = true;
  }

  CHAR NextChar();
  CHAR LookAheadChar();
  void Advance(TokenKind);
  void NextToken();

  void check_r(bool allowed = true);
  bool check_w(bool required = true);
  void check_m();
  bool check_d(bool required = true);
  void check_e();

  const CHAR *const format_;  // format text
  const CHAR *const end_;  // one-past-last of format_ text
  IoStmtKind stmt_;
  bool warnOnNonstandardUsage_;
  FormatMessage *messages_;
  int maxMessages_;
  int messageCount_{0};

  const CHAR *cursor_{};  // current location in format_
  const CHAR *laCursor_{};  // lookahead cursor
  Token token_{};  // current token
  int64_t integerValue_{-1};  // value of UnsignedInteger token
  Token knrToken_{};  // k, n, or r UnsignedInteger token
  int64_t knrValue_{-1};  // -1 ==> not present
  int64_t wValue_{-1};
  char argString_[3]{};  // 1-2 character msg arg; usually edit descriptor name
  bool suppressMessageCascade_{false};
};

extern template class FormatValidator<char>;
extern template class FormatValidator<char16_t>;
extern template class FormatValidator<char32_t>;
}
#endif  // FORTRAN_COMMON_FORMAT_H_
