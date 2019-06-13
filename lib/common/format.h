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

namespace Fortran::common {

struct FormatError {
  static constexpr int maxArgLength_{25};

  const char *text_;  // error text; may have one %s argument
  char arg_[maxArgLength_ + 1];  // optional %s argument value
  int offset_;  // offset to error marker
  int length_;  // length of error marker
};

template<typename CHAR = char> class FormatValidator {
public:
  explicit FormatValidator(const CHAR *format, size_t length, IoStmtKind stmt,
      bool warnOnNonstandardUsage, FormatError *errors, int maxErrors)
    : format_{format}, end_{format ? format + length : nullptr}, stmt_{stmt},
      warnOnNonstandardUsage_{warnOnNonstandardUsage}, errors_{errors},
      maxErrors_{maxErrors}, cursor_{format - 1} {}

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
    String,  // char-literal-constant or hollerith constant
  };

  struct Token {
    explicit Token() : kind_{TokenKind::None}, offset_{0}, length_{1} {}

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
    TokenKind kind_;
    int offset_;
    int length_;
  };

  CHAR NextChar();
  CHAR LookAheadChar();
  void Advance(TokenKind);
  void NextToken();

  void AppendError(const char *text) { AppendError(text, token_); }

  void AppendError(const char *text, Token &token, const char *arg = nullptr) {
    FormatError *error{errorCount_ ? errors_ + errorCount_ - 1 : nullptr};
    if (errorCount_ &&
        (suppressErrorCascade_ || errorCount_ >= maxErrors_ ||
            error->offset_ == token.offset())) {
      return;
    }
    error = errors_ + errorCount_;
    error->text_ = text;
    error->offset_ = token.offset();
    error->length_ = token.length();
    strncpy(error->arg_, arg ? arg : argString_, FormatError::maxArgLength_);
    CHECK(error->arg_[FormatError::maxArgLength_ - 1] == 0);
    ++errorCount_;
    suppressErrorCascade_ = true;
  }

  const CHAR *const format_;  // format text
  const CHAR *const end_;  // one-past-last of format_ text
  IoStmtKind stmt_;
  bool warnOnNonstandardUsage_;
  FormatError *errors_;
  int maxErrors_;
  int errorCount_{0};

  const CHAR *cursor_;  // current location in format_
  const CHAR *laCursor_;  // lookahead cursor
  Token token_{};  // current token
  char argString_[3]{};  // 1-2 character message arg; usually descriptor name
  int64_t integerValue_{-1};  // value of UnsignedInteger token
  bool suppressErrorCascade_{false};
};

extern template class FormatValidator<char>;
extern template class FormatValidator<char16_t>;
extern template class FormatValidator<char32_t>;

}
#endif  // FORTRAN_COMMON_FORMAT_H_
