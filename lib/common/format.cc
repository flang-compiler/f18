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

#include "format.h"

namespace Fortran::common {

template<typename CHAR> CHAR FormatValidator<CHAR>::NextChar() {
  for (++cursor_; cursor_ < end_; ++cursor_) {
    if (*cursor_ != ' ') {
      return toupper(*cursor_);
    }
  }
  cursor_ = end_;
  return ' ';
};

template<typename CHAR> CHAR FormatValidator<CHAR>::LookAheadChar() {
  for (laCursor_ = cursor_ + 1; laCursor_ < end_; ++laCursor_) {
    if (*laCursor_ != ' ') {
      return toupper(*laCursor_);
    }
  }
  laCursor_ = end_;
  return ' ';
};

template<typename CHAR> void FormatValidator<CHAR>::Advance(TokenKind tk) {
  cursor_ = laCursor_;
  token_.set_kind(tk);
};

template<typename CHAR> void FormatValidator<CHAR>::NextToken() {
  // At entry, cursor_ points before the start of the next token.
  // At exit, cursor_ points to last CHAR of token_.

  CHAR c{NextChar()};
  token_.set_kind(TokenKind::None);
  token_.set_offset(cursor_ - format_);
  token_.set_length(1);
  if (c == '_' && integerValue_ >= 0) {  // C1305, C1309, C1310, C1312, C1313
    AppendError("Kind parameter '_' character in format expression");
  }
  integerValue_ = -1;

  switch (c) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9': {
    int64_t lastValue;
    const CHAR *lastCursor;
    integerValue_ = 0;
    bool overflow{false};
    do {
      lastValue = integerValue_;
      lastCursor = cursor_;
      integerValue_ = 10 * integerValue_ + c - '0';
      if (lastValue > integerValue_) {
        overflow = true;
      }
      c = NextChar();
    } while (isdigit(c));
    cursor_ = lastCursor;
    token_.set_kind(TokenKind::UnsignedInteger);
    if (overflow) {
      token_.set_length(cursor_ - format_ - token_.offset() + 1);
      AppendError("Integer overflow in format expression");
      break;
    }
    if (LookAheadChar() != 'H') {
      break;
    }
    // Hollerith constant
    if (stmt_ == IoStmtKind::Read) {
      AppendError("String edit descriptor in READ format");  // 13.3.2p6
    }
    if (laCursor_ + integerValue_ < end_) {
      token_.set_kind(TokenKind::String);
      cursor_ = laCursor_ + integerValue_;
    } else {
      token_.set_kind(TokenKind::None);
      cursor_ = end_;
      token_.set_length(cursor_ - format_ - token_.offset());
      AppendError("Unterminated Hollerith constant");
    }
    break;
  }
  case 'A': token_.set_kind(TokenKind::A); break;
  case 'B':
    switch (LookAheadChar()) {
    case 'N': Advance(TokenKind::BN); break;
    case 'Z': Advance(TokenKind::BZ); break;
    default: token_.set_kind(TokenKind::B); break;
    }
    break;
  case 'D':
    switch (LookAheadChar()) {
    case 'C': Advance(TokenKind::DC); break;
    case 'P': Advance(TokenKind::DP); break;
    case 'T': Advance(TokenKind::DT); break;
    default: token_.set_kind(TokenKind::D); break;
    }
    break;
  case 'E':
    switch (LookAheadChar()) {
    case 'N': Advance(TokenKind::EN); break;
    case 'S': Advance(TokenKind::ES); break;
    case 'X': Advance(TokenKind::EX); break;
    default: token_.set_kind(TokenKind::E); break;
    }
    break;
  case 'F': token_.set_kind(TokenKind::F); break;
  case 'G': token_.set_kind(TokenKind::G); break;
  case 'I': token_.set_kind(TokenKind::I); break;
  case 'L': token_.set_kind(TokenKind::L); break;
  case 'O': token_.set_kind(TokenKind::O); break;
  case 'P': token_.set_kind(TokenKind::P); break;
  case 'R':
    switch (LookAheadChar()) {
    case 'C': Advance(TokenKind::RC); break;
    case 'D': Advance(TokenKind::RD); break;
    case 'N': Advance(TokenKind::RN); break;
    case 'P': Advance(TokenKind::RP); break;
    case 'U': Advance(TokenKind::RU); break;
    case 'Z': Advance(TokenKind::RZ); break;
    default: token_.set_kind(TokenKind::None); break;
    }
    break;
  case 'S':
    switch (LookAheadChar()) {
    case 'P': Advance(TokenKind::SP); break;
    case 'S': Advance(TokenKind::SS); break;
    default: token_.set_kind(TokenKind::S); break;
    }
    break;
  case 'T':
    switch (LookAheadChar()) {
    case 'L': Advance(TokenKind::TL); break;
    case 'R': Advance(TokenKind::TR); break;
    default: token_.set_kind(TokenKind::T); break;
    }
    break;
  case 'X': token_.set_kind(TokenKind::X); break;
  case 'Z': token_.set_kind(TokenKind::Z); break;
  case '-':
  case '+': token_.set_kind(TokenKind::Sign); break;
  case '/': token_.set_kind(TokenKind::Slash); break;
  case '(': token_.set_kind(TokenKind::LParen); break;
  case ')': token_.set_kind(TokenKind::RParen); break;
  case '.': token_.set_kind(TokenKind::Point); break;
  case ':': token_.set_kind(TokenKind::Colon); break;
  case '\\': token_.set_kind(TokenKind::Backslash); break;
  case '$': token_.set_kind(TokenKind::Dollar); break;
  case '*':
    token_.set_kind(LookAheadChar() == '(' ? TokenKind::Star : TokenKind::None);
    break;
  case ',': {
    token_.set_kind(TokenKind::Comma);
    CHAR laChar = LookAheadChar();
    if (laChar == ',') {
      Advance(TokenKind::Comma);
      token_.set_offset(cursor_ - format_);
      AppendError("Unexpected ',' in format expression");
    } else if (laChar == ')') {
      AppendError("Unexpected ',' before ')' in format expression");
    }
    break;
  }
  case '\'':
  case '"':
    for (++cursor_; cursor_ < end_; ++cursor_) {
      if (*cursor_ == c) {
        if (cursor_ >= end_ || *(cursor_ + 1) != c) {
          token_.set_kind(TokenKind::String);
          break;
        }
        ++cursor_;
      }
    }
    if (stmt_ == IoStmtKind::Read) {
      AppendError("String edit descriptor in READ format");  // 13.3.2p6
    }
    if (token_.kind() == TokenKind::String) {
      break;
    }
    token_.set_kind(TokenKind::None);
    cursor_ = end_;
    token_.set_length(cursor_ - format_ - token_.offset());
    AppendError("Unterminated string");
    break;
  default:
    if (cursor_ >= end_) {
      suppressErrorCascade_ = false;
      AppendError("Unterminated format expression");
    }
    token_.set_kind(TokenKind::None);
    break;
  }

  token_.set_length(cursor_ - format_ - token_.offset() + 1);
}

template<typename CHAR> int FormatValidator<CHAR>::Check() {
  if (format_ == nullptr || !*format_) {
    AppendError("Empty format expression");
    return errorCount_;
  }
  NextToken();
  if (token_.kind() != TokenKind::LParen) {
    AppendError("Format expression must have an initial '('");
    return errorCount_;
  }
  NextToken();

  int nestLevel{0};  // Outer level ()s are at level 0.
  Token starToken{};  // unlimited format token
  bool hasDataEditDesc{false};

  // Subject to error processing exceptions, a loop iteration processes an
  // edit descriptor or does list management.  The loop terminates when
  //  - a level-0 right paren is processed
  //  - the end of an incomplete format is reached
  //  - a threshold caller-chosen number of errors have been diagnosed
  while (errorCount_ < maxErrors_) {
    Token signToken{};
    Token knrToken{};  // (nonnegative) k, n, or r value
    int64_t knrValue{-1};  // -1 ==> not present
    int64_t wValue{-1};
    bool commaRequired{true};

    if (token_.kind() == TokenKind::Sign) {
      signToken = token_;
      NextToken();
    }
    if (token_.kind() == TokenKind::UnsignedInteger) {
      knrToken = token_;
      knrValue = integerValue_;
      NextToken();
    }
    if (signToken.IsSet() && (knrValue < 0 || token_.kind() != TokenKind::P)) {
      argString_[0] = format_[signToken.offset()];
      argString_[1] = 0;
      AppendError(
          "Unexpected '%s' in format expression", signToken, argString_);
    }
    // Default error argument.
    // Alphabetic descriptor names are one or two characters in length.
    argString_[0] = toupper(format_[token_.offset()]);
    argString_[1] = token_.length() > 1 ? toupper(*cursor_) : 0;

    auto check_r = [&](bool allowed = true, const char *name = {}) -> void {
      if (!allowed && knrValue >= 0) {
        AppendError("Repeat specifier before %s descriptor", knrToken, name);
      } else if (knrValue == 0) {
        AppendError(  // C1304
            "%s descriptor repeat specifier must be positive", knrToken, name);
      }
    };

    // Return the predicate "w value is present" to control further processing.
    [[maybe_unused]] auto check_w = [&](bool required = true) -> bool {
      if (token_.kind() == TokenKind::UnsignedInteger) {
        wValue = integerValue_;
        if (wValue == 0 &&
            (*argString_ == 'A' || *argString_ == 'L' ||
                stmt_ == IoStmtKind::Read)) {  // C1306, 13.7.2.1p6
          AppendError("%s descriptor 'w' value must be positive");
        }
        NextToken();
        return true;
      }
      if (required) {
        AppendError("Expected %s descriptor 'w' value");  // C1306
      }
      return false;
    };

    auto check_m = [&]() -> void {
      if (token_.kind() != TokenKind::Point) {
        return;
      }
      NextToken();
      if (token_.kind() != TokenKind::UnsignedInteger) {
        AppendError("Expected integer value after '.'");
        return;
      }
      if ((stmt_ == IoStmtKind::Print || stmt_ == IoStmtKind::Write) &&
          wValue > 0 && integerValue_ > wValue) {  // 13.7.2.2p5, 13.7.2.4p6
        AppendError("%s descriptor 'm' value is greater than 'w' value");
      }
      NextToken();
    };

    // Return the predicate "d value is present" to control further processing.
    [[maybe_unused]] auto check_d = [&](bool required = true) -> bool {
      if (token_.kind() != TokenKind::Point) {
        AppendError("Expected %s descriptor 'd' value");
        return false;
      }
      NextToken();
      if (token_.kind() != TokenKind::UnsignedInteger) {
        AppendError("Expected integer value after '.' in %s descriptor");
        return false;
      }
      NextToken();
      return true;
    };

    auto check_e = [&]() -> void {
      if (token_.kind() != TokenKind::E) {
        return;
      }
      NextToken();
      if (token_.kind() != TokenKind::UnsignedInteger) {
        AppendError("Expected integer value after 'E' in %s descriptor");
        return;
      }
      NextToken();
    };

    // Process one format descriptor or do format list management.
    switch (token_.kind()) {
    case TokenKind::A:
      // R1307 data-edit-desc -> A [w]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      check_w(false);
      break;
    case TokenKind::B:
    case TokenKind::I:
    case TokenKind::O:
    case TokenKind::Z:
      // R1307 data-edit-desc -> B w [. m] | I w [. m] | O w [. m] | Z w [. m]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w(warnOnNonstandardUsage_)) {
        check_m();
      }
      break;
    case TokenKind::D:
    case TokenKind::F:
      // R1307 data-edit-desc -> D w . d | F w . d
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w(warnOnNonstandardUsage_)) {
        check_d();
      }
      break;
    case TokenKind::E:
    case TokenKind::EN:
    case TokenKind::ES:
    case TokenKind::EX:
      // R1307 data-edit-desc ->
      //   E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w(warnOnNonstandardUsage_) && check_d()) {
        check_e();
      }
      break;
    case TokenKind::G:
      // R1307 data-edit-desc -> G w [. d [E e]]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w(warnOnNonstandardUsage_)) {
        if (wValue > 0) {
          if (check_d(true)) {  // C1307
            check_e();
          }
        } else if (check_d() && token_.kind() == TokenKind::E) {
          AppendError("Unexpected 'e' in G0 edit descriptor");  // C1308
          NextToken();
          if (token_.kind() == TokenKind::UnsignedInteger) {
            NextToken();
          }
        }
      }
      break;
    case TokenKind::L:
      // R1307 data-edit-desc -> L w
      hasDataEditDesc = true;
      check_r();
      NextToken();
      check_w(warnOnNonstandardUsage_);
      break;
    case TokenKind::DT:
      // R1307 data-edit-desc -> DT [char-literal-constant] [( v-list )]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (token_.kind() == TokenKind::String) {
        NextToken();
      }
      if (token_.kind() == TokenKind::LParen) {
        do {
          NextToken();
          if (token_.kind() == TokenKind::Sign) {
            NextToken();
          }
          if (token_.kind() != TokenKind::UnsignedInteger) {
            AppendError(
                "Expected integer constant in DT edit descriptor v-list");
            break;
          }
          NextToken();
        } while (token_.kind() == TokenKind::Comma);
        if (token_.kind() != TokenKind::RParen) {
          AppendError("Expected ',' or ')' in DT edit descriptor v-list");
          while (cursor_ < end_ && token_.kind() != TokenKind::RParen) {
            NextToken();
          }
        }
        NextToken();
      }
      break;
    case TokenKind::String:
      // R1304 data-edit-desc -> char-string-edit-desc
      check_r(false, "character string");
      NextToken();
      break;
    case TokenKind::BN:
    case TokenKind::BZ:
    case TokenKind::DC:
    case TokenKind::DP:
    case TokenKind::RC:
    case TokenKind::RD:
    case TokenKind::RN:
    case TokenKind::RP:
    case TokenKind::RU:
    case TokenKind::RZ:
    case TokenKind::S:
    case TokenKind::SP:
    case TokenKind::SS:
      // R1317 sign-edit-desc -> SS | SP | S
      // R1318 blank-interp-edit-desc -> BN | BZ
      // R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
      // R1320 decimal-edit-desc -> DC | DP
      check_r(false);
      NextToken();
      break;
    case TokenKind::P: {
      // R1313 control-edit-desc -> k P
      if (knrValue < 0) {
        AppendError("P descriptor must have a scale factor");
      }
      // Diagnosing C1302 may require multiple token lookahead.
      // Save current cursor position to enable backup.
      const CHAR *saveCursor{cursor_};
      NextToken();
      if (token_.kind() == TokenKind::UnsignedInteger) {
        NextToken();
      }
      switch (token_.kind()) {
      case TokenKind::D:
      case TokenKind::E:
      case TokenKind::EN:
      case TokenKind::ES:
      case TokenKind::EX:
      case TokenKind::F:
      case TokenKind::G: commaRequired = false; break;
      default:;
      }
      cursor_ = saveCursor;
      NextToken();
      break;
    }
    case TokenKind::T:
    case TokenKind::TL:
    case TokenKind::TR:
      // R1315 position-edit-desc -> T n | TL n | TR n
      check_r(false);
      NextToken();
      if (integerValue_ <= 0) {  // C1311
        AppendError("%s descriptor must have a positive position value");
      }
      NextToken();
      break;
    case TokenKind::X:
      // R1315 position-edit-desc -> n X
      if (knrValue == 0) {  // C1311
        AppendError(
            "X descriptor must have a positive position value", knrToken);
      } else if (knrValue < 0 && warnOnNonstandardUsage_) {
        AppendError("X descriptor must have a positive position value");
      }
      NextToken();
      break;
    case TokenKind::Colon:
      // R1313 control-edit-desc -> :
      check_r(false);
      commaRequired = false;
      NextToken();
      break;
    case TokenKind::Slash:
      // R1313 control-edit-desc -> [r] /
      commaRequired = false;
      NextToken();
      break;
    case TokenKind::Backslash:
      check_r(false);
      if (warnOnNonstandardUsage_) {
        AppendError("Non-standard \\ edit descriptor");
      }
      NextToken();
      break;
    case TokenKind::Dollar:
      check_r(false);
      if (warnOnNonstandardUsage_) {
        AppendError("Non-standard $ edit descriptor");
      }
      NextToken();
      break;
    case TokenKind::Star:
      // NextToken assigns a token kind of Star only if * is followed by (.
      // So the next token is guaranteed to be LParen.
      if (nestLevel > 0) {
        AppendError("Nested unlimited format list");
      }
      starToken = token_;
      check_r(false, "unlimited format item");
      hasDataEditDesc = false;
      NextToken();
      // fall through
    case TokenKind::LParen:
      if (knrValue == 0) {
        AppendError("List repeat specifier must be positive", knrToken);
      }
      ++nestLevel;
      break;
    case TokenKind::RParen:
      if (knrValue >= 0) {
        AppendError("Unexpected integer constant", knrToken);
      }
      do {
        if (nestLevel == 0) {
          // Any characters after level-0 ) are ignored.
          return errorCount_;  // normal exit (may have errors)
        }
        if (nestLevel == 1 && starToken.IsSet() && !hasDataEditDesc) {
          starToken.set_length(cursor_ - format_ - starToken.offset() + 1);
          AppendError(  // C1303
              "Unlimited format list must contain a data edit descriptor",
              starToken);
        }
        --nestLevel;
        NextToken();
      } while (token_.kind() == TokenKind::RParen);
      if (nestLevel == 0 && starToken.IsSet()) {
        AppendError("Character in format after unlimited format list");
      }
      break;
    case TokenKind::Comma:
      if (knrValue >= 0) {
        AppendError("Unexpected integer constant", knrToken);
      }
      break;
    default: AppendError("Unexpected '%s' in format expression"); NextToken();
    }

    // Process comma separator and exit an incomplete format.
    switch (token_.kind()) {
    case TokenKind::Colon:  // Comma not required; token not yet processed.
    case TokenKind::Slash:  // Comma not required; token not yet processed.
    case TokenKind::RParen:  // Comma not allowed; token not yet processed.
      suppressErrorCascade_ = false;
      break;
    case TokenKind::LParen:  // Comma not allowed; token already processed.
    case TokenKind::Comma:  // Normal comma case; move past token.
      suppressErrorCascade_ = false;
      NextToken();
      break;
    case TokenKind::Sign:  // Error; main switch has a better message.
    case TokenKind::None:  // Error; token not yet processed.
      if (cursor_ >= end_) {
        return errorCount_;  // incomplete format error exit
      }
      break;
    default:
      // Possible first token of the next format item; token not yet processed.
      if (commaRequired) {
        AppendError("Expected ',' or ')' in format expression");  // C1302
      }
    }
  }

  return errorCount_;  // error threshold exit
}

template class FormatValidator<char>;
template class FormatValidator<char16_t>;
template class FormatValidator<char32_t>;

}
