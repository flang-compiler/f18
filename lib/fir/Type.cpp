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

#include "fir/Type.h"
#include "fir/Dialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace {

// Tokens

enum class TokenKind {
  error,
  eof,
  leftang,
  rightang,
  leftparen,
  rightparen,
  leftbrace,
  rightbrace,
  leftbracket,
  rightbracket,
  colon,
  comma,
  period,
  eroteme,
  ecphoneme,
  star,
  arrow,
  ident,
  string,
  intlit,
};

struct Token {
  TokenKind kind;
  L::StringRef text;
};

// Lexer

class Lexer {
public:
  Lexer(L::StringRef source) : srcBuff{source}, srcPtr{source.begin()} {}

  // consume and return next token from the input. input is advanced to after
  // the token.
  Token lexToken();

  int nextNonWSChar() {
    skipWhitespace();
    if (atEnd())
      return -1;
    return *srcPtr++;
  }

  // peek ahead to the next non-whitespace character, leaving it on the input
  // stream
  char nextChar() {
    skipWhitespace();
    if (atEnd())
      return '\0';
    return *srcPtr;
  }

  // advance the input stream `count` characters
  void advance(unsigned count = 1) {
    while (count--) {
      if (atEnd())
        break;
      ++srcPtr;
    }
  }

  const char *getMarker() { return srcPtr; }

  L::StringRef getNextType() {
    char const *const marker = srcPtr;
    L::SmallVector<char, 8> nestedPunc;
    for (bool scanning = true; scanning && !atEnd(); ++srcPtr) {
      char c = *srcPtr;
      switch (c) {
      case '<':
      case '[':
      case '(':
      case '{':
        nestedPunc.push_back(c);
        break;
      case '>':
        if (nestedPunc.empty() || nestedPunc.pop_back_val() != '<') {
          goto done;
        }
        continue;
      case ']':
        if (nestedPunc.empty() || nestedPunc.pop_back_val() != '[') {
          goto done;
        }
        continue;
      case ')':
        if (nestedPunc.empty() || nestedPunc.pop_back_val() != '(') {
          goto done;
        }
        continue;
      case '}':
        if (nestedPunc.empty() || nestedPunc.pop_back_val() != '{') {
          goto done;
        }
        continue;
      case ',':
        if (nestedPunc.empty()) {
          goto done;
        }
        continue;
      case '-':
        if ((srcPtr + 1 != srcBuff.end()) && *(srcPtr + 1) == '>') {
          ++srcPtr;
        }
        continue;
      done:
        --srcPtr;
        [[fallthrough]];
      case '\0': {
        scanning = false;
      } break;
      default:
        break;
      }
    }
    std::size_t count = srcPtr - marker;
    return {marker, count};
  }

private:
  void skipWhitespace() {
    while (!atEnd()) {
      switch (*srcPtr) {
      case ' ':
      case '\f':
      case '\n':
      case '\r':
      case '\t':
      case '\v':
        ++srcPtr;
        continue;
      default:
        break;
      }
      break;
    }
  }

  Token formToken(TokenKind kind, const char *tokStart) {
    return Token{kind, L::StringRef(tokStart, srcPtr - tokStart)};
  }

  Token emitError(const char *loc, const L::Twine &message) {
    return formToken(TokenKind::error, loc);
  }

  bool atEnd() const { return srcPtr == srcBuff.end(); }

  Token lexIdent(const char *tokStart);
  Token lexNumber(const char *tokStart);
  Token lexString(const char *tokStart);

  L::StringRef srcBuff;
  const char *srcPtr;
};

Token Lexer::lexToken() {
  skipWhitespace();
  if (atEnd()) {
    return formToken(TokenKind::eof, "");
  }

  const char *tokStart = srcPtr;
  switch (*srcPtr++) {
  case '<':
    return formToken(TokenKind::leftang, tokStart);
  case '>':
    return formToken(TokenKind::rightang, tokStart);
  case '{':
    return formToken(TokenKind::leftbrace, tokStart);
  case '}':
    return formToken(TokenKind::rightbrace, tokStart);
  case '[':
    return formToken(TokenKind::leftbracket, tokStart);
  case ']':
    return formToken(TokenKind::rightbracket, tokStart);
  case '(':
    return formToken(TokenKind::leftparen, tokStart);
  case ')':
    return formToken(TokenKind::rightparen, tokStart);
  case ':':
    return formToken(TokenKind::colon, tokStart);
  case ',':
    return formToken(TokenKind::comma, tokStart);
  case '"':
    return lexString(tokStart + 1);
  case '-':
    if (*srcPtr == '>') {
      srcPtr++;
      return formToken(TokenKind::arrow, tokStart);
    }
    return lexNumber(tokStart);
  case '+':
    return lexNumber(tokStart + 1);
  case '!':
    return formToken(TokenKind::ecphoneme, tokStart);
  case '?':
    return formToken(TokenKind::eroteme, tokStart);
  case '*':
    return formToken(TokenKind::star, tokStart);
  case '.':
    return formToken(TokenKind::period, tokStart);
  default:
    if (std::isalpha(*tokStart)) {
      return lexIdent(tokStart);
    }
    if (std::isdigit(*tokStart)) {
      return lexNumber(tokStart);
    }
    return emitError(tokStart, "unexpected character");
  }
}

Token Lexer::lexString(const char *tokStart) {
  while (!atEnd() && *srcPtr != '"') {
    ++srcPtr;
  }
  Token token{formToken(TokenKind::string, tokStart)};
  ++srcPtr;
  return token;
}

Token Lexer::lexIdent(const char *tokStart) {
  while (!atEnd() && (std::isalnum(*srcPtr) || *srcPtr == '_')) {
    ++srcPtr;
  }
  return formToken(TokenKind::ident, tokStart);
}

Token Lexer::lexNumber(const char *tokStart) {
  while (!atEnd() && std::isdigit(*srcPtr)) {
    ++srcPtr;
  }
  return formToken(TokenKind::intlit, tokStart);
}

class auto_counter {
public:
  explicit auto_counter(int &ref) : counter(ref) { ++counter; }
  ~auto_counter() { --counter; }

private:
  auto_counter() = delete;
  auto_counter(const auto_counter &) = delete;
  int &counter;
};

/// A FIROpsDialect instance uses a FIRTypeParser object to parse and
/// instantiate all FIR types from .fir files.
class FIRTypeParser {
public:
  FIRTypeParser(FIROpsDialect *dialect, L::StringRef rawData, M::Location loc)
      : context{dialect->getContext()}, lexer{rawData}, loc{loc} {}

  M::Type parseType();

protected:
  inline void emitError(M::Location loc, const L::Twine &msg) {
    M::emitError(loc, msg);
  }

  bool consumeToken(TokenKind tk, const L::Twine &msg) {
    auto token = lexer.lexToken();
    if (token.kind != tk) {
      emitError(loc, msg);
      return true; // error!
    }
    return false;
  }

  bool consumeChar(char ch, const L::Twine &msg) {
    auto lexCh = lexer.nextChar();
    if (lexCh != ch) {
      emitError(loc, msg);
      return true; // error!
    }
    lexer.advance();
    return false;
  }

  template <typename A>
  A parseIntLitSingleton(const char *msg) {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto token{lexer.lexToken()};
    if (token.kind != TokenKind::intlit) {
      emitError(loc, msg);
      return {};
    }
    KindTy kind;
    if (token.text.getAsInteger(0, kind)) {
      emitError(loc, "expected integer constant");
      return {};
    }
    if (consumeToken(TokenKind::rightang, "expected '>' in type")) {
      return {};
    }
    if (checkAtEnd())
      return {};
    return A::get(getContext(), kind);
  }

  // `<` kind `>`
  template <typename A>
  A parseKindSingleton() {
    return parseIntLitSingleton<A>("expected kind parameter");
  }

  // `<` rank `>`
  template <typename A>
  A parseRankSingleton() {
    return parseIntLitSingleton<A>("expected rank parameter");
  }

  // '<' type '>'
  template <typename A>
  A parseTypeSingleton() {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto ofTy = parseNextType();
    if (!ofTy) {
      emitError(loc, "expected type parameter");
      return {};
    }
    if (consumeToken(TokenKind::rightang, "expected '>' in type")) {
      return {};
    }
    if (checkAtEnd())
      return {};
    return A::get(ofTy);
  }

  M::Type parseNextType();

  bool checkAtEnd() {
    if (!recursiveCall) {
      auto token = lexer.lexToken();
      if (token.kind != TokenKind::eof) {
        emitError(loc, "unexpected extra characters");
        return true;
      }
    }
    return false;
  }

  // `box` `<` type (',' affine-map)? `>`
  BoxType parseBox() {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto ofTy = parseNextType();
    if (!ofTy) {
      emitError(loc, "expected type parameter");
      return {};
    }
    auto token = lexer.lexToken();
    M::AffineMapAttr map;
    if (token.kind == TokenKind::comma) {
      map = parseAffineMapAttr();
      token = lexer.lexToken();
    }
    if (token.kind != TokenKind::rightang) {
      emitError(loc, "expected '>' in type");
      return {};
    }
    if (checkAtEnd())
      return {};
    return BoxType::get(ofTy, map);
  }

  // `boxchar` `<` kind `>`
  BoxCharType parseBoxChar() { return parseKindSingleton<BoxCharType>(); }

  // `boxproc` `<` return-type `>`
  BoxProcType parseBoxProc() { return parseTypeSingleton<BoxProcType>(); }

  // `char` `<` kind `>`
  CharacterType parseCharacter() { return parseKindSingleton<CharacterType>(); }

  // `dims` `<` rank `>`
  DimsType parseDims() { return parseRankSingleton<DimsType>(); }

  // `field`
  FieldType parseField() {
    if (checkAtEnd())
      return {};
    return FieldType::get(getContext());
  }

  // `logical` `<` kind `>`
  LogicalType parseLogical() { return parseKindSingleton<LogicalType>(); }

  // `int` `<` kind `>`
  IntType parseInteger() { return parseKindSingleton<IntType>(); }

  // `complex` `<` kind `>`
  CplxType parseComplex() { return parseKindSingleton<CplxType>(); }

  // `real` `<` kind `>`
  RealType parseReal() { return parseKindSingleton<RealType>(); }

  // `ref` `<` type `>`
  ReferenceType parseReference() { return parseTypeSingleton<ReferenceType>(); }

  // `ptr` `<` type `>`
  PointerType parsePointer() { return parseTypeSingleton<PointerType>(); }

  // `heap` `<` type `>`
  HeapType parseHeap() { return parseTypeSingleton<HeapType>(); }

  SequenceType::Shape parseShape();
  SequenceType parseSequence();

  RecordType::TypeList parseTypeList();
  RecordType::TypeList parseLenParamList();
  RecordType parseDerived();
  RecordType verifyDerived(RecordType derivedTy,
                           llvm::ArrayRef<RecordType::TypePair> lenPList,
                           llvm::ArrayRef<RecordType::TypePair> typeList);

  // `tdesc` `<` type `>`
  TypeDescType parseTypeDesc() { return parseTypeSingleton<TypeDescType>(); }

  // `void`
  M::Type parseVoid() {
    if (checkAtEnd())
      return {};
    return M::TupleType::get(getContext());
  }

  M::MLIRContext *getContext() const { return context; }

  M::AffineMapAttr parseAffineMapAttr();

private:
  M::MLIRContext *context;
  Lexer lexer;
  M::Location loc;
  int recursiveCall{-1};
};

// If this is a `!fir.x` type then recursively parse it now, otherwise figure
// out its extent and call into the standard type parser.
M::Type FIRTypeParser::parseNextType() {
  return M::parseType(lexer.getNextType(), getContext());
}

// Parses either `*` `:`
//            or (int | `?`) (`x` (int | `?`))* `:`
SequenceType::Shape FIRTypeParser::parseShape() {
  SequenceType::Bounds bounds;
  int extent;
  int nextChar;
  Token token = lexer.lexToken();
  if (token.kind == TokenKind::star) {
    token = lexer.lexToken();
    if (token.kind != TokenKind::colon) {
      emitError(loc, "expected '*' to be followed by ':'");
      return {};
    }
    return SequenceType::Shape();
  }
  while (true) {
    if (token.kind != TokenKind::eroteme) {
      goto shape_spec;
    }
    bounds.emplace_back(SequenceType::Extent());
    goto check_xchar;
  shape_spec:
    if (token.kind != TokenKind::intlit) {
      emitError(loc, "expected an integer or '?' in shape specification");
      return {};
    }
    token.text.getAsInteger(10, extent);
    bounds.emplace_back(extent);
  check_xchar:
    nextChar = lexer.nextNonWSChar();
    if (nextChar == ':') {
      return SequenceType::Shape(bounds);
    }
    if (nextChar != 'x') {
      emitError(loc, "expected an 'x' or ':' after integer");
      return {};
    }
    token = lexer.lexToken();
  }
}

// affine-map ::= `#` ident
M::AffineMapAttr FIRTypeParser::parseAffineMapAttr() {
  return {}; // FIXME opParser->parseAttr();
}

// bounds ::= lo extent stride | `?`
// `array` `<` bounds (`,` bounds)* `:` type (',' affine-map)? `>`
SequenceType FIRTypeParser::parseSequence() {
  if (consumeToken(TokenKind::leftang, "expected '<' in array type")) {
    return {};
  }
  auto shape = parseShape();
  M::Type eleTy = parseNextType();
  if (!eleTy) {
    emitError(loc, "invalid element type");
    return {};
  }
  auto token = lexer.lexToken();
  M::AffineMapAttr map;
  if (token.kind == TokenKind::comma) {
    map = parseAffineMapAttr();
    token = lexer.lexToken();
  }
  if (token.kind != TokenKind::rightang) {
    emitError(loc, "expected '>' in array type");
    return {};
  }
  if (checkAtEnd()) {
    return {};
  }
  return SequenceType::get(shape, eleTy, map);
}

// Parses: string `:` type (',' string `:` type)* '}'
RecordType::TypeList FIRTypeParser::parseTypeList() {
  RecordType::TypeList result;
  while (true) {
    auto name{lexer.lexToken()};
    if (name.kind != TokenKind::ident) {
      emitError(loc, "expected identifier");
      return {};
    }
    if (consumeToken(TokenKind::colon, "expected colon")) {
      return {};
    }
    auto memTy{parseNextType()};
    result.emplace_back(name.text, memTy);
    auto token{lexer.lexToken()};
    if (token.kind == TokenKind::rightbrace) {
      return result;
    }
    if (token.kind != TokenKind::comma) {
      emitError(loc, "expected ','");
      return {};
    }
  }
}

// Parses: string `:` int-type (',' string `:` int-type)* ')'
RecordType::TypeList FIRTypeParser::parseLenParamList() {
  RecordType::TypeList result;
  while (true) {
    auto name{lexer.lexToken()};
    if (name.kind != TokenKind::ident) {
      emitError(loc, "expected identifier");
      return {};
    }
    if (consumeToken(TokenKind::colon, "expected colon")) {
      return {};
    }
    auto memTy{parseNextType()};
    result.emplace_back(name.text, memTy);
    auto token{lexer.lexToken()};
    if (token.kind == TokenKind::rightparen) {
      return result;
    }
    if (token.kind != TokenKind::comma) {
      emitError(loc, "expected ','");
      return {};
    }
  }
}

bool verifyIntegerType(M::Type ty) {
  return ty.dyn_cast<M::IntegerType>() || ty.dyn_cast<IntType>();
}

bool verifyRecordMemberType(M::Type ty) {
  return !(ty.dyn_cast<BoxType>() || ty.dyn_cast<BoxCharType>() ||
           ty.dyn_cast<BoxProcType>() || ty.dyn_cast<DimsType>() ||
           ty.dyn_cast<FieldType>() || ty.dyn_cast<ReferenceType>() ||
           ty.dyn_cast<TypeDescType>());
}

bool verifySameLists(L::ArrayRef<RecordType::TypePair> a1,
                     L::ArrayRef<RecordType::TypePair> a2) {
  if (a1.size() != a2.size())
    return false;
  auto iter = a1.begin();
  for (auto lp : a2) {
    if (!((iter->first == lp.first) && (iter->second == lp.second)))
      return false;
    ++iter;
  }
  return true;
}

RecordType
FIRTypeParser::verifyDerived(RecordType derivedTy,
                             L::ArrayRef<RecordType::TypePair> lenPList,
                             L::ArrayRef<RecordType::TypePair> typeList) {
  if (!verifySameLists(derivedTy.getLenParamList(), lenPList) ||
      !verifySameLists(derivedTy.getTypeList(), typeList)) {
    emitError(loc, "cannot redefine record type members");
    return {};
  }
  for (auto &p : lenPList)
    if (!verifyIntegerType(p.second)) {
      emitError(loc, "LEN parameter must be integral type");
      return {};
    }
  for (auto &p : typeList)
    if (!verifyRecordMemberType(p.second)) {
      emitError(loc, "field parameter has invalid type");
      return {};
    }
  llvm::StringSet<> uniq;
  for (auto &p : lenPList)
    if (!uniq.insert(p.first).second) {
      emitError(loc, "LEN parameter cannot have duplicate name");
      return {};
    }
  for (auto &p : typeList)
    if (!uniq.insert(p.first).second) {
      emitError(loc, "field cannot have duplicate name");
      return {};
    }
  return derivedTy;
}

// Fortran derived type
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`{` id `:` type (`,` id `:` type)* `}`)? '>'
RecordType FIRTypeParser::parseDerived() {
  if (consumeToken(TokenKind::leftang, "expected '<' in type type")) {
    return {};
  }
  auto name{lexer.lexToken()};
  if (name.kind != TokenKind::ident) {
    emitError(loc, "expected a identifier as name of derived type");
    return {};
  }
  RecordType result = RecordType::get(getContext(), name.text);
  auto token = lexer.lexToken();
  RecordType::TypeList lenParamList;
  RecordType::TypeList typeList;
  if (token.kind == TokenKind::leftbrace) {
    goto parse_fields;
  } else if (token.kind != TokenKind::leftparen) {
    // degenerate case?
    goto check_close;
  }
  lenParamList = parseLenParamList();
  token = lexer.lexToken();
  if (token.kind != TokenKind::leftbrace) {
    // no fields?
    goto check_close;
  }
parse_fields:
  typeList = parseTypeList();
  token = lexer.lexToken();
check_close:
  if (token.kind != TokenKind::rightang) {
    emitError(loc, "expected '>' in type type");
    return {};
  }
  if (checkAtEnd()) {
    return {};
  }
  if (lenParamList.empty() && typeList.empty())
    return result;
  result.finalize(lenParamList, typeList);
  return verifyDerived(result, lenParamList, typeList);
}

M::Type FIRTypeParser::parseType() {
  auto_counter c{recursiveCall};
  auto token = lexer.lexToken();
  if (token.kind == TokenKind::ident) {
    if (token.text == "ref")
      return parseReference();
    if (token.text == "array")
      return parseSequence();
    if (token.text == "char")
      return parseCharacter();
    if (token.text == "logical")
      return parseLogical();
    if (token.text == "real")
      return parseReal();
    if (token.text == "type")
      return parseDerived();
    if (token.text == "box")
      return parseBox();
    if (token.text == "boxchar")
      return parseBoxChar();
    if (token.text == "boxproc")
      return parseBoxProc();
    if (token.text == "ptr")
      return parsePointer();
    if (token.text == "heap")
      return parseHeap();
    if (token.text == "dims")
      return parseDims();
    if (token.text == "tdesc")
      return parseTypeDesc();
    if (token.text == "field")
      return parseField();
    if (token.text == "int")
      return parseInteger();
    if (token.text == "complex")
      return parseComplex();
    if (token.text == "void")
      return parseVoid();
    emitError(loc, "not a known fir type");
    return {};
  }
  emitError(loc, "invalid token");
  return {};
}

// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
bool singleIndirectionLevel(M::Type ty) {
  return !(ty.dyn_cast<ReferenceType>() || ty.dyn_cast<PointerType>() ||
           ty.dyn_cast<HeapType>());
}

} // namespace

namespace fir {
namespace detail {

// Type storage classes

/// `CHARACTER` storage
struct CharacterTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static CharacterTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                         KindTy kind) {
    auto *storage = allocator.allocate<CharacterTypeStorage>();
    return new (storage) CharacterTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  CharacterTypeStorage() = delete;
  explicit CharacterTypeStorage(KindTy kind) : kind{kind} {}
};

struct DimsTypeStorage : public M::TypeStorage {
  using KeyTy = unsigned;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const {
    return key == static_cast<unsigned>(getRank());
  }

  static DimsTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                    int rank) {
    auto *storage = allocator.allocate<DimsTypeStorage>();
    return new (storage) DimsTypeStorage{rank};
  }

  int getRank() const { return rank; }

protected:
  int rank;

private:
  DimsTypeStorage() = delete;
  explicit DimsTypeStorage(int rank) : rank{rank} {}
};

/// The type of a derived type part reference
struct FieldTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &) { return L::hash_combine(0); }

  bool operator==(const KeyTy &) const { return true; }

  static FieldTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                     KindTy) {
    auto *storage = allocator.allocate<FieldTypeStorage>();
    return new (storage) FieldTypeStorage{0};
  }

private:
  FieldTypeStorage() = delete;
  explicit FieldTypeStorage(KindTy) {}
};

/// `LOGICAL` storage
struct LogicalTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static LogicalTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<LogicalTypeStorage>();
    return new (storage) LogicalTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  LogicalTypeStorage() = delete;
  explicit LogicalTypeStorage(KindTy kind) : kind{kind} {}
};

/// `INTEGER` storage
struct IntTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static IntTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                   KindTy kind) {
    auto *storage = allocator.allocate<IntTypeStorage>();
    return new (storage) IntTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  IntTypeStorage() = delete;
  explicit IntTypeStorage(KindTy kind) : kind{kind} {}
};

/// `COMPLEX` storage
struct CplxTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static CplxTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                    KindTy kind) {
    auto *storage = allocator.allocate<CplxTypeStorage>();
    return new (storage) CplxTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  CplxTypeStorage() = delete;
  explicit CplxTypeStorage(KindTy kind) : kind{kind} {}
};

/// `REAL` storage (for reals of unsupported sizes)
struct RealTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static RealTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                    KindTy kind) {
    auto *storage = allocator.allocate<RealTypeStorage>();
    return new (storage) RealTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  RealTypeStorage() = delete;
  explicit RealTypeStorage(KindTy kind) : kind{kind} {}
};

/// Boxed object (a Fortran descriptor)
struct BoxTypeStorage : public M::TypeStorage {
  using KeyTy = std::tuple<M::Type, M::AffineMapAttr>;

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{L::hash_combine(std::get<M::Type>(key))};
    return L::hash_combine(hashVal,
                           L::hash_combine(std::get<M::AffineMapAttr>(key)));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<M::Type>(key) == getElementType() &&
           std::get<M::AffineMapAttr>(key) == getLayoutMap();
  }

  static BoxTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto *storage = allocator.allocate<BoxTypeStorage>();
    return new (storage)
        BoxTypeStorage{std::get<M::Type>(key), std::get<M::AffineMapAttr>(key)};
  }

  M::Type getElementType() const { return eleTy; }
  M::AffineMapAttr getLayoutMap() const { return map; }

protected:
  M::Type eleTy;
  M::AffineMapAttr map;

private:
  BoxTypeStorage() = delete;
  explicit BoxTypeStorage(M::Type eleTy, M::AffineMapAttr map)
      : eleTy{eleTy}, map{map} {}
};

/// Boxed CHARACTER object type
struct BoxCharTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static BoxCharTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<BoxCharTypeStorage>();
    return new (storage) BoxCharTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

  // a !fir.boxchar<k> always wraps a !fir.char<k>
  CharacterType getElementType(M::MLIRContext *ctxt) const {
    return CharacterType::get(ctxt, getFKind());
  }

protected:
  KindTy kind;

private:
  BoxCharTypeStorage() = delete;
  explicit BoxCharTypeStorage(KindTy kind) : kind{kind} {}
};

/// Boxed PROCEDURE POINTER object type
struct BoxProcTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static BoxProcTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                       M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<BoxProcTypeStorage>();
    return new (storage) BoxProcTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  BoxProcTypeStorage() = delete;
  explicit BoxProcTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer-like object storage
struct ReferenceTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static ReferenceTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                         M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<ReferenceTypeStorage>();
    return new (storage) ReferenceTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  ReferenceTypeStorage() = delete;
  explicit ReferenceTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer object storage
struct PointerTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static PointerTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                       M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<PointerTypeStorage>();
    return new (storage) PointerTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  PointerTypeStorage() = delete;
  explicit PointerTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Heap memory reference object storage
struct HeapTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static HeapTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                    M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<HeapTypeStorage>();
    return new (storage) HeapTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  HeapTypeStorage() = delete;
  explicit HeapTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Sequence-like object storage
struct SequenceTypeStorage : public M::TypeStorage {
  using KeyTy = std::tuple<SequenceType::Shape, M::Type, M::AffineMapAttr>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{hash_value(std::get<SequenceType::Shape>(key))};
    shapeHash = L::hash_combine(shapeHash, std::get<M::Type>(key));
    return L::hash_combine(shapeHash, std::get<M::AffineMapAttr>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType(), getLayoutMap()};
  }

  static SequenceTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    auto *storage = allocator.allocate<SequenceTypeStorage>();
    return new (storage) SequenceTypeStorage{std::get<SequenceType::Shape>(key),
                                             std::get<M::Type>(key),
                                             std::get<M::AffineMapAttr>(key)};
  }

  SequenceType::Shape getShape() const { return shape; }
  M::Type getElementType() const { return eleTy; }
  M::AffineMapAttr getLayoutMap() const { return map; }

protected:
  SequenceType::Shape shape;
  M::Type eleTy;
  M::AffineMapAttr map;

private:
  SequenceTypeStorage() = delete;
  explicit SequenceTypeStorage(const SequenceType::Shape &shape, M::Type eleTy,
                               M::AffineMapAttr map)
      : shape{shape}, eleTy{eleTy}, map{map} {}
};

/// Derived type storage
struct RecordTypeStorage : public M::TypeStorage {
  using KeyTy = L::StringRef;

  static unsigned hashKey(const KeyTy &key) {
    return L::hash_combine(key.str());
  }

  bool operator==(const KeyTy &key) const { return key == getName(); }

  static RecordTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto *storage = allocator.allocate<RecordTypeStorage>();
    return new (storage) RecordTypeStorage{key};
  }

  L::StringRef getName() const { return name; }

  void setLenParamList(L::ArrayRef<RecordType::TypePair> list) { lens = list; }
  L::ArrayRef<RecordType::TypePair> getLenParamList() const { return lens; }

  void setTypeList(L::ArrayRef<RecordType::TypePair> list) { types = list; }
  L::ArrayRef<RecordType::TypePair> getTypeList() const { return types; }

  void finalize(L::ArrayRef<RecordType::TypePair> lenParamList,
                L::ArrayRef<RecordType::TypePair> typeList) {
    if (finalized)
      return;
    finalized = true;
    setLenParamList(lenParamList);
    setTypeList(typeList);
  }

protected:
  std::string name;
  bool finalized;
  std::vector<RecordType::TypePair> lens;
  std::vector<RecordType::TypePair> types;

private:
  RecordTypeStorage() = delete;
  explicit RecordTypeStorage(L::StringRef name)
      : name{name}, finalized{false} {}
};

/// Type descriptor type storage
struct TypeDescTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getOfType(); }

  static TypeDescTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                        M::Type ofTy) {
    assert(ofTy && "descriptor type is null");
    auto *storage = allocator.allocate<TypeDescTypeStorage>();
    return new (storage) TypeDescTypeStorage{ofTy};
  }

  // The type described by this type descriptor instance
  M::Type getOfType() const { return ofTy; }

protected:
  M::Type ofTy;

private:
  TypeDescTypeStorage() = delete;
  explicit TypeDescTypeStorage(M::Type ofTy) : ofTy{ofTy} {}
};

} // namespace detail
} // namespace fir

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool fir::isa_fir_type(mlir::Type t) {
  return inbounds(t.getKind(), M::Type::FIRST_FIR_TYPE, M::Type::LAST_FIR_TYPE);
}

bool fir::isa_std_type(mlir::Type t) {
  return inbounds(t.getKind(), M::Type::FIRST_STANDARD_TYPE,
                  M::Type::LAST_STANDARD_TYPE);
}

bool fir::isa_fir_or_std_type(mlir::Type t) {
  return isa_fir_type(t) || isa_std_type(t);
}

// CHARACTER

CharacterType fir::CharacterType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_CHARACTER, kind);
}

int fir::CharacterType::getFKind() const { return getImpl()->getFKind(); }

// Dims

DimsType fir::DimsType::get(M::MLIRContext *ctxt, unsigned rank) {
  return Base::get(ctxt, FIR_DIMS, rank);
}

int fir::DimsType::getRank() const { return getImpl()->getRank(); }

// Field

FieldType fir::FieldType::get(M::MLIRContext *ctxt, KindTy) {
  return Base::get(ctxt, FIR_FIELD, 0);
}

// LOGICAL

LogicalType fir::LogicalType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_LOGICAL, kind);
}

int fir::LogicalType::getFKind() const { return getImpl()->getFKind(); }

// INTEGER

IntType fir::IntType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_INT, kind);
}

int fir::IntType::getFKind() const { return getImpl()->getFKind(); }

// COMPLEX

CplxType fir::CplxType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_COMPLEX, kind);
}

KindTy fir::CplxType::getFKind() const { return getImpl()->getFKind(); }

// REAL

RealType fir::RealType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_REAL, kind);
}

int fir::RealType::getFKind() const { return getImpl()->getFKind(); }

// Box<T>

BoxType fir::BoxType::get(M::Type elementType, M::AffineMapAttr map) {
  return Base::get(elementType.getContext(), FIR_BOX, elementType, map);
}

M::Type fir::BoxType::getEleTy() const { return getImpl()->getElementType(); }

M::AffineMapAttr fir::BoxType::getLayoutMap() const {
  return getImpl()->getLayoutMap();
}

M::LogicalResult
fir::BoxType::verifyConstructionInvariants(L::Optional<M::Location>,
                                           M::MLIRContext *ctx, M::Type eleTy,
                                           mlir::AffineMapAttr map) {
  // TODO
  return M::success();
}

// BoxChar<C>

BoxCharType fir::BoxCharType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_BOXCHAR, kind);
}

CharacterType fir::BoxCharType::getEleTy() const {
  return getImpl()->getElementType(getContext());
}

// BoxProc<T>

BoxProcType fir::BoxProcType::get(M::Type elementType) {
  return Base::get(elementType.getContext(), FIR_BOXPROC, elementType);
}

M::Type fir::BoxProcType::getEleTy() const {
  return getImpl()->getElementType();
}

M::LogicalResult fir::BoxProcType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  if (eleTy.dyn_cast<M::FunctionType>() || eleTy.dyn_cast<ReferenceType>())
    return M::success();
  return M::failure();
}

// Reference<T>

ReferenceType fir::ReferenceType::get(M::Type elementType) {
  return Base::get(elementType.getContext(), FIR_REFERENCE, elementType);
}

M::Type fir::ReferenceType::getEleTy() const {
  return getImpl()->getElementType();
}

M::LogicalResult fir::ReferenceType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  if (eleTy.dyn_cast<DimsType>() || eleTy.dyn_cast<FieldType>() ||
      eleTy.dyn_cast<ReferenceType>() || eleTy.dyn_cast<TypeDescType>())
    return M::failure();
  return M::success();
}

// Pointer<T>

PointerType fir::PointerType::get(M::Type elementType) {
  if (!singleIndirectionLevel(elementType)) {
    assert(false && "FIXME: invalid element type");
    return {};
  }
  return Base::get(elementType.getContext(), FIR_POINTER, elementType);
}

M::Type fir::PointerType::getEleTy() const {
  return getImpl()->getElementType();
}

M::LogicalResult fir::PointerType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  if (eleTy.dyn_cast<BoxType>() || eleTy.dyn_cast<BoxCharType>() ||
      eleTy.dyn_cast<BoxProcType>() || eleTy.dyn_cast<DimsType>() ||
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<HeapType>() ||
      eleTy.dyn_cast<PointerType>() || eleTy.dyn_cast<ReferenceType>() ||
      eleTy.dyn_cast<TypeDescType>())
    return M::failure();
  return M::success();
}

// Heap<T>

HeapType fir::HeapType::get(M::Type elementType) {
  if (!singleIndirectionLevel(elementType)) {
    assert(false && "FIXME: invalid element type");
    return {};
  }
  return Base::get(elementType.getContext(), FIR_HEAP, elementType);
}

M::Type fir::HeapType::getEleTy() const { return getImpl()->getElementType(); }

M::LogicalResult fir::HeapType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  if (eleTy.dyn_cast<BoxType>() || eleTy.dyn_cast<BoxCharType>() ||
      eleTy.dyn_cast<BoxProcType>() || eleTy.dyn_cast<DimsType>() ||
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<HeapType>() ||
      eleTy.dyn_cast<PointerType>() || eleTy.dyn_cast<ReferenceType>() ||
      eleTy.dyn_cast<TypeDescType>())
    return M::failure();
  return M::success();
}

// Sequence<T>

SequenceType fir::SequenceType::get(const Shape &shape, M::Type elementType,
                                    M::AffineMapAttr map) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FIR_SEQUENCE, shape, elementType, map);
}

M::Type fir::SequenceType::getEleTy() const {
  return getImpl()->getElementType();
}

M::AffineMapAttr fir::SequenceType::getLayoutMap() const {
  return getImpl()->getLayoutMap();
}

SequenceType::Shape fir::SequenceType::getShape() const {
  return getImpl()->getShape();
}

M::LogicalResult fir::SequenceType::verifyConstructionInvariants(
    L::Optional<mlir::Location> loc, M::MLIRContext *context,
    const SequenceType::Shape &shape, M::Type eleTy, M::AffineMapAttr map) {
  // DIMENSION attribute can only be applied to an intrinsic or record type
  if (eleTy.dyn_cast<BoxType>() || eleTy.dyn_cast<BoxCharType>() ||
      eleTy.dyn_cast<BoxProcType>() || eleTy.dyn_cast<DimsType>() ||
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<HeapType>() ||
      eleTy.dyn_cast<PointerType>() || eleTy.dyn_cast<ReferenceType>() ||
      eleTy.dyn_cast<TypeDescType>() || eleTy.dyn_cast<SequenceType>())
    return M::failure();
  return M::success();
}

// compare if two shapes are equivalent
bool fir::operator==(const SequenceType::Shape &sh_1,
                     const SequenceType::Shape &sh_2) {
  if (sh_1.hasValue() != sh_2.hasValue()) {
    return false;
  }
  if (!sh_1.hasValue()) {
    return true;
  }
  auto &bnd_1 = *sh_1;
  auto &bnd_2 = *sh_2;
  if (bnd_1.size() != bnd_2.size()) {
    return false;
  }
  for (std::size_t i = 0, end = bnd_1.size(); i != end; ++i) {
    if (bnd_1[i].hasValue() != bnd_2[i].hasValue()) {
      return false;
    }
    if (bnd_1[i].hasValue() && *bnd_1[i] != *bnd_2[i]) {
      return false;
    }
  }
  return true;
}

// compute the hash of an Extent
L::hash_code fir::hash_value(const SequenceType::Extent &ext) {
  return L::hash_combine(ext.hasValue() ? *ext : 0);
}

// compute the hash of a Shape
L::hash_code fir::hash_value(const SequenceType::Shape &sh) {
  if (sh.hasValue()) {
    return L::hash_combine_range(sh->begin(), sh->end());
  }
  return L::hash_combine(0);
}

/// RecordType
///
/// This type captures a Fortran "derived type"

RecordType fir::RecordType::get(M::MLIRContext *ctxt, L::StringRef name) {
  return Base::get(ctxt, FIR_DERIVED, name);
}

void fir::RecordType::finalize(L::ArrayRef<TypePair> lenPList,
                               L::ArrayRef<TypePair> typeList) {
  getImpl()->finalize(lenPList, typeList);
}

L::StringRef fir::RecordType::getName() { return getImpl()->getName(); }

RecordType::TypeList fir::RecordType::getTypeList() {
  return getImpl()->getTypeList();
}

RecordType::TypeList fir::RecordType::getLenParamList() {
  return getImpl()->getLenParamList();
}

detail::RecordTypeStorage const *fir::RecordType::uniqueKey() const {
  return getImpl();
}

M::LogicalResult fir::RecordType::verifyConstructionInvariants(
    L::Optional<mlir::Location>, M::MLIRContext *context, L::StringRef name) {
  if (name.size() == 0)
    return M::failure();
  return M::success();
}

/// Type descriptor type
///
/// This is the type of a type descriptor object (similar to a class instance)

TypeDescType fir::TypeDescType::get(M::Type ofType) {
  assert(!ofType.dyn_cast<ReferenceType>());
  return Base::get(ofType.getContext(), FIR_TYPEDESC, ofType);
}

M::Type fir::TypeDescType::getOfTy() const { return getImpl()->getOfType(); }

M::LogicalResult fir::TypeDescType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  if (eleTy.dyn_cast<BoxType>() || eleTy.dyn_cast<BoxCharType>() ||
      eleTy.dyn_cast<BoxProcType>() || eleTy.dyn_cast<DimsType>() ||
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<ReferenceType>() ||
      eleTy.dyn_cast<TypeDescType>())
    return M::failure();
  return M::success();
}

// Implementation of the thin interface from dialect to type parser

M::Type fir::parseFirType(FIROpsDialect *dialect, L::StringRef rawData,
                          M::Location loc) {
  FIRTypeParser parser{dialect, rawData, loc};
  return parser.parseType();
}

namespace {
class TypePrinter {
public:
  void print(FIROpsDialect *dialect, M::Type ty, llvm::raw_ostream &os) {
    if (auto type = ty.dyn_cast<fir::ReferenceType>()) {
      os << "ref<";
      type.getEleTy().print(os);
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::LogicalType>()) {
      os << "logical<" << type.getFKind() << '>';
    } else if (auto type = ty.dyn_cast<fir::RealType>()) {
      os << "real<" << type.getFKind() << '>';
    } else if (auto type = ty.dyn_cast<fir::CharacterType>()) {
      os << "char<" << type.getFKind() << '>';
    } else if (auto type = ty.dyn_cast<fir::TypeDescType>()) {
      os << "tdesc<";
      type.getOfTy().print(os);
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::FieldType>()) {
      os << "field";
    } else if (auto type = ty.dyn_cast<fir::BoxType>()) {
      os << "box<";
      type.getEleTy().print(os);
      if (auto map = type.getLayoutMap()) {
        os << ", ";
        map.print(os);
      }
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::BoxCharType>()) {
      auto eleTy = type.getEleTy().cast<fir::CharacterType>();
      os << "boxchar<" << eleTy.getFKind() << '>';
    } else if (auto type = ty.dyn_cast<fir::BoxProcType>()) {
      os << "boxproc<";
      type.getEleTy().print(os);
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::DimsType>()) {
      os << "dims<" << type.getRank() << '>';
    } else if (auto type = ty.dyn_cast<fir::SequenceType>()) {
      os << "array";
      auto shape = type.getShape();
      if (shape.hasValue()) {
        printBounds(os, *shape);
      } else {
        os << "<*";
      }
      os << ':';
      type.getEleTy().print(os);
      if (auto map = type.getLayoutMap()) {
        os << ", ";
        map.print(os);
      }
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::HeapType>()) {
      os << "heap<";
      type.getEleTy().print(os);
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::PointerType>()) {
      os << "ptr<";
      type.getEleTy().print(os);
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::RecordType>()) {
      os << "type<" << type.getName();
      if (!recordTypeVisited.count(type.uniqueKey())) {
        recordTypeVisited.insert(type.uniqueKey());
        if (type.getLenParamList().size()) {
          char ch = '(';
          for (auto p : type.getLenParamList()) {
            os << ch << p.first << ':';
            p.second.print(os);
            ch = ',';
          }
          os << ')';
        }
        if (type.getTypeList().size()) {
          char ch = '{';
          for (auto p : type.getTypeList()) {
            os << ch << p.first << ':';
            p.second.print(os);
            ch = ',';
          }
          os << '}';
        }
        recordTypeVisited.erase(type.uniqueKey());
      }
      os << '>';
    } else if (auto type = ty.dyn_cast<fir::IntType>()) {
      os << "int<" << type.getFKind() << '>';
    } else if (auto type = ty.dyn_cast<fir::CplxType>()) {
      os << "complex<" << type.getFKind() << '>';
    } else {
      assert(false);
    }
  }

private:
  void printBounds(llvm::raw_ostream &os, const SequenceType::Bounds &bounds) {
    char ch = '<';
    for (auto &b : bounds) {
      if (b.hasValue()) {
        os << ch << *b;
      } else {
        os << ch << '?';
      }
      ch = 'x';
    }
  }

  // must be in a global context because the printer must be able to track
  // context through multiple recursive invocations of the mlir type printer
  static llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4>
      recordTypeVisited;
};

llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4>
    TypePrinter::recordTypeVisited; // instantiate
} // namespace

void fir::printFirType(FIROpsDialect *dialect, M::Type ty,
                       llvm::raw_ostream &os) {
  TypePrinter().print(dialect, ty, os);
}
