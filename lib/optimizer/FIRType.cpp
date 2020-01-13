//===-- lib/fir/FIRType.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "optimizer/FIRType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "optimizer/FIRDialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace {

template <typename TYPE>
TYPE parseIntSingleton(M::DialectAsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "kind value expected");
    return {};
  }
  return TYPE::get(parser.getBuilder().getContext(), kind);
}

template <typename TYPE>
TYPE parseKindSingleton(M::DialectAsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseRankSingleton(M::DialectAsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseTypeSingleton(M::DialectAsmParser &parser, M::Location) {
  M::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "type expected");
    return {};
  }
  return TYPE::get(ty);
}

// `box` `<` type (',' affine-map)? `>`
BoxType parseBox(M::DialectAsmParser &parser, M::Location loc) {
  M::Type ofTy;
  if (parser.parseLess() || parser.parseType(ofTy)) {
    parser.emitError(parser.getCurrentLocation(), "expected type parameter");
    return {};
  }

  M::AffineMapAttr map;
  if (!parser.parseOptionalComma())
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getCurrentLocation(), "expected affine map");
      return {};
    }
  if (parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "expected '>'");
    return {};
  }
  return BoxType::get(ofTy, map);
}

// `boxchar` `<` kind `>`
BoxCharType parseBoxChar(M::DialectAsmParser &parser) {
  return parseKindSingleton<BoxCharType>(parser);
}

// `boxproc` `<` return-type `>`
BoxProcType parseBoxProc(M::DialectAsmParser &parser, M::Location loc) {
  return parseTypeSingleton<BoxProcType>(parser, loc);
}

// `char` `<` kind `>`
CharacterType parseCharacter(M::DialectAsmParser &parser) {
  return parseKindSingleton<CharacterType>(parser);
}

// `complex` `<` kind `>`
CplxType parseComplex(M::DialectAsmParser &parser) {
  return parseKindSingleton<CplxType>(parser);
}

// `dims` `<` rank `>`
DimsType parseDims(M::DialectAsmParser &parser) {
  return parseRankSingleton<DimsType>(parser);
}

// `field`
FieldType parseField(M::DialectAsmParser &parser) {
  return FieldType::get(parser.getBuilder().getContext());
}

// `heap` `<` type `>`
HeapType parseHeap(M::DialectAsmParser &parser, M::Location loc) {
  return parseTypeSingleton<HeapType>(parser, loc);
}

// `int` `<` kind `>`
IntType parseInteger(M::DialectAsmParser &parser) {
  return parseKindSingleton<IntType>(parser);
}

// `len`
LenType parseLen(M::DialectAsmParser &parser) {
  return LenType::get(parser.getBuilder().getContext());
}

// `logical` `<` kind `>`
LogicalType parseLogical(M::DialectAsmParser &parser) {
  return parseKindSingleton<LogicalType>(parser);
}

// `ptr` `<` type `>`
PointerType parsePointer(M::DialectAsmParser &parser, M::Location loc) {
  return parseTypeSingleton<PointerType>(parser, loc);
}

// `real` `<` kind `>`
RealType parseReal(M::DialectAsmParser &parser) {
  return parseKindSingleton<RealType>(parser);
}

// `ref` `<` type `>`
ReferenceType parseReference(M::DialectAsmParser &parser, M::Location loc) {
  return parseTypeSingleton<ReferenceType>(parser, loc);
}

// `tdesc` `<` type `>`
TypeDescType parseTypeDesc(M::DialectAsmParser &parser, M::Location loc) {
  return parseTypeSingleton<TypeDescType>(parser, loc);
}

// `void`
M::Type parseVoid(M::DialectAsmParser &parser) {
  return parser.getBuilder().getNoneType();
}

// `array` `<` `*` | bounds (`x` bounds)* `:` type (',' affine-map)? `>`
// bounds ::= `?` | int-lit
SequenceType parseSequence(M::DialectAsmParser &parser, M::Location) {
  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expecting '<'");
    return {};
  }
  SequenceType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, true)) {
      parser.emitError(parser.getNameLoc(), "invalid shape");
      return {};
    }
  } else if (parser.parseColon()) {
    parser.emitError(parser.getNameLoc(), "expected ':'");
    return {};
  }
  M::Type eleTy;
  if (parser.parseType(eleTy) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expecting element type");
    return {};
  }
  M::AffineMapAttr map;
  if (!parser.parseOptionalComma())
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getNameLoc(), "expecting affine map");
      return {};
    }
  return SequenceType::get(shape, eleTy, map);
}

bool verifyIntegerType(M::Type ty) {
  return ty.dyn_cast<M::IntegerType>() || ty.dyn_cast<IntType>();
}

bool verifyRecordMemberType(M::Type ty) {
  return !(ty.dyn_cast<BoxType>() || ty.dyn_cast<BoxCharType>() ||
           ty.dyn_cast<BoxProcType>() || ty.dyn_cast<DimsType>() ||
           ty.dyn_cast<FieldType>() || ty.dyn_cast<LenType>() ||
           ty.dyn_cast<ReferenceType>() || ty.dyn_cast<TypeDescType>());
}

bool verifySameLists(L::ArrayRef<RecordType::TypePair> a1,
                     L::ArrayRef<RecordType::TypePair> a2) {
  // FIXME: do we need to allow for any variance here?
  return a1 == a2;
}

RecordType verifyDerived(M::DialectAsmParser &parser, RecordType derivedTy,
                         L::ArrayRef<RecordType::TypePair> lenPList,
                         L::ArrayRef<RecordType::TypePair> typeList) {
  auto loc = parser.getNameLoc();
  if (!verifySameLists(derivedTy.getLenParamList(), lenPList) ||
      !verifySameLists(derivedTy.getTypeList(), typeList)) {
    parser.emitError(loc, "cannot redefine record type members");
    return {};
  }
  for (auto &p : lenPList)
    if (!verifyIntegerType(p.second)) {
      parser.emitError(loc, "LEN parameter must be integral type");
      return {};
    }
  for (auto &p : typeList)
    if (!verifyRecordMemberType(p.second)) {
      parser.emitError(loc, "field parameter has invalid type");
      return {};
    }
  llvm::StringSet<> uniq;
  for (auto &p : lenPList)
    if (!uniq.insert(p.first).second) {
      parser.emitError(loc, "LEN parameter cannot have duplicate name");
      return {};
    }
  for (auto &p : typeList)
    if (!uniq.insert(p.first).second) {
      parser.emitError(loc, "field cannot have duplicate name");
      return {};
    }
  return derivedTy;
}

// Fortran derived type
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`{` id `:` type (`,` id `:` type)* `}`)? '>'
RecordType parseDerived(M::DialectAsmParser &parser, M::Location) {
  L::StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name)) {
    parser.emitError(parser.getNameLoc(),
                     "expected a identifier as name of derived type");
    return {};
  }
  RecordType result = RecordType::get(parser.getBuilder().getContext(), name);

  RecordType::TypeList lenParamList;
  if (!parser.parseOptionalLParen()) {
    while (true) {
      L::StringRef lenparam;
      M::Type intTy;
      if (parser.parseKeyword(&lenparam) || parser.parseColon() ||
          parser.parseType(intTy)) {
        parser.emitError(parser.getNameLoc(), "expected LEN parameter list");
        return {};
      }
      lenParamList.emplace_back(lenparam, intTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRParen()) {
      parser.emitError(parser.getNameLoc(), "expected ')'");
      return {};
    }
  }

  RecordType::TypeList typeList;
  if (!parser.parseOptionalLBrace()) {
    while (true) {
      L::StringRef field;
      M::Type fldTy;
      if (parser.parseKeyword(&field) || parser.parseColon() ||
          parser.parseType(fldTy)) {
        parser.emitError(parser.getNameLoc(), "expected field type list");
        return {};
      }
      typeList.emplace_back(field, fldTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRBrace()) {
      parser.emitError(parser.getNameLoc(), "expected '}'");
      return {};
    }
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>' in type type");
    return {};
  }

  if (lenParamList.empty() && typeList.empty())
    return result;

  result.finalize(lenParamList, typeList);
  return verifyDerived(parser, result, lenParamList, typeList);
}

// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
bool singleIndirectionLevel(M::Type ty) {
  return !(ty.dyn_cast<ReferenceType>() || ty.dyn_cast<PointerType>() ||
           ty.dyn_cast<HeapType>());
}

} // namespace

// Implementation of the thin interface from dialect to type parser

M::Type fir::parseFirType(FIROpsDialect *, M::DialectAsmParser &parser) {
  L::StringRef typeNameLit;
  if (M::failed(parser.parseKeyword(&typeNameLit)))
    return {};

  auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  if (typeNameLit == "array")
    return parseSequence(parser, loc);
  if (typeNameLit == "box")
    return parseBox(parser, loc);
  if (typeNameLit == "boxchar")
    return parseBoxChar(parser);
  if (typeNameLit == "boxproc")
    return parseBoxProc(parser, loc);
  if (typeNameLit == "char")
    return parseCharacter(parser);
  if (typeNameLit == "complex")
    return parseComplex(parser);
  if (typeNameLit == "dims")
    return parseDims(parser);
  if (typeNameLit == "field")
    return parseField(parser);
  if (typeNameLit == "heap")
    return parseHeap(parser, loc);
  if (typeNameLit == "int")
    return parseInteger(parser);
  if (typeNameLit == "len")
    return parseLen(parser);
  if (typeNameLit == "logical")
    return parseLogical(parser);
  if (typeNameLit == "ptr")
    return parsePointer(parser, loc);
  if (typeNameLit == "real")
    return parseReal(parser);
  if (typeNameLit == "ref")
    return parseReference(parser, loc);
  if (typeNameLit == "tdesc")
    return parseTypeDesc(parser, loc);
  if (typeNameLit == "type")
    return parseDerived(parser, loc);
  if (typeNameLit == "void")
    return parseVoid(parser);

  parser.emitError(parser.getNameLoc(), "unknown FIR type " + typeNameLit);
  return {};
}

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

/// The type of a derived type LEN parameter reference
struct LenTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &) { return L::hash_combine(0); }

  bool operator==(const KeyTy &) const { return true; }

  static LenTypeStorage *construct(M::TypeStorageAllocator &allocator, KindTy) {
    auto *storage = allocator.allocate<LenTypeStorage>();
    return new (storage) LenTypeStorage{0};
  }

private:
  LenTypeStorage() = delete;
  explicit LenTypeStorage(KindTy) {}
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

// Len

LenType fir::LenType::get(M::MLIRContext *ctxt, KindTy) {
  return Base::get(ctxt, FIR_LEN, 0);
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
      eleTy.dyn_cast<LenType>() || eleTy.dyn_cast<ReferenceType>() ||
      eleTy.dyn_cast<TypeDescType>())
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
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<LenType>() ||
      eleTy.dyn_cast<HeapType>() || eleTy.dyn_cast<PointerType>() ||
      eleTy.dyn_cast<ReferenceType>() || eleTy.dyn_cast<TypeDescType>())
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
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<LenType>() ||
      eleTy.dyn_cast<HeapType>() || eleTy.dyn_cast<PointerType>() ||
      eleTy.dyn_cast<ReferenceType>() || eleTy.dyn_cast<TypeDescType>())
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
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<LenType>() ||
      eleTy.dyn_cast<HeapType>() || eleTy.dyn_cast<PointerType>() ||
      eleTy.dyn_cast<ReferenceType>() || eleTy.dyn_cast<TypeDescType>() ||
      eleTy.dyn_cast<SequenceType>())
    return M::failure();
  return M::success();
}

// compare if two shapes are equivalent
bool fir::operator==(const SequenceType::Shape &sh_1,
                     const SequenceType::Shape &sh_2) {
  if (sh_1.size() != sh_2.size())
    return false;
  for (std::size_t i = 0, e = sh_1.size(); i != e; ++i)
    if (sh_1[i] != sh_2[i])
      return false;
  return true;
}

// compute the hash of a Shape
L::hash_code fir::hash_value(const SequenceType::Shape &sh) {
  if (sh.size()) {
    return L::hash_combine_range(sh.begin(), sh.end());
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

M::Type fir::RecordType::getType(L::StringRef ident) {
  for (auto f : getTypeList())
    if (ident == f.first)
      return f.second;
  assert(false && "query for field not present in record");
  return {};
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
      eleTy.dyn_cast<FieldType>() || eleTy.dyn_cast<LenType>() ||
      eleTy.dyn_cast<ReferenceType>() || eleTy.dyn_cast<TypeDescType>())
    return M::failure();
  return M::success();
}

namespace {

void printBounds(llvm::raw_ostream &os, const SequenceType::Shape &bounds) {
  os << '<';
  for (auto &b : bounds) {
    if (b >= 0) {
      os << b << 'x';
    } else {
      os << "?x";
    }
  }
}

llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4> recordTypeVisited;

} // namespace

void fir::printFirType(FIROpsDialect *, M::Type ty, M::DialectAsmPrinter &p) {
  auto &os = p.getStream();
  switch (ty.getKind()) {
  case fir::FIR_BOX: {
    auto type = ty.cast<BoxType>();
    os << "box<";
    p.printType(type.getEleTy());
    if (auto map = type.getLayoutMap()) {
      os << ", ";
      p.printAttribute(map);
    }
    os << '>';
  } break;
  case fir::FIR_BOXCHAR: {
    auto type = ty.cast<BoxCharType>().getEleTy();
    os << "boxchar<" << type.cast<fir::CharacterType>().getFKind() << '>';
  } break;
  case fir::FIR_BOXPROC:
    os << "boxproc<";
    p.printType(ty.cast<BoxProcType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_CHARACTER: // intrinsic
    os << "char<" << ty.cast<CharacterType>().getFKind() << '>';
    break;
  case fir::FIR_COMPLEX: // intrinsic
    os << "complex<" << ty.cast<CplxType>().getFKind() << '>';
    break;
  case fir::FIR_DERIVED: { // derived
    auto type = ty.cast<fir::RecordType>();
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
  } break;
  case fir::FIR_DIMS:
    os << "dims<" << ty.cast<DimsType>().getRank() << '>';
    break;
  case fir::FIR_FIELD:
    os << "field";
    break;
  case fir::FIR_HEAP:
    os << "heap<";
    p.printType(ty.cast<HeapType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_INT: // intrinsic
    os << "int<" << ty.cast<fir::IntType>().getFKind() << '>';
    break;
  case fir::FIR_LEN:
    os << "len";
    break;
  case fir::FIR_LOGICAL: // intrinsic
    os << "logical<" << ty.cast<LogicalType>().getFKind() << '>';
    break;
  case fir::FIR_POINTER:
    os << "ptr<";
    p.printType(ty.cast<PointerType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_REAL: // intrinsic
    os << "real<" << ty.cast<fir::RealType>().getFKind() << '>';
    break;
  case fir::FIR_REFERENCE:
    os << "ref<";
    p.printType(ty.cast<ReferenceType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_SEQUENCE: {
    os << "array";
    auto type = ty.cast<SequenceType>();
    auto shape = type.getShape();
    if (shape.size()) {
      printBounds(os, shape);
    } else {
      os << "<*:";
    }
    p.printType(ty.cast<SequenceType>().getEleTy());
    if (auto map = type.getLayoutMap()) {
      os << ", ";
      map.print(os);
    }
    os << '>';
  } break;
  case fir::FIR_TYPEDESC:
    os << "tdesc<";
    p.printType(ty.cast<TypeDescType>().getOfTy());
    os << '>';
    break;
  }
}
