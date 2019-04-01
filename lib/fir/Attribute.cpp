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

#include "fir/Attribute.h"
#include "fir/Dialect.h"
#include "fir/Type.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace {

// Our very stripped down parser for Attributes.
// TODO: clean this up
class AttributeParser {
public:
  AttributeParser(L::StringRef rawText)
    : srcBuff{rawText}, srcPtr{rawText.begin()} {}

  M::Attribute parseAttribute(FIROpsDialect *dialect, M::Location loc) {
    skipWhitespace();
    L::StringRef keyword{lexKeyword()};
    skipWhitespace();
    auto data{srcBuff.drop_front(srcPtr - srcBuff.begin())};
    if (keyword == ExactTypeAttr::getAttrName()) {
      data = consumeAngles(loc, data);
      M::Type type{M::parseType(data, dialect->getContext())};
      return ExactTypeAttr::get(type);
    }
    if (keyword == SubclassAttr::getAttrName()) {
      data = consumeAngles(loc, data);
      M::Type type{M::parseType(data, dialect->getContext())};
      return SubclassAttr::get(type);
    }
    if (keyword == PointIntervalAttr::getAttrName()) {
      return PointIntervalAttr::get(dialect->getContext());
    }
    if (keyword == LowerBoundAttr::getAttrName()) {
      return LowerBoundAttr::get(dialect->getContext());
    }
    if (keyword == UpperBoundAttr::getAttrName()) {
      return UpperBoundAttr::get(dialect->getContext());
    }
    if (keyword == ClosedIntervalAttr::getAttrName()) {
      return ClosedIntervalAttr::get(dialect->getContext());
    }
    L::Twine msg{"unknown FIR attribute: "};
    M::emitError(loc, msg.concat(srcBuff));
    return {};
  }

private:
  bool atEnd() const { return srcPtr == srcBuff.end(); }

  L::StringRef lexKeyword() {
    const char *tokStart = srcPtr;
    while (!atEnd() && (std::isalnum(*srcPtr) || *srcPtr == '_')) {
      ++srcPtr;
    }
    return {tokStart, static_cast<std::size_t>(srcPtr - tokStart)};
  }

  void skipWhitespace() {
    while (!atEnd()) {
      switch (*srcPtr) {
      case ' ':
      case '\f':
      case '\n':
      case '\r':
      case '\t':
      case '\v': ++srcPtr; continue;
      default: break;
      }
      break;
    }
  }

  L::StringRef consumeAngles(M::Location loc, L::StringRef data) {
    data = data.ltrim().rtrim();
    if (data.size() <= 2) {
      M::emitError(loc, "expecting '<' type '>' in attribute");
      return {};
    }
    if (data.front() != '<') {
      M::emitError(loc, "expecting '<' in attribute");
      return {};
    }
    data = data.drop_front(1);
    if (data.back() != '>') {
      M::emitError(loc, "expecting '>' in attribute");
      return {};
    }
    data = data.drop_back(1);
    return data;
  }

  L::StringRef srcBuff;
  const char *srcPtr;
};

}  // namespace

namespace fir {
namespace detail {

/// An attribute representing a reference to a type.
struct TypeAttributeStorage : public M::AttributeStorage {
  using KeyTy = M::Type;

  TypeAttributeStorage(M::Type value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static TypeAttributeStorage *construct(
      M::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<TypeAttributeStorage>())
        TypeAttributeStorage(key);
  }

  M::Type value;
};
}  // detail

ExactTypeAttr ExactTypeAttr::get(M::Type value) {
  return Base::get(value.getContext(), FIR_EXACTTYPE, value);
}

M::Type ExactTypeAttr::getType() const { return getImpl()->value; }

SubclassAttr SubclassAttr::get(M::Type value) {
  return Base::get(value.getContext(), FIR_SUBCLASS, value);
}

M::Type SubclassAttr::getType() const { return getImpl()->value; }

using AttributeUniquer = mlir::detail::AttributeUniquer;

ClosedIntervalAttr ClosedIntervalAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<ClosedIntervalAttr>(ctxt, getId());
}
UpperBoundAttr UpperBoundAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<UpperBoundAttr>(ctxt, getId());
}
LowerBoundAttr LowerBoundAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<LowerBoundAttr>(ctxt, getId());
}
PointIntervalAttr PointIntervalAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<PointIntervalAttr>(ctxt, getId());
}

M::Attribute parseFirAttribute(FIROpsDialect *dialect, L::StringRef rawText,
    M::Type type, M::Location loc) {
  AttributeParser parser{rawText};
  return parser.parseAttribute(dialect, loc);
}

}  // fir
