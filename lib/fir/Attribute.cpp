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
#include "fir/FIRDialect.h"
#include "fir/FIRType.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace fir {
namespace detail {

/// An attribute representing a reference to a type.
struct TypeAttributeStorage : public M::AttributeStorage {
  using KeyTy = M::Type;

  TypeAttributeStorage(M::Type value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static TypeAttributeStorage *
  construct(M::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<TypeAttributeStorage>())
        TypeAttributeStorage(key);
  }

  M::Type value;
};
} // namespace detail

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

M::Attribute parseFirAttribute(FIROpsDialect *dialect,
                               M::DialectAsmParser &parser, M::Type type) {
  auto loc = parser.getNameLoc();
  L::StringRef attrName;
  if (parser.parseKeyword(&attrName)) {
    parser.emitError(loc, "expected an attribute name");
    return {};
  }

  if (attrName == ExactTypeAttr::getAttrName()) {
    M::Type type;
    if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      parser.emitError(loc, "expected a type");
    }
    return ExactTypeAttr::get(type);
  }
  if (attrName == SubclassAttr::getAttrName()) {
    M::Type type;
    if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      parser.emitError(loc, "expected a subtype");
    }
    return SubclassAttr::get(type);
  }
  if (attrName == PointIntervalAttr::getAttrName()) {
    return PointIntervalAttr::get(dialect->getContext());
  }
  if (attrName == LowerBoundAttr::getAttrName()) {
    return LowerBoundAttr::get(dialect->getContext());
  }
  if (attrName == UpperBoundAttr::getAttrName()) {
    return UpperBoundAttr::get(dialect->getContext());
  }
  if (attrName == ClosedIntervalAttr::getAttrName()) {
    return ClosedIntervalAttr::get(dialect->getContext());
  }

  L::Twine msg{"unknown FIR attribute: "};
  parser.emitError(loc, msg.concat(attrName));
  return {};
}

void printFirAttribute(FIROpsDialect *dialect, M::Attribute attr,
                       M::DialectAsmPrinter &p) {
  auto &os = p.getStream();
  if (auto exact = attr.dyn_cast<fir::ExactTypeAttr>()) {
    os << fir::ExactTypeAttr::getAttrName() << '<';
    p.printType(exact.getType());
    os << '>';
  } else if (auto sub = attr.dyn_cast<fir::SubclassAttr>()) {
    os << fir::SubclassAttr::getAttrName() << '<';
    p.printType(sub.getType());
    os << '>';
  } else if (attr.dyn_cast_or_null<fir::PointIntervalAttr>()) {
    os << fir::PointIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
    os << fir::ClosedIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::LowerBoundAttr>()) {
    os << fir::LowerBoundAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::UpperBoundAttr>()) {
    os << fir::UpperBoundAttr::getAttrName();
  } else {
    assert(false && "attribute pretty-printer is not implemented");
  }
}

} // namespace fir
