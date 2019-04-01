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

#include "fir/Dialect.h"
#include "fir/Attribute.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "../evaluate/expression.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"

namespace M = mlir;

using namespace fir;

namespace {

template<typename A>
void selectBuild(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<typename A::BranchTuple> tuples) {
  result->addOperands(condition);
  for (auto &tup : tuples) {
    auto *cond{std::get<typename A::Conditions>(tup)};
    result->addOperands(cond);
  }
  // Note: succs must be added *after* operands
  for (auto &tup : tuples) {
    auto *block{std::get<M::Block *>(tup)};
    assert(block);
    auto blkArgs{std::get<llvm::ArrayRef<M::Value *>>(tup)};
    result->addSuccessor(block, blkArgs);
  }
}
}  // namespace

fir::FIROpsDialect::FIROpsDialect(M::MLIRContext *ctx)
  : M::Dialect("fir", ctx) {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, CplxType, DimsType,
      FieldType, HeapType, IntType, LogicalType, PointerType, RealType,
      RecordType, ReferenceType, SequenceType, TypeDescType>();
  addAttributes<ClosedIntervalAttr, ExactTypeAttr, LowerBoundAttr,
      PointIntervalAttr, SubclassAttr, UpperBoundAttr>();
  addOperations<GlobalOp, DispatchTableOp,
#define GET_OP_LIST
#include "fir/FIROps.cpp.inc"
      >();
}

// anchor the class vtable
fir::FIROpsDialect::~FIROpsDialect() {}

M::Type fir::FIROpsDialect::parseType(
    llvm::StringRef rawData, M::Location loc) const {
  return parseFirType(const_cast<FIROpsDialect *>(this), rawData, loc);
}

void printBounds(llvm::raw_ostream &os, const SequenceType::Bounds &bounds) {
  char ch = '<';
  for (auto &b : bounds) {
    if (b.known) {
      os << ch << b.bound;
    } else {
      os << ch << '?';
    }
    ch = 'x';
  }
}

void fir::FIROpsDialect::printType(M::Type ty, llvm::raw_ostream &os) const {
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
    if (shape.known) {
      printBounds(os, shape.bounds);
    } else {
      os << "<*";
    }
    os << ':';
    type.getEleTy().print(os);
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
    os << '>';
  } else if (auto type = ty.dyn_cast<fir::IntType>()) {
    os << "int<" << type.getFKind() << '>';
  } else if (auto type = ty.dyn_cast<fir::CplxType>()) {
    os << "complex<" << type.getFKind() << '>';
  } else {
    assert(false);
  }
}

M::Attribute fir::FIROpsDialect::parseAttribute(
    llvm::StringRef rawText, M::Type type, M::Location loc) const {
  return parseFirAttribute(
      const_cast<FIROpsDialect *>(this), rawText, type, loc);
}

void fir::FIROpsDialect::printAttribute(
    M::Attribute attr, llvm::raw_ostream &os) const {
  if (auto exact = attr.dyn_cast<fir::ExactTypeAttr>()) {
    os << fir::ExactTypeAttr::getAttrName() << '<' << exact.getType() << '>';
  } else if (auto sub = attr.dyn_cast<fir::SubclassAttr>()) {
    os << fir::SubclassAttr::getAttrName() << '<' << sub.getType() << '>';
  } else if (attr.dyn_cast_or_null<fir::PointIntervalAttr>()) {
    os << fir::PointIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
    os << fir::ClosedIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::LowerBoundAttr>()) {
    os << fir::LowerBoundAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::UpperBoundAttr>()) {
    os << fir::UpperBoundAttr::getAttrName();
  } else {
    assert(false);
  }
}
