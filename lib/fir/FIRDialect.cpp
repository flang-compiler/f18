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

#include "fir/FIRDialect.h"
#include "fir/Attribute.h"
#include "fir/FIROps.h"
#include "fir/FIRType.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/SideEffectsInterface.h"

namespace M = mlir;

using namespace fir;

namespace {

template <typename A>
void selectBuild(M::OpBuilder *builder, M::OperationState *result,
                 M::Value *condition,
                 llvm::ArrayRef<typename A::BranchTuple> tuples) {
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

M::DialectRegistration<fir::FIROpsDialect> FIROps;

} // namespace

fir::FIROpsDialect::FIROpsDialect(M::MLIRContext *ctx)
    : M::Dialect("fir", ctx) {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, CplxType, DimsType,
           FieldType, HeapType, IntType, LenType, LogicalType, PointerType,
           RealType, RecordType, ReferenceType, SequenceType, TypeDescType>();
  addAttributes<ClosedIntervalAttr, ExactTypeAttr, LowerBoundAttr,
                PointIntervalAttr, SubclassAttr, UpperBoundAttr>();
  addOperations<GlobalOp, DispatchTableOp,
#define GET_OP_LIST
#include "fir/FIROps.cpp.inc"
                >();
}

// anchor the class vtable to this compilation unit
fir::FIROpsDialect::~FIROpsDialect() {
  // do nothing
}

M::Type fir::FIROpsDialect::parseType(M::DialectAsmParser &parser) const {
  return parseFirType(const_cast<FIROpsDialect *>(this), parser);
}

void fir::FIROpsDialect::printType(M::Type ty, M::DialectAsmPrinter &p) const {
  return printFirType(const_cast<FIROpsDialect *>(this), ty, p);
}

M::Attribute fir::FIROpsDialect::parseAttribute(M::DialectAsmParser &parser,
                                                M::Type type) const {
  return parseFirAttribute(const_cast<FIROpsDialect *>(this), parser, type);
}

void fir::FIROpsDialect::printAttribute(M::Attribute attr,
                                        M::DialectAsmPrinter &p) const {
  printFirAttribute(const_cast<FIROpsDialect *>(this), attr, p);
}
