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

#ifndef FORTRAN_BURNSIDE_COMPLEX_H_
#define FORTRAN_BURNSIDE_COMPLEX_H_

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

#include "fe-helper.h"
#include "fir/FIROps.h"
#include "fir/Type.h"

namespace Fortran::burnside {
/// Provide helpers to generate Complex manipulations in FIR.

class ComplexHandler {
public:
  ComplexHandler(mlir::OpBuilder &b, mlir::Location l) : builder{b}, loc{l} {}
  mlir::Type getComplexPartType(fir::KindTy complexKind) {
    return convertReal(builder.getContext(), complexKind);
  }

  mlir::Type getComplexPartType(mlir::Type complexType) {
    return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
  }

  mlir::Type getComplexPartType(mlir::Value *cplx) {
    assert(cplx != nullptr);
    return getComplexPartType(cplx->getType());
  }

  mlir::Value *createComplexPart(mlir::Value *cplx, bool isImaginaryPart) {
    return builder.create<fir::ExtractValueOp>(
        loc, getComplexPartType(cplx), cplx, getPartId(isImaginaryPart));
  }

  mlir::Value *createComplexRealPart(mlir::Value *cplx) {
    return createComplexPart(cplx, false);
  }

  mlir::Value *createComplexImagPart(mlir::Value *cplx) {
    return createComplexPart(cplx, true);
  }

  mlir::Value *setComplexPart(
      mlir::Value *cplx, mlir::Value *part, bool isImaginaryPart) {
    assert(cplx != nullptr);
    return builder.create<fir::InsertValueOp>(
        loc, cplx->getType(), cplx, part, getPartId(isImaginaryPart));
  }
  mlir::Value *setRealPart(mlir::Value *cplx, mlir::Value *part) {
    return setComplexPart(cplx, part, false);
  }
  mlir::Value *setImagPart(mlir::Value *cplx, mlir::Value *part) {
    return setComplexPart(cplx, part, true);
  }

  mlir::Value *createComplex(
      fir::KindTy kind, mlir::Value *real, mlir::Value *imag) {
    mlir::Type complexTy{fir::CplxType::get(builder.getContext(), kind)};
    mlir::Value *und{builder.create<fir::UndefOp>(loc, complexTy)};
    return setImagPart(setRealPart(und, real), imag);
  }

  mlir::Value *createComplexCompare(
      mlir::Value *cplx1, mlir::Value *cplx2, bool eq) {
    mlir::Value *real1{createComplexRealPart(cplx1)};
    mlir::Value *real2{createComplexRealPart(cplx2)};
    mlir::Value *imag1{createComplexImagPart(cplx1)};
    mlir::Value *imag2{createComplexImagPart(cplx2)};

    mlir::CmpFPredicate predicate{
        eq ? mlir::CmpFPredicate::UEQ : mlir::CmpFPredicate::UNE};
    mlir::Value *realCmp{
        builder.create<mlir::CmpFOp>(loc, predicate, real1, real2).getResult()};
    mlir::Value *imagCmp{
        builder.create<mlir::CmpFOp>(loc, predicate, imag1, imag2).getResult()};

    if (eq) {
      return builder.create<mlir::AndOp>(loc, realCmp, imagCmp).getResult();
    } else {
      return builder.create<mlir::OrOp>(loc, realCmp, imagCmp).getResult();
    }
  }

private:
  inline mlir::Value *getPartId(bool isImaginaryPart) {
    auto type{mlir::IntegerType::get(32, builder.getContext())};
    auto attr{builder.getIntegerAttr(type, isImaginaryPart ? 1 : 0)};
    return builder.create<mlir::ConstantOp>(loc, type, attr).getResult();
  }

  mlir::OpBuilder &builder;
  mlir::Location loc;
};

}
#endif  // FORTRAN_BURNSIDE_COMPLEX_H_
