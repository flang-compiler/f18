//===-- lib/lower/complex-handler.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_COMPLEX_HANDLER_H_
#define FORTRAN_LOWER_COMPLEX_HANDLER_H_

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

#include "convert-type.h"
#include "optimizer/FIROps.h"
#include "optimizer/FIRType.h"

namespace Fortran::lower {
/// Provide helpers to generate Complex manipulations in FIR.

class ComplexHandler {
public:
  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  ComplexHandler(mlir::OpBuilder &b, mlir::Location l) : builder{b}, loc{l} {}

  mlir::Type getComplexPartType(fir::KindTy complexKind) {
    return convertReal(builder.getContext(), complexKind);
  }

  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag) {
    mlir::Type complexTy{fir::CplxType::get(builder.getContext(), kind)};
    mlir::Value und = builder.create<fir::UndefOp>(loc, complexTy);
    return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
  }

  // Complex part manipulation helpers
  mlir::Type getComplexPartType(mlir::Type complexType) {
    return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
  }
  mlir::Type getComplexPartType(mlir::Value cplx) {
    assert(cplx != nullptr);
    return getComplexPartType(cplx->getType());
  }

  template <Part partId>
  mlir::Value extract(mlir::Value cplx) {
    return builder.create<fir::ExtractValueOp>(loc, getComplexPartType(cplx),
                                               cplx, getPartId<partId>());
  }
  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part) {
    assert(cplx != nullptr);
    return builder.create<fir::InsertValueOp>(loc, cplx->getType(), cplx, part,
                                              getPartId<partId>());
  }

  /// Complex part access helper dynamic versions
  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
  }
  mlir::Value insertComplexPart(mlir::Value cplx, mlir::Value part,
                                bool isImagPart) {
    return isImagPart ? insert<Part::Imag>(cplx, part)
                      : insert<Part::Real>(cplx, part);
  }

  // Complex operation helpers
  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq) {
    mlir::Value real1 = extract<Part::Real>(cplx1);
    mlir::Value real2 = extract<Part::Real>(cplx2);
    mlir::Value imag1 = extract<Part::Imag>(cplx1);
    mlir::Value imag2 = extract<Part::Imag>(cplx2);

    mlir::CmpFPredicate predicate =
        eq ? mlir::CmpFPredicate::UEQ : mlir::CmpFPredicate::UNE;
    mlir::Value realCmp =
        builder.create<mlir::CmpFOp>(loc, predicate, real1, real2).getResult();
    mlir::Value imagCmp =
        builder.create<mlir::CmpFOp>(loc, predicate, imag1, imag2).getResult();

    if (eq)
      return builder.create<mlir::AndOp>(loc, realCmp, imagCmp).getResult();
    return builder.create<mlir::OrOp>(loc, realCmp, imagCmp).getResult();
  }

private:
  // Make mlir ConstantOp from template part id.
  template <Part partId>
  inline mlir::Value getPartId() {
    auto type = mlir::IntegerType::get(32, builder.getContext());
    auto attr = builder.getIntegerAttr(type, static_cast<int>(partId));
    return builder.create<mlir::ConstantOp>(loc, type, attr).getResult();
  }

  mlir::OpBuilder &builder;
  mlir::Location loc;
};

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_COMPLEX_HANDLER_H_
