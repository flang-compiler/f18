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

#include "convert-expr.h"
#include "builder.h"
#include "complex.h"
#include "fe-helper.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "intrinsics.h"
#include "runtime.h"
#include "../common/default-kinds.h"
#include "../common/unwrap.h"
#include "../evaluate/fold.h"
#include "../evaluate/real.h"
#include "../semantics/expression.h"
#include "../semantics/symbol.h"
#include "../semantics/type.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

#define TODO() \
  assert(false); \
  return {}

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
  M::Location location;
  M::OpBuilder &builder;
  SomeExpr const &expr;
  SymMap &symMap;
  SymMap loadedSymbols{};
  Co::IntrinsicTypeDefaultKinds const &defaults;
  IntrinsicLibrary const &intrinsics;

  M::Location getLoc() { return location; }

  /// Convert parser's INTEGER relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static M::CmpIPredicate translateRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT: return M::CmpIPredicate::SLT;
    case Co::RelationalOperator::LE: return M::CmpIPredicate::SLE;
    case Co::RelationalOperator::EQ: return M::CmpIPredicate::EQ;
    case Co::RelationalOperator::NE: return M::CmpIPredicate::NE;
    case Co::RelationalOperator::GT: return M::CmpIPredicate::SGT;
    case Co::RelationalOperator::GE: return M::CmpIPredicate::SGE;
    }
    assert(false && "unhandled INTEGER relational operator");
    return {};
  }

  /// Convert parser's REAL relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static M::CmpFPredicate translateFloatRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT: return M::CmpFPredicate::ULT;
    case Co::RelationalOperator::LE: return M::CmpFPredicate::ULE;
    case Co::RelationalOperator::EQ: return M::CmpFPredicate::UEQ;
    case Co::RelationalOperator::NE: return M::CmpFPredicate::UNE;
    case Co::RelationalOperator::GT: return M::CmpFPredicate::UGT;
    case Co::RelationalOperator::GE: return M::CmpFPredicate::UGE;
    }
    assert(false && "unhandled REAL relational operator");
    return {};
  }

  /// Generate an integral constant of `value`
  template<int KIND>
  M::Value *genIntegerConstant(M::MLIRContext *context, std::int64_t value) {
    M::Type type{getFIRType(context, defaults, IntegerCat, KIND)};
    auto attr{builder.getIntegerAttr(type, value)};
    auto res{builder.create<M::ConstantOp>(getLoc(), type, attr)};
    return res.getResult();
  }

  /// Generate a logical/boolean constant of `value`
  template<int KIND>
  M::Value *genLogicalConstant(M::MLIRContext *context, bool value) {
    auto attr{builder.getBoolAttr(value)};
    M::Type logTy{getFIRType(context, defaults, LogicalCat, KIND)};
    auto res{builder.create<M::ConstantOp>(getLoc(), logTy, attr)};
    return res.getResult();
  }

  template<int KIND>
  M::Value *genRealConstant(M::MLIRContext *context, L::APFloat const &value) {
    M::Type fltTy{convertReal(context, KIND)};
    auto attr{builder.getFloatAttr(fltTy, value)};
    auto res{builder.create<M::ConstantOp>(getLoc(), fltTy, attr)};
    return res.getResult();
  }

  M::Type getSomeKindInteger() {
    return M::IndexType::get(builder.getContext());
  }

  template<typename OpTy, typename A>
  M::Value *createBinaryOp(A const &ex, M::Value *lhs, M::Value *rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  M::Value *createBinaryOp(A const &ex, M::Value *rhs) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), rhs);
  }
  template<typename OpTy, typename A> M::Value *createBinaryOp(A const &ex) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), genval(ex.right()));
  }

  M::FuncOp getFunction(L::StringRef name, M::FunctionType funTy) {
    auto module{getModule(&builder)};
    if (M::FuncOp func{getNamedFunction(module, name)}) {
      assert(func.getType() == funTy &&
          "function already declared with a different type");
      return func;
    }
    return createFunction(module, name, funTy);
  }

  M::FuncOp getRuntimeFunction(RuntimeEntryCode callee, M::FunctionType funTy) {
    auto name{getRuntimeEntryName(callee)};
    return getFunction(name, funTy);
  }

  // FIXME binary operation :: ('a, 'a) -> 'a
  template<Co::TypeCategory TC, int KIND> M::FunctionType createFunctionType() {
    if constexpr (TC == IntegerCat) {
      M::Type output{
          getFIRType(builder.getContext(), defaults, IntegerCat, KIND)};
      L::SmallVector<M::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return M::FunctionType::get(inputs, output, builder.getContext());
    } else if constexpr (TC == RealCat) {
      M::Type output{convertReal(builder.getContext(), KIND)};
      L::SmallVector<M::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return M::FunctionType::get(inputs, output, builder.getContext());
    } else {
      assert(false);
      return {};
    }
  }

  /// Create a call to a Fortran runtime entry point
  template<Co::TypeCategory TC, int KIND, typename A>
  M::Value *createBinaryFIRTCall(A const &ex, RuntimeEntryCode callee) {
    L::SmallVector<M::Value *, 2> operands;
    operands.push_back(genval(ex.left()));
    operands.push_back(genval(ex.right()));
    M::FunctionType funTy = createFunctionType<TC, KIND>();
    auto func{getRuntimeFunction(callee, funTy)};
    auto x{builder.create<M::CallOp>(getLoc(), func, operands)};
    return x.getResult(0);  // FIXME
  }

  template<typename OpTy, typename A>
  M::Value *createCompareOp(
      A const &ex, M::CmpIPredicate pred, M::Value *lhs, M::Value *rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  M::Value *createCompareOp(A const &ex, M::CmpIPredicate pred) {
    return createCompareOp<OpTy>(
        ex, pred, genval(ex.left()), genval(ex.right()));
  }
  template<typename OpTy, typename A>
  M::Value *createFltCmpOp(
      A const &ex, M::CmpFPredicate pred, M::Value *lhs, M::Value *rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  M::Value *createFltCmpOp(A const &ex, M::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(
        ex, pred, genval(ex.left()), genval(ex.right()));
  }

  M::Value *gen(Se::Symbol const *sym) {
    // FIXME: not all symbols are local
    return createTemporary(getLoc(), builder, symMap,
        translateSymbolToFIRType(builder.getContext(), defaults, sym), sym);
  }
  M::Value *gendef(Se::Symbol const *sym) { return gen(sym); }
  M::Value *genval(Se::Symbol const *sym) {
    // Do not load the same symbols several time in one expression.
    // Fortran guarantees variable value must be the same wherever it
    // appears in one expression.
    if (mlir::Value * loaded{loadedSymbols.lookupSymbol(sym)}) {
      return loaded;
    } else {
      mlir::Value *load{builder.create<fir::LoadOp>(getLoc(), gen(sym))};
      loadedSymbols.addSymbol(sym, load);
      return load;
    }
  }

  M::Value *genval(Ev::BOZLiteralConstant const &) { TODO(); }
  M::Value *genval(Ev::ProcedureRef const &) { TODO(); }
  M::Value *genval(Ev::ProcedureDesignator const &) { TODO(); }
  M::Value *genval(Ev::NullPointer const &) { TODO(); }
  M::Value *genval(Ev::StructureConstructor const &) { TODO(); }
  M::Value *genval(Ev::ImpliedDoIndex const &) { TODO(); }
  M::Value *genval(Ev::DescriptorInquiry const &) { TODO(); }
  template<int KIND> M::Value *genval(Ev::TypeParamInquiry<KIND> const &) {
    TODO();
  }

  template<int KIND> M::Value *genval(Ev::ComplexComponent<KIND> const &part) {
    return ComplexHandler{builder, getLoc()}.createComplexPart(
        genval(part.left()), part.isImaginaryPart);
  }

  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Negate<Ev::Type<TC, KIND>> const &) {
    TODO();
  }

  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Add<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::AddIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::AddfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::AddcOp>(op);
    }
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Subtract<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::SubIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::SubfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::SubcOp>(op);
    }
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Multiply<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::MulIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::MulfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::MulcOp>(op);
    }
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Divide<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::DivISOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::DivfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::DivcOp>(op);
    }
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Power<Ev::Type<TC, KIND>> const &op) {
    llvm::SmallVector<mlir::Value *, 2> operands{
        genval(op.left()), genval(op.right())};
    M::Type ty{getFIRType(builder.getContext(), defaults, TC, KIND)};
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::RealToIntPower<Ev::Type<TC, KIND>> const &op) {
    // TODO: runtime as limited integer kind support. Look if the conversions
    // are ok
    llvm::SmallVector<mlir::Value *, 2> operands{
        genval(op.left()), genval(op.right())};
    M::Type ty{getFIRType(builder.getContext(), defaults, TC, KIND)};
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }

  template<int KIND> M::Value *genval(Ev::ComplexConstructor<KIND> const &op) {
    return ComplexHandler{builder, getLoc()}.createComplex(
        KIND, genval(op.left()), genval(op.right()));
  }
  template<int KIND> M::Value *genval(Ev::Concat<KIND> const &op) {
    // TODO this is a bogus call
    return createBinaryFIRTCall<CharacterCat, KIND>(op, FIRT_CONCAT);
  }

  /// MIN and MAX operations
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Extremum<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryFIRTCall<TC, KIND>(
          op, op.ordering == Ev::Ordering::Greater ? FIRT_MAX : FIRT_MIN);
    } else {
      TODO();
    }
  }

  template<int KIND> M::Value *genval(Ev::SetLength<KIND> const &) { TODO(); }

  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Relational<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createCompareOp<M::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == RealCat) {
      return createFltCmpOp<M::CmpFOp>(op, translateFloatRelational(op.opr));
    } else if constexpr (TC == ComplexCat) {
      bool eq{op.opr == Co::RelationalOperator::EQ};
      assert(eq ||
          op.opr == Co::RelationalOperator::NE &&
              "relation undefined for complex");
      return ComplexHandler{builder, getLoc()}.createComplexCompare(
          genval(op.left()), genval(op.right()), eq);
    } else {
      static_assert(TC == CharacterCat);
      TODO();
    }
  }

  // TODO JP: the thing below should not be required.
  M::Value *genval(Ev::Relational<Ev::SomeType> const &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template<Co::TypeCategory TC1, int KIND, Co::TypeCategory TC2>
  M::Value *genval(Ev::Convert<Ev::Type<TC1, KIND>, TC2> const &convert) {
    auto ty{getFIRType(builder.getContext(), defaults, TC1, KIND)};
    return builder.create<fir::ConvertOp>(getLoc(), ty, genval(convert.left()));
  }
  template<typename A> M::Value *genval(Ev::Parentheses<A> const &) { TODO(); }
  template<int KIND> M::Value *genval(const Ev::Not<KIND> &op) {
    auto *context{builder.getContext()};
    return createBinaryOp<M::XOrOp>(op, genLogicalConstant<KIND>(context, 1));
  }

  template<int KIND> M::Value *genval(Ev::LogicalOperation<KIND> const &op) {
    switch (op.logicalOperator) {
    case Ev::LogicalOperator::And: return createBinaryOp<M::AndOp>(op);
    case Ev::LogicalOperator::Or: return createBinaryOp<M::OrOp>(op);
    case Ev::LogicalOperator::Eqv:
      return createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::EQ);
    case Ev::LogicalOperator::Neqv:
      return createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::NE);
    }
    assert(false && "unhandled logical operation");
    return {};
  }

  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(Ev::Constant<Ev::Type<TC, KIND>> const &con) {
    // TODO:
    // - character type constant
    // - array constant not handled
    // - derived type constant
    if constexpr (TC == IntegerCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value())
        return genIntegerConstant<KIND>(builder.getContext(), opt->ToInt64());
      assert(false && "integer constant has no value");
      return {};
    } else if constexpr (TC == LogicalCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value())
        return genLogicalConstant<KIND>(builder.getContext(), opt->IsTrue());
      assert(false && "logical constant has no value");
      return {};
    } else if constexpr (TC == RealCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value()) {
        std::string str{opt.value().DumpHexadecimal()};
        if constexpr (KIND == 2) {
          L::APFloat floatVal{L::APFloatBase::IEEEhalf(), str};
          return genRealConstant<KIND>(builder.getContext(), floatVal);
        } else if constexpr (KIND == 4) {
          L::APFloat floatVal{L::APFloatBase::IEEEsingle(), str};
          return genRealConstant<KIND>(builder.getContext(), floatVal);
        } else if constexpr (KIND == 10) {
          L::APFloat floatVal{L::APFloatBase::x87DoubleExtended(), str};
          return genRealConstant<KIND>(builder.getContext(), floatVal);
        } else if constexpr (KIND == 16) {
          L::APFloat floatVal{L::APFloatBase::IEEEquad(), str};
          return genRealConstant<KIND>(builder.getContext(), floatVal);
        } else {
          // convert everything else to double
          L::APFloat floatVal{L::APFloatBase::IEEEdouble(), str};
          return genRealConstant<KIND>(builder.getContext(), floatVal);
        }
      }
      assert(false && "real constant has no value");
      return {};
    } else if constexpr (TC == ComplexCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value()) {
        using TR = Ev::Type<RealCat, KIND>;
        return genval(Ev::ComplexConstructor<KIND>{
            Ev::Expr<TR>{Ev::Constant<TR>{opt->REAL()}},
            Ev::Expr<TR>{Ev::Constant<TR>{opt->AIMAG()}}});
      }
      assert(false && "array of complex unhandled");
      return {};
    } else {
      assert(false && "unhandled constant");
      return {};
    }
  }

  template<Co::TypeCategory TC>
  M::Value *genval(Ev::Constant<Ev::SomeKind<TC>> const &con) {
    if constexpr (TC == IntegerCat) {
      auto opt = (*con).ToInt64();
      M::Type type{getSomeKindInteger()};
      auto attr{builder.getIntegerAttr(type, opt)};
      auto res{builder.create<M::ConstantOp>(getLoc(), type, attr)};
      return res.getResult();
    } else {
      assert(false && "unhandled constant of unknown kind");
      return {};
    }
  }

  template<typename A> M::Value *genval(Ev::ArrayConstructor<A> const &) {
    TODO();
  }
  M::Value *gen(Ev::ComplexPart const &) { TODO(); }
  M::Value *gendef(Ev::ComplexPart const &cp) { return gen(cp); }
  M::Value *genval(Ev::ComplexPart const &) { TODO(); }
  M::Value *gen(const Ev::Substring &) { TODO(); }
  M::Value *gendef(const Ev::Substring &ss) { return gen(ss); }
  M::Value *genval(const Ev::Substring &) { TODO(); }
  M::Value *genval(const Ev::Triplet &trip) { TODO(); }

  M::Value *genval(const Ev::Subscript &subs) {
    return std::visit(Co::visitors{
                          [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                            return genval(x.value());
                          },
                          [&](const Ev::Triplet &x) { return genval(x); },
                      },
        subs.u);
  }

  M::Value *gen(const Ev::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  M::Value *gendef(Ev::DataRef const &dref) { return gen(dref); }
  M::Value *genval(Ev::DataRef const &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list.
  // Returns the object used as the base coordinate for the component chain.
  static Ev::DataRef const *reverseComponents(
      Ev::Component const &cmpt, std::list<Ev::Component const *> &list) {
    list.push_front(&cmpt);
    return std::visit(
        Co::visitors{
            [&](Ev::Component const &x) { return reverseComponents(x, list); },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  M::Value *gen(const Ev::Component &cmpt) {
    std::list<const Ev::Component *> list;
    auto *base{reverseComponents(cmpt, list)};
    L::SmallVector<M::Value *, 2> coorArgs;
    auto obj{gen(*base)};
    const Se::Symbol *sym{nullptr};
    for (auto *field : list) {
      sym = &field->GetLastSymbol();
      auto name{sym->name().ToString()};
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(getLoc(), name));
    }
    assert(sym && "no component(s)?");
    M::Type ty{translateSymbolToFIRType(builder.getContext(), defaults, sym)};
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, obj, coorArgs);
  }
  M::Value *gendef(const Ev::Component &cmpt) { return gen(cmpt); }
  M::Value *genval(const Ev::Component &cmpt) {
    return builder.create<fir::LoadOp>(getLoc(), gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  M::Type genSubType(M::Type arrTy, unsigned dims) {
    if (auto memRef{arrTy.dyn_cast<M::MemRefType>()}) {
      if (dims < memRef.getRank()) {
        auto shape{memRef.getShape()};
        llvm::SmallVector<int64_t, 4> newShape;
        // TODO: should we really remove rows here?
        for (unsigned i = dims, e = memRef.getRank(); i < e; ++i) {
          newShape.push_back(shape[i]);
        }
        return M::MemRefType::get(newShape, memRef.getElementType());
      }
      return memRef.getElementType();
    }
    auto unwrapTy{arrTy.cast<fir::ReferenceType>().getEleTy()};
    auto seqTy{unwrapTy.cast<fir::SequenceType>()};
    auto shape = seqTy.getShape();
    if (shape.hasValue()) {
      if (dims < shape->size()) {
        fir::SequenceType::Bounds newBnds;
        // follow Fortran semantics and remove columns
        for (unsigned i = 0; i < dims; ++i) {
          newBnds.push_back((*shape)[i]);
        }
        return fir::SequenceType::get({newBnds}, seqTy.getEleTy());
      }
    }
    return seqTy.getEleTy();
  }

  // Return the coordinate of the array reference
  M::Value *gen(Ev::ArrayRef const &aref) {
    M::Value *base;
    if (aref.base().IsSymbol())
      base = gen(const_cast<Se::Symbol *>(&aref.base().GetFirstSymbol()));
    else
      base = gen(aref.base().GetComponent());
    llvm::SmallVector<M::Value *, 8> args;
    for (auto &subsc : aref.subscript()) {
      args.push_back(genval(subsc));
    }
    auto ty{genSubType(base->getType(), args.size() - 1)};
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }
  M::Value *gendef(const Ev::ArrayRef &aref) { return gen(aref); }
  M::Value *genval(const Ev::ArrayRef &aref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(aref));
  }

  // Return a coordinate of the coarray reference. This is necessary as a
  // Component may have a CoarrayRef as its base coordinate.
  M::Value *gen(Ev::CoarrayRef const &coref) {
    // FIXME: need to visit the cosubscripts...
    // return gen(coref.base());
    TODO();
  }
  M::Value *gendef(const Ev::CoarrayRef &coref) { return gen(coref); }
  M::Value *genval(const Ev::CoarrayRef &coref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(coref));
  }

  template<typename A> M::Value *gen(Ev::Designator<A> const &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template<typename A> M::Value *gendef(Ev::Designator<A> const &des) {
    return gen(des);
  }
  template<typename A> M::Value *genval(Ev::Designator<A> const &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  // call a function
  template<typename A> M::Value *gen(const Ev::FunctionRef<A> &funRef) {
    TODO();
  }
  template<typename A> M::Value *gendef(const Ev::FunctionRef<A> &funRef) {
    return gen(funRef);
  }
  template<typename A> M::Value *genval(const Ev::FunctionRef<A> &funRef) {
    TODO();  // Derived type functions (user + intrinsics)
  }
  template<Co::TypeCategory TC, int KIND>
  M::Value *genval(const Ev::FunctionRef<Ev::Type<TC, KIND>> &funRef) {
    if (const auto &intrinsic{funRef.proc().GetSpecificIntrinsic()}) {
      M::Type ty{getFIRType(builder.getContext(), defaults, TC, KIND)};
      L::SmallVector<M::Value *, 2> operands;
      // Lower arguments
      for (const auto &arg : funRef.arguments()) {
        if (auto *expr{Ev::UnwrapExpr<Ev::Expr<Ev::SomeType>>(arg)}) {
          operands.push_back(genval(*expr));
        } else {
          operands.push_back(nullptr);  // optional
        }
      }
      // Let the intrinsic library lower the intrinsic function call
      L::StringRef name{intrinsic->name};
      return intrinsics.genval(getLoc(), builder, name, ty, operands);
    } else {
      // implicit interface implementation only
      // TODO: explicit interface
      L::SmallVector<M::Type, 2> argTypes;
      L::SmallVector<M::Value *, 2> operands;
      for (const auto &arg : funRef.arguments()) {
        assert(
            arg.has_value() && "optional argument requires explicit interface");
        const auto *expr{arg->UnwrapExpr()};
        assert(expr && "assumed type argument requires explicit interface");
        if (const Se::Symbol * sym{Ev::UnwrapWholeSymbolDataRef(*expr)}) {
          M::Value *argRef{symMap.lookupSymbol(sym)};
          assert(argRef && "could not get symbol reference");
          argTypes.push_back(argRef->getType());
          operands.push_back(argRef);
        } else {
          // TODO create temps for expressions
          TODO();
        }
      }
      M::Type resultType{getFIRType(builder.getContext(), defaults, TC, KIND)};
      M::FunctionType funTy{
          M::FunctionType::get(argTypes, resultType, builder.getContext())};
      M::FuncOp func{getFunction(funRef.proc().GetName(), funTy)};
      M::CallOp call{builder.create<M::CallOp>(getLoc(), func, operands)};
      return call.getResult(0);
    }
  }

  template<typename A> M::Value *gen(const Ev::Expr<A> &exp) {
    // must be a designator or function-reference (R902)
    return std::visit([&](const auto &e) { return gendef(e); }, exp.u);
  }
  template<typename A> M::Value *gendef(Ev::Expr<A> const &exp) {
    return gen(exp);
  }
  template<typename A> M::Value *genval(Ev::Expr<A> const &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  template<typename A> M::Value *gendef(const A &) {
    assert(false && "expression error");
    return {};
  }

public:
  explicit ExprLowering(M::Location loc, M::OpBuilder &bldr,
      SomeExpr const &vop, SymMap &map,
      Co::IntrinsicTypeDefaultKinds const &defaults,
      IntrinsicLibrary const &intr)
    : location{loc}, builder{bldr}, expr{vop}, symMap{map}, defaults{defaults},
      intrinsics{intr} {}

  /// Lower the expression `expr` into MLIR standard dialect
  M::Value *gen() { return gen(expr); }
  M::Value *genval() { return genval(expr); }
};

}  // namespace

M::Value *Br::createSomeExpression(M::Location loc, M::OpBuilder &builder,
    Ev::Expr<Ev::SomeType> const &expr, SymMap &symMap,
    Co::IntrinsicTypeDefaultKinds const &defaults,
    IntrinsicLibrary const &intrinsics) {
  return ExprLowering{loc, builder, expr, symMap, defaults, intrinsics}
      .genval();
}

M::Value *Br::createSomeAddress(M::Location loc, M::OpBuilder &builder,
    Ev::Expr<Ev::SomeType> const &expr, SymMap &symMap,
    Co::IntrinsicTypeDefaultKinds const &defaults,
    IntrinsicLibrary const &intrinsics) {
  return ExprLowering{loc, builder, expr, symMap, defaults, intrinsics}.gen();
}

/// Create a temporary variable
/// `symbol` will be nullptr for an anonymous temporary
M::Value *Br::createTemporary(M::Location loc, M::OpBuilder &builder,
    SymMap &symMap, M::Type type, Se::Symbol const *symbol) {
  if (symbol)
    if (auto *val{symMap.lookupSymbol(symbol)}) {
      if (auto *op{val->getDefiningOp()}) {
        return op->getResult(0);
      }
      return val;
    }
  auto insPt(builder.saveInsertionPoint());
  builder.setInsertionPointToStart(getEntryBlock(&builder));
  fir::AllocaOp ae;
  assert(!type.dyn_cast<fir::ReferenceType>() && "cannot be a reference");
  if (symbol) {
    ae = builder.create<fir::AllocaOp>(loc, type, symbol->name().ToString());
    symMap.addSymbol(symbol, ae);
  } else {
    ae = builder.create<fir::AllocaOp>(loc, type);
  }
  builder.restoreInsertionPoint(insPt);
  return ae;
}
