//===-- lib/lower/convert-expr.cc -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/ConvertExpr.h"
#include "../common/default-kinds.h"
#include "../common/unwrap.h"
#include "../evaluate/fold.h"
#include "../evaluate/real.h"
#include "../semantics/expression.h"
#include "../semantics/symbol.h"
#include "../semantics/type.h"
#include "fir/Dialect/FIRDialect.h"
#include "fir/Dialect/FIROps.h"
#include "fir/Dialect/FIRType.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/ConvertType.h"
#include "flang/lower/OpBuilder.h"
#include "flang/lower/Runtime.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace Br = Fortran::lower;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::lower;

namespace {

#define TODO()                                                                 \
  assert(false);                                                               \
  return {}

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
  M::Location location;
  AbstractConverter &converter;
  M::OpBuilder &builder;
  SomeExpr const &expr;
  SymMap &symMap;
  IntrinsicLibrary const &intrinsics;
  bool genLogicalAsI1{false};

  M::Location getLoc() { return location; }

  /// Convert parser's INTEGER relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static M::CmpIPredicate translateRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT:
      return M::CmpIPredicate::slt;
    case Co::RelationalOperator::LE:
      return M::CmpIPredicate::sle;
    case Co::RelationalOperator::EQ:
      return M::CmpIPredicate::eq;
    case Co::RelationalOperator::NE:
      return M::CmpIPredicate::ne;
    case Co::RelationalOperator::GT:
      return M::CmpIPredicate::sgt;
    case Co::RelationalOperator::GE:
      return M::CmpIPredicate::sge;
    }
    assert(false && "unhandled INTEGER relational operator");
    return {};
  }

  /// Convert parser's REAL relational operators to MLIR.
  /// The choice of order (O prefix) vs unorder (U prefix) follows Fortran 2018
  /// requirements in the IEEE context (table 17.1 of F2018). This choice is
  /// also applied in other contexts because it is easier and in line with
  /// other Fortran compilers.
  /// FIXME: The signaling/quiet aspect of the table 17.1 requirement is not
  /// fully enforced. FIR and LLVM `fcmp` instructions do not give any guarantee
  /// whether the comparison will signal or not in case of quiet NaN argument.
  static fir::CmpFPredicate
  translateFloatRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT:
      return fir::CmpFPredicate::OLT;
    case Co::RelationalOperator::LE:
      return fir::CmpFPredicate::OLE;
    case Co::RelationalOperator::EQ:
      return fir::CmpFPredicate::OEQ;
    case Co::RelationalOperator::NE:
      return fir::CmpFPredicate::UNE;
    case Co::RelationalOperator::GT:
      return fir::CmpFPredicate::OGT;
    case Co::RelationalOperator::GE:
      return fir::CmpFPredicate::OGE;
    }
    assert(false && "unhandled REAL relational operator");
    return {};
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  M::Value genIntegerConstant(M::MLIRContext *context, std::int64_t value) {
    M::Type type{converter.genType(IntegerCat, KIND)};
    auto attr{builder.getIntegerAttr(type, value)};
    auto res{builder.create<M::ConstantOp>(getLoc(), type, attr)};
    return res.getResult();
  }

  /// Generate a logical/boolean constant of `value`
  M::Value genLogicalConstantAsI1(M::MLIRContext *context, bool value) {
    M::Type i1Type{M::IntegerType::get(1, builder.getContext())};
    auto attr{builder.getIntegerAttr(i1Type, value ? 1 : 0)};
    return builder.create<M::ConstantOp>(getLoc(), i1Type, attr).getResult();
  }

  template <int KIND>
  M::Value genRealConstant(M::MLIRContext *context, L::APFloat const &value) {
    M::Type fltTy{convertReal(context, KIND)};
    auto attr{builder.getFloatAttr(fltTy, value)};
    auto res{builder.create<M::ConstantOp>(getLoc(), fltTy, attr)};
    return res.getResult();
  }

  M::Type getSomeKindInteger() {
    return M::IndexType::get(builder.getContext());
  }

  template <typename OpTy, typename A>
  M::Value createBinaryOp(A const &ex, M::Value lhs, M::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), lhs, rhs);
    return x.getResult();
  }
  template <typename OpTy, typename A>
  M::Value createBinaryOp(A const &ex, M::Value rhs) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), rhs);
  }
  template <typename OpTy, typename A>
  M::Value createBinaryOp(A const &ex) {
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

  // FIXME binary operation :: ('a, 'a) -> 'a
  template <Co::TypeCategory TC, int KIND>
  M::FunctionType createFunctionType() {
    if constexpr (TC == IntegerCat) {
      M::Type output{converter.genType(IntegerCat, KIND)};
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

  template <typename OpTy, typename A>
  M::Value createCompareOp(A const &ex, M::CmpIPredicate pred, M::Value lhs,
                           M::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template <typename OpTy, typename A>
  M::Value createCompareOp(A const &ex, M::CmpIPredicate pred) {
    return createCompareOp<OpTy>(ex, pred, genval(ex.left()),
                                 genval(ex.right()));
  }
  template <typename OpTy, typename A>
  M::Value createFltCmpOp(A const &ex, fir::CmpFPredicate pred, M::Value lhs,
                          M::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template <typename OpTy, typename A>
  M::Value createFltCmpOp(A const &ex, fir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(ex, pred, genval(ex.left()),
                                genval(ex.right()));
  }

  M::Value gen(Se::SymbolRef sym) {
    // FIXME: not all symbols are local
    auto addr{createTemporary(getLoc(), builder, symMap, converter.genType(sym),
                              &*sym)};
    assert(addr && "failed generating symbol address");
    // Get address from descriptor if symbol has one.
    auto type{addr.getType()};
    if (auto boxCharType{type.dyn_cast<fir::BoxCharType>()}) {
      auto refType{fir::ReferenceType::get(boxCharType.getEleTy())};
      auto lenType{mlir::IntegerType::get(64, builder.getContext())};
      addr = builder.create<fir::UnboxCharOp>(getLoc(), refType, lenType, addr)
                 .getResult(0);
    } else if (type.isa<fir::BoxType>()) {
      TODO();
    }
    return addr;
  }

  M::Value gendef(Se::SymbolRef sym) { return gen(sym); }

  M::Value genval(Se::SymbolRef sym) {
    auto var = gen(sym);
    if (fir::isReferenceLike(var.getType())) {
      return builder.create<fir::LoadOp>(getLoc(), var);
    }
    return var;
  }

  M::Value genval(Ev::BOZLiteralConstant const &) { TODO(); }
  M::Value genval(Ev::ProcedureRef const &) { TODO(); }
  M::Value genval(Ev::ProcedureDesignator const &) { TODO(); }
  M::Value genval(Ev::NullPointer const &) { TODO(); }
  M::Value genval(Ev::StructureConstructor const &) { TODO(); }
  M::Value genval(Ev::ImpliedDoIndex const &) { TODO(); }
  M::Value genval(Ev::DescriptorInquiry const &desc) {
    auto descRef{symMap.lookupSymbol(desc.base().GetLastSymbol())};
    assert(descRef && "no mlir::Value associated to Symbol");
    auto descType{descRef.getType()};
    M::Value res{};
    switch (desc.field()) {
    case Ev::DescriptorInquiry::Field::Len:
      if (auto boxCharType{descType.dyn_cast<fir::BoxCharType>()}) {
        auto refType{fir::ReferenceType::get(boxCharType.getEleTy())};
        auto lenType{mlir::IntegerType::get(64, builder.getContext())};
        res = builder
                  .create<fir::UnboxCharOp>(getLoc(), refType, lenType, descRef)
                  .getResult(1);
      } else if (descType.isa<fir::BoxType>()) {
        TODO();
      } else {
        assert(false && "not a descriptor");
      }
      break;
    default:
      TODO();
    }
    return res;
  }

  template <int KIND>
  M::Value genval(Ev::TypeParamInquiry<KIND> const &) {
    TODO();
  }

  template <int KIND>
  M::Value genval(Ev::ComplexComponent<KIND> const &part) {
    return ComplexOpsBuilder{builder, getLoc()}.extractComplexPart(
        genval(part.left()), part.isImaginaryPart);
  }

  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Negate<Ev::Type<TC, KIND>> const &op) {
    auto input{genval(op.left())};
    if constexpr (TC == IntegerCat) {
      // Currently no Standard/FIR op for integer negation.
      auto zero{genIntegerConstant<KIND>(builder.getContext(), 0)};
      return builder.create<M::SubIOp>(getLoc(), zero, input);
    } else if constexpr (TC == RealCat) {
      return builder.create<fir::NegfOp>(getLoc(), input);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::NegcOp>(op);
    }
  }

  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Add<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::AddIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::AddfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::AddcOp>(op);
    }
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Subtract<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::SubIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::SubfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::SubcOp>(op);
    }
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Multiply<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::MulIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::MulfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::MulcOp>(op);
    }
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Divide<Ev::Type<TC, KIND>> const &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::SignedDivIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<fir::DivfOp>(op);
    } else {
      static_assert(TC == ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::DivcOp>(op);
    }
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Power<Ev::Type<TC, KIND>> const &op) {
    llvm::SmallVector<mlir::Value, 2> operands{genval(op.left()),
                                               genval(op.right())};
    M::Type ty{converter.genType(TC, KIND)};
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::RealToIntPower<Ev::Type<TC, KIND>> const &op) {
    // TODO: runtime as limited integer kind support. Look if the conversions
    // are ok
    llvm::SmallVector<mlir::Value, 2> operands{genval(op.left()),
                                               genval(op.right())};
    M::Type ty{converter.genType(TC, KIND)};
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }

  template <int KIND>
  M::Value genval(Ev::ComplexConstructor<KIND> const &op) {
    return ComplexOpsBuilder{builder, getLoc()}.createComplex(
        KIND, genval(op.left()), genval(op.right()));
  }
  template <int KIND>
  M::Value genval(Ev::Concat<KIND> const &op) {
    TODO();
  }

  /// MIN and MAX operations
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Extremum<Ev::Type<TC, KIND>> const &op) {
    std::string name{op.ordering == Ev::Ordering::Greater ? "max" : "min"};
    M::Type type{converter.genType(TC, KIND)};
    L::SmallVector<M::Value, 2> operands{genval(op.left()), genval(op.right())};
    return intrinsics.genval(getLoc(), builder, name, type, operands);
  }

  template <int KIND>
  M::Value genval(Ev::SetLength<KIND> const &) {
    TODO();
  }

  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Relational<Ev::Type<TC, KIND>> const &op) {
    mlir::Value result{nullptr};
    if constexpr (TC == IntegerCat) {
      result = createCompareOp<M::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == RealCat) {
      result =
          createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
    } else if constexpr (TC == ComplexCat) {
      bool eq{op.opr == Co::RelationalOperator::EQ};
      assert(eq || op.opr == Co::RelationalOperator::NE &&
                       "relation undefined for complex");
      result = ComplexOpsBuilder{builder, getLoc()}.createComplexCompare(
          genval(op.left()), genval(op.right()), eq);
    } else {
      static_assert(TC == CharacterCat);
      TODO();
    }
    return result;
  }

  M::Value genval(Ev::Relational<Ev::SomeType> const &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Co::TypeCategory TC1, int KIND, Co::TypeCategory TC2>
  M::Value genval(Ev::Convert<Ev::Type<TC1, KIND>, TC2> const &convert) {
    auto ty{converter.genType(TC1, KIND)};
    M::Value operand{genval(convert.left())};
    if (TC1 == LogicalCat && genLogicalAsI1) {
      // If an i1 result is needed, it does not make sens to convert between
      // `fir.logical` types to later convert back to the result to i1.
      return operand;
    }
    return builder.create<fir::ConvertOp>(getLoc(), ty, operand);
  }

  template <typename A>
  M::Value genval(Ev::Parentheses<A> const &) {
    TODO();
  }

  template <int KIND>
  M::Value genval(const Ev::Not<KIND> &op) {
    // Request operands to be generated as `i1` and restore after this scope.
    auto restorer{common::ScopedSet(genLogicalAsI1, true)};
    auto *context{builder.getContext()};
    auto logical{genval(op.left())};
    auto one{genLogicalConstantAsI1(context, true)};
    return builder.create<M::XOrOp>(getLoc(), logical, one).getResult();
  }

  template <int KIND>
  M::Value genval(Ev::LogicalOperation<KIND> const &op) {
    // Request operands to be generated as `i1` and restore after this scope.
    auto restorer{common::ScopedSet(genLogicalAsI1, true)};
    mlir::Value result;
    switch (op.logicalOperator) {
    case Ev::LogicalOperator::And:
      result = createBinaryOp<M::AndOp>(op);
      break;
    case Ev::LogicalOperator::Or:
      result = createBinaryOp<M::OrOp>(op);
      break;
    case Ev::LogicalOperator::Eqv:
      result = createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::eq);
      break;
    case Ev::LogicalOperator::Neqv:
      result = createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::ne);
      break;
    case Ev::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is evaluate::Not<KIND>.
      assert(false && ".NOT. is not a binary operator");
      break;
    }
    if (!result) {
      assert(false && "unhandled logical operation");
    }
    return result;
  }

  template <Co::TypeCategory TC, int KIND>
  M::Value genval(Ev::Constant<Ev::Type<TC, KIND>> const &con) {
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
        return genLogicalConstantAsI1(builder.getContext(), opt->IsTrue());
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

  template <Co::TypeCategory TC>
  M::Value genval(Ev::Constant<Ev::SomeKind<TC>> const &con) {
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

  template <typename A>
  M::Value genval(Ev::ArrayConstructor<A> const &) {
    TODO();
  }
  M::Value gen(Ev::ComplexPart const &) { TODO(); }
  M::Value gendef(Ev::ComplexPart const &cp) { return gen(cp); }
  M::Value genval(Ev::ComplexPart const &) { TODO(); }

  M::Value gen(const Ev::Substring &s) {
    // Get base address
    auto baseAddr{std::visit(
        Co::visitors{
            [&](const Ev::DataRef &x) { return gen(x); },
            [&](const Ev::StaticDataObject::Pointer &) -> M::Value { TODO(); },
        },
        s.parent())};
    // Get a SequenceType to compute address with fir::CoordinateOp
    auto charRefType{baseAddr.getType()};
    auto arrayRefType{getSequenceRefType(charRefType)};
    auto arrayView{
        builder.create<fir::ConvertOp>(getLoc(), arrayRefType, baseAddr)};
    // Compute lower bound
    auto indexType{M::IndexType::get(builder.getContext())};
    auto lowerBoundExpr{s.lower()};
    auto lowerBoundValue{builder.create<fir::ConvertOp>(
        getLoc(), indexType, genval(lowerBoundExpr))};
    // FIR CoordinateOp is zero based but Fortran substring are one based.
    auto one{builder.create<M::ConstantOp>(
        getLoc(), indexType, builder.getIntegerAttr(indexType, 1))};
    auto offsetIndex{
        builder.create<M::SubIOp>(getLoc(), lowerBoundValue, one).getResult()};
    // Get address from offset and base address
    return builder.create<fir::CoordinateOp>(getLoc(), charRefType, arrayView,
                                             offsetIndex);
  }

  M::Value gendef(const Ev::Substring &ss) { return gen(ss); }
  M::Value genval(const Ev::Substring &) { TODO(); }
  M::Value genval(const Ev::Triplet &trip) { TODO(); }

  M::Value genval(const Ev::Subscript &subs) {
    return std::visit(Co::visitors{
                          [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                            return genval(x.value());
                          },
                          [&](const Ev::Triplet &x) { return genval(x); },
                      },
                      subs.u);
  }

  M::Value gen(const Ev::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  M::Value gendef(Ev::DataRef const &dref) { return gen(dref); }
  M::Value genval(Ev::DataRef const &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list.
  // Returns the object used as the base coordinate for the component chain.
  static Ev::DataRef const *
  reverseComponents(Ev::Component const &cmpt,
                    std::list<Ev::Component const *> &list) {
    list.push_front(&cmpt);
    return std::visit(
        Co::visitors{
            [&](Ev::Component const &x) { return reverseComponents(x, list); },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  M::Value gen(const Ev::Component &cmpt) {
    std::list<const Ev::Component *> list;
    auto *base{reverseComponents(cmpt, list)};
    L::SmallVector<M::Value, 2> coorArgs;
    auto obj{gen(*base)};
    auto *sym = &cmpt.GetFirstSymbol();
    M::Type ty{converter.genType(*sym)};
    for (auto *field : list) {
      sym = &field->GetLastSymbol();
      auto name{sym->name().ToString()};
      // FIXME: as we're walking the chain of field names, we need to update the
      // subtype as we drill down
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(getLoc(), name, ty));
    }
    assert(sym && "no component(s)?");
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, obj, coorArgs);
  }
  M::Value gendef(const Ev::Component &cmpt) { return gen(cmpt); }
  M::Value genval(const Ev::Component &cmpt) {
    return builder.create<fir::LoadOp>(getLoc(), gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  M::Type genSubType(M::Type arrTy, unsigned dims) {
    auto unwrapTy{arrTy.cast<fir::ReferenceType>().getEleTy()};
    auto seqTy{unwrapTy.cast<fir::SequenceType>()};
    auto shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    for (unsigned i = 0, e = shape.size() - dims; i < e; ++i) {
      newBnds.push_back(shape[i]);
    }
    if (!newBnds.empty()) {
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    }
    return seqTy.getEleTy();
  }

  // Return the coordinate of the array reference
  M::Value gen(Ev::ArrayRef const &aref) {
    M::Value base = aref.base().IsSymbol() ? gen(aref.base().GetFirstSymbol())
                                           : gen(aref.base().GetComponent());
    llvm::SmallVector<M::Value, 8> args;
    for (auto &subsc : aref.subscript()) {
      args.push_back(genval(subsc));
    }
    auto ty{genSubType(base.getType(), args.size())};
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }

  M::Value gendef(const Ev::ArrayRef &aref) { return gen(aref); }
  M::Value genval(const Ev::ArrayRef &aref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(aref));
  }

  // Return a coordinate of the coarray reference. This is necessary as a
  // Component may have a CoarrayRef as its base coordinate.
  M::Value gen(Ev::CoarrayRef const &coref) {
    // FIXME: need to visit the cosubscripts...
    // return gen(coref.base());
    TODO();
  }
  M::Value gendef(const Ev::CoarrayRef &coref) { return gen(coref); }
  M::Value genval(const Ev::CoarrayRef &coref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(coref));
  }

  template <typename A>
  M::Value gen(Ev::Designator<A> const &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  M::Value gendef(Ev::Designator<A> const &des) {
    return gen(des);
  }
  template <typename A>
  M::Value genval(Ev::Designator<A> const &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  // call a function
  template <typename A>
  M::Value gen(const Ev::FunctionRef<A> &funRef) {
    TODO();
  }
  template <typename A>
  M::Value gendef(const Ev::FunctionRef<A> &funRef) {
    return gen(funRef);
  }
  template <typename A>
  M::Value genval(const Ev::FunctionRef<A> &funRef) {
    TODO(); // Derived type functions (user + intrinsics)
  }
  template <Co::TypeCategory TC, int KIND>
  M::Value genval(const Ev::FunctionRef<Ev::Type<TC, KIND>> &funRef) {
    if (const auto &intrinsic{funRef.proc().GetSpecificIntrinsic()}) {
      M::Type ty{converter.genType(TC, KIND)};
      L::SmallVector<M::Value, 2> operands;
      // Lower arguments
      // For now, logical arguments for intrinsic are lowered to `fir.logical`
      // so that TRANSFER can work. For some arguments, it could lead to useless
      // conversions (e.g scalar MASK of MERGE will be converted to `i1`), but
      // the generated code is at least correct. To improve this, the intrinsic
      // lowering facility should control argument lowering.
      auto restorer{common::ScopedSet(genLogicalAsI1, false)};
      for (const auto &arg : funRef.arguments()) {
        if (auto *expr{Ev::UnwrapExpr<Ev::Expr<Ev::SomeType>>(arg)}) {
          operands.push_back(genval(*expr));
        } else {
          operands.push_back(nullptr); // optional
        }
      }
      // Let the intrinsic library lower the intrinsic function call
      L::StringRef name{intrinsic->name};
      return intrinsics.genval(getLoc(), builder, name, ty, operands);
    } else {
      // implicit interface implementation only
      // TODO: explicit interface
      L::SmallVector<M::Type, 2> argTypes;
      L::SmallVector<M::Value, 2> operands;
      // Logical arguments of user functions must be lowered to `fir.logical`
      // and not `i1`.
      auto restorer{common::ScopedSet(genLogicalAsI1, false)};
      for (const auto &arg : funRef.arguments()) {
        assert(arg.has_value() &&
               "optional argument requires explicit interface");
        const auto *expr{arg->UnwrapExpr()};
        assert(expr && "assumed type argument requires explicit interface");
        if (const Se::Symbol * sym{Ev::UnwrapWholeSymbolDataRef(*expr)}) {
          M::Value argRef{symMap.lookupSymbol(*sym)};
          assert(argRef && "could not get symbol reference");
          argTypes.push_back(argRef.getType());
          operands.push_back(argRef);
        } else {
          // TODO create temps for expressions
          TODO();
        }
      }
      M::Type resultType{converter.genType(TC, KIND)};
      M::FunctionType funTy{
          M::FunctionType::get(argTypes, resultType, builder.getContext())};
      M::FuncOp func{getFunction(applyNameMangling(funRef.proc()), funTy)};
      auto call{builder.create<M::CallOp>(getLoc(), func, operands)};
      // For now, Fortran return value are implemented with a single MLIR
      // function return value.
      assert(call.getNumResults() == 1 &&
             "Expected exactly one result in FUNCTION call");
      return call.getResult(0);
    }
  }

  template <typename A>
  M::Value gen(const Ev::Expr<A> &exp) {
    // must be a designator or function-reference (R902)
    return std::visit([&](const auto &e) { return gendef(e); }, exp.u);
  }
  template <typename A>
  M::Value gendef(Ev::Expr<A> const &exp) {
    return gen(exp);
  }
  template <typename A>
  M::Value genval(Ev::Expr<A> const &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  template <int KIND>
  M::Value genval(Ev::Expr<Ev::Type<LogicalCat, KIND>> const &exp) {
    auto result{std::visit([&](const auto &e) { return genval(e); }, exp.u)};
    // Handle the `i1` to `fir.logical` conversions as needed.
    if (result) {
      M::Type type{result.getType()};
      if (type.isa<fir::LogicalType>()) {
        if (genLogicalAsI1) {
          result = builder.create<fir::ConvertOp>(getLoc(), builder.getI1Type(),
                                                  result);
        }
      } else if (type.isa<M::IntegerType>()) {
        if (!genLogicalAsI1) {
          M::Type firLogicalType{converter.genType(LogicalCat, KIND)};
          result =
              builder.create<fir::ConvertOp>(getLoc(), firLogicalType, result);
        }
      } else if (auto seqType{type.dyn_cast_or_null<fir::SequenceType>()}) {
        // TODO: Conversions at array level should probably be avoided.
        // This depends on how array expressions will be lowered.
        assert(false && "logical array loads not yet implemented");
      } else {
        assert(false && "unexpected logical type in expression");
      }
    }
    return result;
  }

  template <typename A>
  M::Value gendef(const A &) {
    assert(false && "expression error");
    return {};
  }

  std::string applyNameMangling(const Ev::ProcedureDesignator &proc) {
    if (const auto *symbol{proc.GetSymbol()})
      return converter.mangleName(*symbol);
    // Do not mangle intrinsic for now
    assert(proc.GetSpecificIntrinsic() &&
           "expected intrinsic procedure in designator");
    return proc.GetName();
  }

public:
  explicit ExprLowering(M::Location loc, AbstractConverter &converter,
                        SomeExpr const &vop, SymMap &map,
                        IntrinsicLibrary const &intr, bool logicalAsI1 = false)
      : location{loc}, converter{converter}, builder{converter.getOpBuilder()},
        expr{vop}, symMap{map}, intrinsics{intr}, genLogicalAsI1{logicalAsI1} {}

  /// Lower the expression `expr` into MLIR standard dialect
  M::Value gen() { return gen(expr); }
  M::Value genval() { return genval(expr); }
};

} // namespace

M::Value Br::createSomeExpression(M::Location loc,
                                  Br::AbstractConverter &converter,
                                  const Ev::Expr<Ev::SomeType> &expr,
                                  SymMap &symMap,
                                  const IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics, false}.genval();
}

M::Value Br::createI1LogicalExpression(M::Location loc,
                                       Br::AbstractConverter &converter,
                                       const Ev::Expr<Ev::SomeType> &expr,
                                       SymMap &symMap,
                                       const IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics, true}.genval();
}

M::Value Br::createSomeAddress(M::Location loc,
                               Br::AbstractConverter &converter,
                               const Ev::Expr<Ev::SomeType> &expr,
                               SymMap &symMap,
                               const IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics}.gen();
}

/// Create a temporary variable
/// `symbol` will be nullptr for an anonymous temporary
M::Value Br::createTemporary(M::Location loc, M::OpBuilder &builder,
                             SymMap &symMap, M::Type type,
                             const Se::Symbol *symbol) {
  if (symbol)
    if (auto val = symMap.lookupSymbol(*symbol)) {
      if (auto op = val.getDefiningOp())
        return op->getResult(0);
      return val;
    }
  auto insPt(builder.saveInsertionPoint());
  builder.setInsertionPointToStart(getEntryBlock(&builder));
  fir::AllocaOp ae;
  assert(!type.dyn_cast<fir::ReferenceType>() && "cannot be a reference");
  if (symbol) {
    ae = builder.create<fir::AllocaOp>(loc, type, symbol->name().ToString());
    symMap.addSymbol(*symbol, ae);
  } else {
    ae = builder.create<fir::AllocaOp>(loc, type);
  }
  builder.restoreInsertionPoint(insPt);
  return ae;
}
