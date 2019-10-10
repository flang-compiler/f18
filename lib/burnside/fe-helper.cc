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

#include "fe-helper.h"
#include "bridge.h"
#include "fir/Type.h"
#include "../semantics/expression.h"
#include "../semantics/tools.h"
#include "../semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

template<typename A> bool isConstant(const Ev::Expr<A> &e) {
  return Ev::IsConstantExpr(SomeExpr{e});
}

template<typename A> int64_t toConstant(const Ev::Expr<A> &e) {
  auto opt = Ev::ToInt64(e);
  assert(opt.has_value() && "expression didn't resolve to a constant");
  return opt.value();
}

#undef TODO
#define TODO() assert(false); return {}

// one argument template, must be specialized
template<Co::TypeCategory TC>
M::Type genFIRType(M::MLIRContext *context, int kind) {
  return {};
}

// two argument template
template<Co::TypeCategory TC, int KIND>
M::Type genFIRType(M::MLIRContext *context) {
  if constexpr (TC == IntegerCat) {
    auto bits{Ev::Type<IntegerCat, KIND>::Scalar::bits};
    return M::IntegerType::get(bits, context);
  } else if constexpr (TC == LogicalCat || TC == CharacterCat ||
      TC == ComplexCat) {
    return genFIRType<TC>(context, KIND);
  } else {
    return {};
  }
}

template<> M::Type genFIRType<RealCat, 2>(M::MLIRContext *context) {
  return M::FloatType::getF16(context);
}

template<> M::Type genFIRType<RealCat, 3>(M::MLIRContext *context) {
  return M::FloatType::getBF16(context);
}

template<> M::Type genFIRType<RealCat, 4>(M::MLIRContext *context) {
  return M::FloatType::getF32(context);
}

template<> M::Type genFIRType<RealCat, 8>(M::MLIRContext *context) {
  return M::FloatType::getF64(context);
}

template<> M::Type genFIRType<RealCat, 10>(M::MLIRContext *context) {
  return fir::RealType::get(context, 10);
}

template<> M::Type genFIRType<RealCat, 16>(M::MLIRContext *context) {
  return fir::RealType::get(context, 16);
}

template<> M::Type genFIRType<RealCat>(M::MLIRContext *context, int kind) {
  if (Ev::IsValidKindOfIntrinsicType(RealCat, kind)) {
    switch (kind) {
    case 2: return genFIRType<RealCat, 2>(context);
    case 3: return genFIRType<RealCat, 3>(context);
    case 4: return genFIRType<RealCat, 4>(context);
    case 8: return genFIRType<RealCat, 8>(context);
    case 10: return genFIRType<RealCat, 10>(context);
    case 16: return genFIRType<RealCat, 16>(context);
    }
    assert(false && "type translation not implemented");
  }
  return {};
}

template<> M::Type genFIRType<IntegerCat>(M::MLIRContext *context, int kind) {
  if (Ev::IsValidKindOfIntrinsicType(IntegerCat, kind)) {
    switch (kind) {
    case 1: return genFIRType<IntegerCat, 1>(context);
    case 2: return genFIRType<IntegerCat, 2>(context);
    case 4: return genFIRType<IntegerCat, 4>(context);
    case 8: return genFIRType<IntegerCat, 8>(context);
    case 16: return genFIRType<IntegerCat, 16>(context);
    }
    assert(false && "type translation not implemented");
  }
  return {};
}

template<> M::Type genFIRType<LogicalCat>(M::MLIRContext *context, int KIND) {
  if (Ev::IsValidKindOfIntrinsicType(LogicalCat, KIND))
    return fir::LogicalType::get(context, KIND);
  return {};
}

template<> M::Type genFIRType<CharacterCat>(M::MLIRContext *context, int KIND) {
  if (Ev::IsValidKindOfIntrinsicType(CharacterCat, KIND))
    return fir::CharacterType::get(context, KIND);
  return {};
}

template<> M::Type genFIRType<ComplexCat>(M::MLIRContext *context, int KIND) {
  if (Ev::IsValidKindOfIntrinsicType(ComplexCat, KIND))
    return fir::CplxType::get(context, KIND);
  return {};
}

/// Recover the type of an evaluate::Expr<T> and convert it to an
/// mlir::Type. The type returned can be a MLIR standard or FIR type.
class TypeBuilder {
  M::MLIRContext *context;
  Co::IntrinsicTypeDefaultKinds const &defaults;

  template<Co::TypeCategory TC> int defaultKind() { return defaultKind(TC); }
  int defaultKind(Co::TypeCategory TC) { return defaults.GetDefaultKind(TC); }

public:
  explicit TypeBuilder(
      M::MLIRContext *context, Co::IntrinsicTypeDefaultKinds const &defaults)
    : context{context}, defaults{defaults} {}

  // non-template, arguments are runtime values
  M::Type genFIRTy(Co::TypeCategory tc, int kind) {
    switch (tc) {
    case RealCat: return genFIRType<RealCat>(context, kind);
    case IntegerCat: return genFIRType<IntegerCat>(context, kind);
    case ComplexCat: return genFIRType<ComplexCat>(context, kind);
    case LogicalCat: return genFIRType<LogicalCat>(context, kind);
    case CharacterCat: return genFIRType<CharacterCat>(context, kind);
    default: break;
    }
    assert(false && "unhandled type category");
    return {};
  }

  // non-template, category is runtime values, kind is defaulted
  M::Type genFIRTy(Co::TypeCategory tc) {
    return genFIRTy(tc, defaultKind(tc));
  }

  M::Type gen(const Ev::ImpliedDoIndex &) {
    return genFIRType<IntegerCat>(context, defaultKind<IntegerCat>());
  }

  template<template<typename> typename A, Co::TypeCategory TC>
  M::Type gen(const A<Ev::SomeKind<TC>> &) {
    return genFIRType<TC>(context, defaultKind<TC>());
  }

  template<int KIND> M::Type gen(const Ev::TypeParamInquiry<KIND> &) {
    return genFIRType<IntegerCat, KIND>(context);
  }

  template<typename A> M::Type gen(const Ev::Relational<A> &) {
    return genFIRType<LogicalCat, 1>(context);
  }

  template<template<typename> typename A, Co::TypeCategory TC, int KIND>
  M::Type gen(const A<Ev::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  // breaks the conflict between A<Type<TC,KIND>> and Expr<B> deduction
  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::Expr<Ev::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  template<typename A> M::Type genVariant(const A &variant) {
    return std::visit([&](const auto &x) { return gen(x); }, variant.u);
  }

  // breaks the conflict between A<SomeKind<TC>> and Expr<B> deduction
  template<Co::TypeCategory TC>
  M::Type gen(const Ev::Expr<Ev::SomeKind<TC>> &expr) {
    return genVariant(expr);
  }

  template<typename A> M::Type gen(const Ev::Expr<A> &expr) {
    return genVariant(expr);
  }

  M::Type gen(const Ev::DataRef &dref) { return genVariant(dref); }

  M::Type mkVoid() { return M::TupleType::get(context); }

  fir::SequenceType::Shape genSeqShape(const Se::Symbol *symbol) {
    assert(symbol->IsObjectArray());
    fir::SequenceType::Bounds bounds;
    auto &details = symbol->get<Se::ObjectEntityDetails>();
    const auto size = details.shape().size();
    for (auto &ss : details.shape()) {
      auto lb = ss.lbound();
      auto ub = ss.ubound();
      if (lb.isAssumed() && ub.isAssumed() && size == 1) {
        return {};
      }
      if (lb.isExplicit() && ub.isExplicit()) {
        auto &lbv = lb.GetExplicit();
        auto &ubv = ub.GetExplicit();
        if (lbv.has_value() && ubv.has_value() && isConstant(lbv.value()) &&
            isConstant(ubv.value())) {
          bounds.emplace_back(
              toConstant(ubv.value()) - toConstant(lbv.value()) + 1);
        } else {
          bounds.emplace_back(0);
        }
      } else {
        bounds.emplace_back(0);
      }
    }
    return {bounds};
  }

  /// Type consing from a symbol. A symbol's type must be created from the type
  /// discovered by the front-end at runtime.
  M::Type gen(const Se::Symbol *symbol) {
    if (auto *proc = symbol->detailsIf<Se::SubprogramDetails>()) {
      M::Type returnTy{mkVoid()};
      if (proc->isFunction()) {
        returnTy = gen(&proc->result());
      }
      // FIXME: handle alt-return
      llvm::SmallVector<M::Type, 4> inputTys;
      for (auto *arg : proc->dummyArgs()) {
        // FIXME: not all args are pass by ref
        inputTys.emplace_back(fir::ReferenceType::get(gen(arg)));
      }
      return M::FunctionType::get(inputTys, returnTy, context);
    }
    M::Type returnTy{};
    if (auto *type{symbol->GetType()}) {
      if (auto *tySpec{type->AsIntrinsic()}) {
        int kind = toConstant(tySpec->kind());
        switch (tySpec->category()) {
        case IntegerCat:
          returnTy = genFIRType<IntegerCat>(context, kind);
          break;
        case RealCat: returnTy = genFIRType<RealCat>(context, kind); break;
        case ComplexCat:
          returnTy = genFIRType<ComplexCat>(context, kind);
          break;
        case CharacterCat:
          returnTy = genFIRType<CharacterCat>(context, kind);
          break;
        case LogicalCat:
          returnTy = genFIRType<LogicalCat>(context, kind);
          break;
        case DerivedCat: TODO(); break;
        }
      } else if (auto *tySpec{type->AsDerived()}) {
        TODO();
      } else {
        assert(false && "type spec not found");
      }
    } else {
      assert(false && "symbol has no type");
    }
    if (symbol->IsObjectArray()) {
      // FIXME: add bounds info
      returnTy = fir::SequenceType::get(genSeqShape(symbol), returnTy);
    } else if (Se::IsPointer(*symbol)) {
      // FIXME: what about allocatable?
      returnTy = fir::ReferenceType::get(returnTy);
    }
    return returnTy;
  }

  fir::SequenceType::Shape trivialShape(int size) {
    fir::SequenceType::Bounds bounds;
    bounds.emplace_back(size);
    return {bounds};
  }

  // some sequence of `n` bytes
  M::Type gen(const Ev::StaticDataObject::Pointer &ptr) {
    M::Type byteTy{M::IntegerType::get(8, context)};
    return fir::SequenceType::get(trivialShape(ptr->itemBytes()), byteTy);
  }

  M::Type gen(const Ev::Substring &ss) {
    return genVariant(ss.GetBaseObject());
  }

  M::Type genTypelessPtr() { return fir::ReferenceType::get(mkVoid()); }
  M::Type gen(const Ev::NullPointer &) { return genTypelessPtr(); }
  M::Type gen(const Ev::ProcedureRef &) { return genTypelessPtr(); }
  M::Type gen(const Ev::ProcedureDesignator &) { return genTypelessPtr(); }
  M::Type gen(const Ev::BOZLiteralConstant &) { return genTypelessPtr(); }

  M::Type gen(const Ev::ArrayRef &) { TODO(); }
  M::Type gen(const Ev::CoarrayRef &) { TODO(); }
  M::Type gen(const Ev::Component &) { TODO(); }
  M::Type gen(const Ev::ComplexPart &) { TODO(); }
  M::Type gen(const Ev::DescriptorInquiry &) { TODO(); }
  M::Type gen(const Ev::StructureConstructor &) { TODO(); }
};

}  // namespace

/// Generate an unknown location
M::Location Br::dummyLoc(M::MLIRContext *ctxt) {
  return M::UnknownLoc::get(ctxt);
}

// What do we need to convert a CharBlock to actual source locations?
// FIXME: replace with a map from a provenance to a source location
M::Location Br::parserPosToLoc(M::MLIRContext &context,
    const Pa::CookedSource *cooked, const Pa::CharBlock &position) {
  if (cooked) {
    auto loc{cooked->GetSourcePositionRange(position)};
    if (loc.has_value()) {
      auto &filePos{loc->first};
      return M::FileLineColLoc::get(
          filePos.file.path(), filePos.line, filePos.column, &context);
    }
  }
  return M::UnknownLoc::get(&context);
}

M::Type Br::getFIRType(M::MLIRContext *context,
    Co::IntrinsicTypeDefaultKinds const &defaults, Co::TypeCategory tc,
    int kind) {
  return TypeBuilder{context, defaults}.genFIRTy(tc, kind);
}

M::Type Br::getFIRType(M::MLIRContext *context,
    Co::IntrinsicTypeDefaultKinds const &defaults, Co::TypeCategory tc) {
  return TypeBuilder{context, defaults}.genFIRTy(tc);
}

M::Type Br::translateDataRefToFIRType(M::MLIRContext *context,
    Co::IntrinsicTypeDefaultKinds const &defaults, const Ev::DataRef &dataRef) {
  return TypeBuilder{context, defaults}.gen(dataRef);
}

// Builds the FIR type from an instance of SomeExpr
M::Type Br::translateSomeExprToFIRType(M::MLIRContext *context,
    Co::IntrinsicTypeDefaultKinds const &defaults, const SomeExpr *expr) {
  return TypeBuilder{context, defaults}.gen(*expr);
}

// This entry point avoids gratuitously wrapping the Symbol instance in layers
// of Expr<T> that will then be immediately peeled back off and discarded.
M::Type Br::translateSymbolToFIRType(M::MLIRContext *context,
    Co::IntrinsicTypeDefaultKinds const &defaults, const Se::Symbol *symbol) {
  return TypeBuilder{context, defaults}.gen(symbol);
}

M::Type Br::convertReal(M::MLIRContext *context, int kind) {
  return genFIRType<RealCat>(context, kind);
}
