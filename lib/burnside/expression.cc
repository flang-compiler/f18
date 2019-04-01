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

#include "expression.h"
#include "builder.h"
#include "fe-helper.h"
#include "../semantics/expression.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"
#include "../semantics/type.h"
#include "fir/Dialect.h"
#include "fir/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/Ops.h"

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

#undef TODO
#define TODO() assert(false)

/// Collect the arguments and build the dictionary for FIR instructions with
/// embedded evaluate::Expr<T> attributes. These arguments strictly conform to
/// data flow between Fortran expressions.
class TreeArgsBuilder {
  M::OpBuilder *builder;
  Args results;
  Dict dictionary;
  ExprType visited{ET_NONE};
  std::set<void *> keys;
  SymMap &symbolMap;

  void addResult(M::Value *v, void *expr) {
    const auto index{results.size()};
    results.push_back(v);
    dictionary[index] = expr;
    keys.insert(expr);
  }

  void addResult(M::Value *v, const void *expr) {
    addResult(v, const_cast<void *>(expr));
  }

  // FIXME: how do we map an evaluate::Expr<T> to a source location?
  M::Location dummyLoc() { return M::UnknownLoc::get(builder->getContext()); }

public:
  explicit TreeArgsBuilder(M::OpBuilder *builder, SymMap &map)
    : builder{builder}, symbolMap{map} {}

  // FIXME: what we generate for a `Symbol` depends on the category of the
  // symbol
  void gen(const Se::Symbol *variable, M::Type ty, bool asAddr) {
    if (keys.find(const_cast<Se::Symbol *>(variable)) == keys.end()) {
      auto *addr = symbolMap.lookupSymbol(variable);
      if (!addr) {
        auto ip{builder->getInsertionBlock()};
        builder->setInsertionPointToStart(getEntryBlock(builder));
        addr = builder->create<fir::AllocaOp>(dummyLoc(), ty);
        symbolMap.addSymbol(variable, addr);
        builder->setInsertionPointToEnd(ip);
      }
      if (asAddr) {
        addResult(addr, variable);
      } else {
        llvm::SmallVector<M::Value *, 2> loadArg{addr};
        auto load{builder->create<fir::LoadOp>(dummyLoc(), loadArg, ty)};
        addResult(load.getResult(), variable);
      }
    }
    visited = ET_Symbol;
  }

  template<typename A> void gen(const Ev::Designator<A> &des, bool asAddr) {
    std::visit(Co::visitors{
                   [&](const Se::Symbol *x) {
                     auto *ctx{builder->getContext()};
                     M::Type ty{translateDesignatorToFIRType(ctx, des)};
                     gen(x, ty, asAddr);
                   },
                   [&](const auto &x) { gen(x, asAddr); },
               },
        des.u);
  }

  template<typename D, typename R, typename... A>
  void gen(const Ev::Operation<D, R, A...> &op, bool asAddr) {
    gen(op.left(), asAddr);
    if constexpr (op.operands > 1) {
      gen(op.right(), asAddr);
    }
    visited = ET_Operation;
  }

  void gen(const Ev::DataRef &dref, bool asAddr) {
    std::visit(Co::visitors{
                   [&](const Se::Symbol *x) {
                     auto *ctx{builder->getContext()};
                     M::Type ty{translateDataRefToFIRType(ctx, dref)};
                     gen(x, ty, asAddr);
                   },
                   [&](const auto &x) { gen(x, asAddr); },
               },
        dref.u);
  }

  // Note that `NamedEntity` hides its variant member
  void gen(const Ev::NamedEntity &ne, bool asAddr) {
    if (ne.IsSymbol()) {
      // GetFirstSymbol recovers the Symbol* variant
      const Se::Symbol *x = &ne.GetFirstSymbol();
      auto *ctx{builder->getContext()};
      M::Type ty{translateSymbolToFIRType(ctx, x)};
      gen(x, ty, asAddr);
    } else {
      gen(ne.GetComponent(), asAddr);
    }
  }

  // Common pattern for all optional<A>
  template<typename A> void gen(const std::optional<A> &opt, bool asAddr) {
    if (opt.has_value()) {
      gen(opt.value(), asAddr);
    }
  }

  void gen(const Ev::Triplet &triplet, bool asAddr) {
    gen(triplet.lower(), asAddr);
    gen(triplet.upper(), asAddr);
    gen(triplet.stride(), asAddr);
  }

  /// Lower an array reference (implies address arithmetic)
  ///
  /// Visit each of the expressions in each dimension of the array access so as
  /// to add them to the data flow.
  void gen(const Ev::ArrayRef &aref, bool asAddr) {
    // add the base
    gen(aref.base(), true);
    // add each expression
    for (int i = 0, e = aref.size(); i < e; ++i)
      std::visit(Co::visitors{
                     [&](const Ev::Triplet &t) { gen(t, false); },
                     [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                       gen(x.value(), false);
                     },
                 },
          aref.at(i).u);
    visited = ET_ArrayRef;
  }

  void gen(const Ev::NullPointer &, bool) { visited = ET_NullPointer; }

  // implies address arithmetic (and inter-image communication)
  void gen(const Ev::CoarrayRef &coref, bool asAddr) {
    for (auto *sym : coref.base()) {
      gen(sym, translateSymbolToFIRType(builder->getContext(), sym), true);
    }
    for (auto &subs : coref.subscript()) {
      std::visit(Co::visitors{
                     [&](const Ev::Triplet &x) { gen(x, false); },
                     [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                       gen(x.value(), false);
                     },
                 },
          subs.u);
    }
    for (auto &cosubs : coref.cosubscript()) {
      gen(cosubs, false);
    }
    visited = ET_CoarrayRef;
  }

  // implies address arithmetic
  void gen(const Ev::Component &cmpt, bool) {
    gen(cmpt.base(), true);
    visited = ET_Component;
  }

  void gen(const Ev::Substring &subs, bool asAddr) {
    gen(subs.lower(), asAddr);
    gen(subs.upper(), asAddr);
    visited = ET_Substring;
  }

  void gen(const Ev::ComplexPart &, bool) { visited = ET_ComplexPart; }
  void gen(const Ev::ImpliedDoIndex &, bool) {
    // do nothing
  }

  void gen(const Ev::StructureConstructor &, bool) {
    visited = ET_StructureCtor;
  }

  // Skip over constants
  void gen(const Ev::BOZLiteralConstant &, bool) { visited = ET_Constant; }
  template<typename A> void gen(const Ev::Constant<A> &, bool) {
    visited = ET_Constant;
  }

  void gen(const Ev::DescriptorInquiry &, bool) {
    TODO();
    visited = ET_DescriptorInquiry;
  }

  template<int KIND>
  void gen(const Ev::TypeParamInquiry<KIND> &inquiry, bool asAddr) {
    gen(inquiry.base(), asAddr);
    visited = ET_TypeParamInquiry;
  }

  void gen(const Ev::ProcedureRef &pref, bool asAddr) {
    gen(pref.proc(), true);
    for (auto &arg : pref.arguments()) {
      if (arg.has_value()) {
        auto &xarg{arg.value()};
        if (auto *x{xarg.UnwrapExpr()}) {
          gen(*x, asAddr);
        } else {
          auto *sym{xarg.GetAssumedTypeDummy()};
          auto *ctx{builder->getContext()};
          gen(sym, translateSymbolToFIRType(ctx, sym), asAddr);
        }
      }
    }
  }

  void gen(const Ev::ProcedureDesignator &des, bool asAddr) {
    std::visit(Co::visitors{
                   [&](const Ev::SpecificIntrinsic &) {
                     // do nothing
                   },
                   [&](const Se::Symbol *sym) {
                     auto *ctx{builder->getContext()};
                     gen(sym, translateSymbolToFIRType(ctx, sym), true);
                   },
                   [&](const Co::CopyableIndirection<Ev::Component> &x) {
                     gen(x.value(), true);
                   },
               },
        des.u);
  }

  M::FunctionType genFunctionType(const Ev::ProcedureDesignator &proc) {
    auto *ctx{builder->getContext()};
    M::Type ty{
        std::visit(Co::visitors{
                       [&](const Ev::SpecificIntrinsic &) {
                         TODO(); /* FIXME */
                         return M::Type{};
                       },
                       [&](const Se::Symbol *x) {
                         return translateSymbolToFIRType(ctx, x);
                       },
                       [&](const Co::CopyableIndirection<Ev::Component> &x) {
                         auto *sym{&x.value().GetLastSymbol()};
                         return translateSymbolToFIRType(ctx, sym);
                       },
                   },
            proc.u)};
    return ty.cast<M::FunctionType>();
  }

  /// Lower a function call
  ///
  /// A call is lowered to a sequence that evaluates the arguments and
  /// passes them to a CallOp to the named function.
  ///
  /// TODO: The function needs to be name-mangled by the front-end or some
  /// utility to avoid collisions.
  ///
  ///   %54 = FIR.ApplyExpr(...)
  ///   %55 = FIR.ApplyExpr(...)
  ///   %56 = call(@target_func, %54, %55)
  template<typename A> void gen(const Ev::FunctionRef<A> &funref, bool) {
    // must be lowered to a call and data flow added for each actual arg
    auto *context = builder->getContext();
    M::Location loc = M::UnknownLoc::get(context);  // FIXME

    // lookup this function
    llvm::StringRef callee = funref.proc().GetName();
    auto module{getModule(builder)};
    auto func = getNamedFunction(callee);
    if (!func) {
      // create new function
      auto funTy{genFunctionType(funref.proc())};
      func = createFunction(module, callee, funTy);
    }
    assert(func);

    // build arguments
    llvm::SmallVector<M::Value *, 4> inputs;
    for (auto &arg : funref.arguments()) {
      if (arg.has_value()) {
        if (auto *aa{arg->UnwrapExpr()}) {
          auto eops{translateSomeExpr(builder, aa, symbolMap)};
          auto aaType{
              FIRReferenceType::get(translateSomeExprToFIRType(context, aa))};
          M::Value *toLoc{nullptr};
          auto *defOp{getArgs(eops)[0]->getDefiningOp()};
          if (auto load = M::dyn_cast<fir::LoadOp>(defOp)) {
            toLoc = load.getOperand();
          } else {
            auto locate{builder->create<LocateExpr>(
                loc, aa, getDict(eops), getArgs(eops), aaType)};
            toLoc = locate.getResult();
          }
          inputs.push_back(toLoc);
        } else {
          auto *x{arg->GetAssumedTypeDummy()};
          // FIXME: can argument not be passed by reference?
          gen(x, translateSymbolToFIRType(context, x), true);
        }
      } else {
        assert(false && "argument is std::nullopt?");
      }
    }

    // generate a call
    auto call = builder->create<M::CallOp>(loc, func, inputs);
    addResult(call.getResult(0), &funref);
    visited = ET_FunctionRef;
  }

  template<typename A> void gen(const Ev::ImpliedDo<A> &ido, bool asAddr) {
    gen(ido.lower(), asAddr);
    gen(ido.upper(), asAddr);
    gen(ido.stride(), asAddr);
  }

  template<typename A>
  void gen(const Ev::ArrayConstructor<A> &arrayCtor, bool asAddr) {
    for (auto i{arrayCtor.begin()}, end{arrayCtor.end()}; i != end; ++i) {
      std::visit([&](auto &x) { gen(x, asAddr); }, i->u);
    }
    visited = ET_ArrayCtor;
  }

  template<typename A> void gen(const Ev::Relational<A> &op, bool asAddr) {
    gen(op.left(), asAddr);
    gen(op.right(), asAddr);
    visited = ET_Relational;
  }

  void gen(const Ev::Relational<Ev::SomeType> &op, bool asAddr) {
    std::visit([&](const auto &x) { gen(x, asAddr); }, op.u);
  }

  template<typename A> void gen(const Ev::Expr<A> &expr, bool asAddr) {
    std::visit([&](const auto &x) { gen(x, asAddr); }, expr.u);
  }

  Args getResults() const { return results; }
  Dict getDictionary() const { return dictionary; }
  ExprType getExprType() const { return visited; }
};

// Builds the `([results], [inputs], {dict})` pair for constructing both
// `fir.apply_expr` and `fir.locate_expr` operations.
template<bool AddressResult>
inline Values translateToFIR(
    M::OpBuilder *bldr, const SomeExpr *exp, SymMap &map) {
  TreeArgsBuilder ab{bldr, map};
  ab.gen(*exp, AddressResult);
  return {ab.getResults(), ab.getDictionary(), ab.getExprType()};
}

}  // namespace

// `fir.apply_expr` builder
Values Br::translateSomeExpr(
    M::OpBuilder *bldr, const SomeExpr *exp, SymMap &map) {
  return translateToFIR<false>(bldr, exp, map);
}

// `fir.locate_expr` builder
Values Br::translateSomeAddrExpr(
    M::OpBuilder *bldr, const SomeExpr *exp, SymMap &map) {
  return translateToFIR<true>(bldr, exp, map);
}
