// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#include "common.h"
#include "int-power.h"
#include "tools.h"
#include "variable.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include <ostream>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// AsFortran() formatting

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::AsFortran(std::ostream &o) const {
  left().AsFortran(derived().Prefix(o));
  if constexpr (operands > 1) {
    right().AsFortran(derived().Infix(o));
  }
  return derived().Suffix(o);
}

template<typename TO, TypeCategory FROMCAT>
std::ostream &Convert<TO, FROMCAT>::AsFortran(std::ostream &o) const {
  static_assert(TO::category == TypeCategory::Integer ||
      TO::category == TypeCategory::Real ||
      TO::category == TypeCategory::Logical || !"Convert<> to bad category!");
  if constexpr (TO::category == TypeCategory::Integer) {
    o << "int";
  } else if constexpr (TO::category == TypeCategory::Real) {
    o << "real";
  } else if constexpr (TO::category == TypeCategory::Logical) {
    o << "logical";
  }
  return this->left().AsFortran(o << '(') << ",kind=" << TO::kind << ')';
}

template<typename A> std::ostream &Relational<A>::Infix(std::ostream &o) const {
  switch (opr) {
  case RelationalOperator::LT: o << '<'; break;
  case RelationalOperator::LE: o << "<="; break;
  case RelationalOperator::EQ: o << "=="; break;
  case RelationalOperator::NE: o << "/="; break;
  case RelationalOperator::GE: o << ">="; break;
  case RelationalOperator::GT: o << '>'; break;
  }
  return o;
}

std::ostream &Relational<SomeType>::AsFortran(std::ostream &o) const {
  std::visit([&](const auto &rel) { rel.AsFortran(o); }, u);
  return o;
}

template<int KIND>
std::ostream &LogicalOperation<KIND>::Infix(std::ostream &o) const {
  switch (logicalOperator) {
  case LogicalOperator::And: o << ".and."; break;
  case LogicalOperator::Or: o << ".or."; break;
  case LogicalOperator::Eqv: o << ".eqv."; break;
  case LogicalOperator::Neqv: o << ".neqv."; break;
  }
  return o;
}

template<typename T>
std::ostream &Constant<T>::AsFortran(std::ostream &o) const {
  if constexpr (T::category == TypeCategory::Integer) {
    return o << value.SignedDecimal() << '_' << T::kind;
  } else if constexpr (T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex) {
    return value.AsFortran(o, T::kind);
  } else if constexpr (T::category == TypeCategory::Character) {
    return o << T::kind << '_' << parser::QuoteCharacterLiteral(value);
  } else if constexpr (T::category == TypeCategory::Logical) {
    if (value.IsTrue()) {
      o << ".true.";
    } else {
      o << ".false.";
    }
    return o << '_' << Result::kind;
  } else {
    return value.u.AsFortran(o);
  }
}

template<typename T>
std::ostream &Emit(std::ostream &o, const CopyableIndirection<Expr<T>> &expr) {
  return expr->AsFortran(o);
}
template<typename T>
std::ostream &Emit(std::ostream &, const ArrayConstructorValues<T> &);

template<typename ITEM, typename INT>
std::ostream &Emit(std::ostream &o, const ImpliedDo<ITEM, INT> &implDo) {
  o << '(';
  Emit(o, *implDo.values);
  o << ',' << INT::AsFortran() << "::";
  o << implDo.controlVariableName.ToString();
  o << '=';
  implDo.lower->AsFortran(o) << ',';
  implDo.upper->AsFortran(o) << ',';
  implDo.stride->AsFortran(o) << ')';
  return o;
}

template<typename T>
std::ostream &Emit(std::ostream &o, const ArrayConstructorValues<T> &values) {
  const char *sep{""};
  for (const auto &value : values.values) {
    o << sep;
    std::visit([&](const auto &x) { Emit(o, x); }, value.u);
    sep = ",";
  }
  return o;
}

template<typename T>
std::ostream &ArrayConstructor<T>::AsFortran(std::ostream &o) const {
  o << '[' << result.AsFortran() << "::";
  Emit(o, *this);
  return o << ']';
}

template<typename RESULT>
std::ostream &ExpressionBase<RESULT>::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const BOZLiteralConstant &x) {
            o << "z'" << x.Hexadecimal() << "'";
          },
          [&](const CopyableIndirection<Substring> &s) { s->AsFortran(o); },
          [&](const auto &x) { x.AsFortran(o); },
      },
      derived().u);
  return o;
}

template<typename T> Expr<SubscriptInteger> ArrayConstructor<T>::LEN() const {
  // TODO pmk: extract from type spec in array constructor
  return AsExpr(Constant<SubscriptInteger>{0});  // TODO placeholder
}

template<int KIND>
Expr<SubscriptInteger> Expr<Type<TypeCategory::Character, KIND>>::LEN() const {
  return std::visit(
      common::visitors{
          [](const Constant<Result> &c) {
            return AsExpr(Constant<SubscriptInteger>{c.value.size()});
          },
          [](const ArrayConstructor<Result> &a) { return a.LEN(); },
          [](const Parentheses<Result> &x) { return x.left().LEN(); },
          [](const Concat<KIND> &c) {
            return c.left().LEN() + c.right().LEN();
          },
          [](const Extremum<Result> &c) {
            return Expr<SubscriptInteger>{
                Extremum<SubscriptInteger>{c.left().LEN(), c.right().LEN()}};
          },
          [](const Designator<Result> &dr) { return dr.LEN(); },
          [](const FunctionRef<Result> &fr) { return fr.LEN(); },
      },
      u);
}

Expr<SomeType>::~Expr() {}

template<typename T> DynamicType ArrayConstructor<T>::GetType() const {
  // TODO: pmk: parameterized derived types, CHARACTER length
  return result.GetType();
}

#if defined(__APPLE__) && defined(__GNUC__)
template<typename A>
typename ExpressionBase<A>::Derived &ExpressionBase<A>::derived() {
  return *static_cast<Derived *>(this);
}

template<typename A>
const typename ExpressionBase<A>::Derived &ExpressionBase<A>::derived() const {
  return *static_cast<const Derived *>(this);
}
#endif

template<typename A>
std::optional<DynamicType> ExpressionBase<A>::GetType() const {
  if constexpr (IsSpecificIntrinsicType<Result>) {
    return Result::GetType();
  } else {
    return std::visit(
        [](const auto &x) -> std::optional<DynamicType> {
          if constexpr (!std::is_same_v<std::decay_t<decltype(x)>,
                            BOZLiteralConstant>) {
            return x.GetType();
          }
          return std::nullopt;  // typeless -> no type
        },
        derived().u);
  }
}

template<typename A> int ExpressionBase<A>::Rank() const {
  return std::visit(
      [](const auto &x) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                          BOZLiteralConstant>) {
          return 0;
        } else {
          return x.Rank();
        }
      },
      derived().u);
}

// Template instantiations to resolve the "extern template" declarations
// that appear in expression.h.

FOR_EACH_INTRINSIC_KIND(template class Expr)
FOR_EACH_CATEGORY_TYPE(template class Expr)
FOR_EACH_INTEGER_KIND(template struct Relational)
FOR_EACH_REAL_KIND(template struct Relational)
FOR_EACH_CHARACTER_KIND(template struct Relational)
template struct Relational<SomeType>;
FOR_EACH_TYPE_AND_KIND(template class ExpressionBase)
}

// For reclamation of analyzed expressions to which owning pointers have
// been embedded in the parse tree.  This destructor appears here, where
// definitions for all the necessary types are available, to obviate a
// need to include lib/evaluate/*.h headers in the parser proper.
namespace Fortran::common {
template<> OwningPointer<evaluate::GenericExprWrapper>::~OwningPointer() {
  delete p_;
  p_ = nullptr;
}
template class OwningPointer<evaluate::GenericExprWrapper>;
}
