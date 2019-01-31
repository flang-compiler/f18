// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_VARIABLE_H_
#define FORTRAN_EVALUATE_VARIABLE_H_

// Defines data structures to represent data access and function calls
// for use in expressions and assignment statements.  Both copy and move
// semantics are supported.  The representation adheres closely to the
// Fortran 2018 language standard (q.v.) and uses strong typing to ensure
// that only admissable combinations can be constructed.

// TODO pmk: convert remaining structs to classes

#include "call.h"
#include "common.h"
#include "static-data.h"
#include "type.h"
#include "../common/idioms.h"
#include "../common/template.h"
#include "../parser/char-block.h"
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::evaluate {

using semantics::Symbol;

// Forward declarations
struct DataRef;
template<typename A> struct Variable;

// Reference a base object in memory.  This can be a Fortran symbol,
// static data (e.g., CHARACTER literal), or compiler-created temporary.
struct BaseObject {
  CLASS_BOILERPLATE(BaseObject)
  explicit BaseObject(const Symbol &symbol) : u{&symbol} {}
  explicit BaseObject(StaticDataObject::Pointer &&p) : u{std::move(p)} {}
  int Rank() const;
  Expr<SubscriptInteger> LEN() const;
  bool operator==(const BaseObject &) const;
  std::ostream &AsFortran(std::ostream &) const;
  std::variant<const Symbol *, StaticDataObject::Pointer> u;
};

// R913 structure-component & C920: Defined to be a multi-part
// data-ref whose last part has no subscripts (or image-selector, although
// that isn't explicit in the document).  Pointer and allocatable components
// are not explicitly indirected in this representation (TODO: yet?)
// Complex components (%RE, %IM) are isolated below in ComplexPart.
// (Type parameter inquiries look like component references but are distinct
// constructs and not represented by this class.)
class Component {
public:
  CLASS_BOILERPLATE(Component)
  Component(const DataRef &b, const Symbol &c) : base_{b}, symbol_{&c} {}
  Component(DataRef &&b, const Symbol &c) : base_{std::move(b)}, symbol_{&c} {}
  Component(CopyableIndirection<DataRef> &&b, const Symbol &c)
    : base_{std::move(b)}, symbol_{&c} {}

  const DataRef &base() const { return *base_; }
  DataRef &base() { return *base_; }
  int Rank() const;
  const Symbol &GetFirstSymbol() const;
  const Symbol &GetLastSymbol() const { return *symbol_; }
  Expr<SubscriptInteger> LEN() const;
  bool operator==(const Component &) const;
  std::ostream &AsFortran(std::ostream &) const;

private:
  CopyableIndirection<DataRef> base_;
  const Symbol *symbol_;
};

using SymbolOrComponent = std::variant<const Symbol *, Component>;

// R916 type-param-inquiry
// N.B. x%LEN for CHARACTER is rewritten in semantics to LEN(x), which is
// then handled via LEN() member functions in the various classes.
// x%KIND for intrinsic types is similarly rewritten in semantics to
// KIND(x), which is then folded to a constant value.
// "Bare" type parameter references within a derived type definition do
// not have base objects here, only symbols.
template<int KIND> struct TypeParamInquiry {
  using Result = Type<TypeCategory::Integer, KIND>;
  CLASS_BOILERPLATE(TypeParamInquiry)
  TypeParamInquiry(const Symbol &symbol, const Symbol &param)
    : u{&symbol}, parameter{&param} {}
  TypeParamInquiry(Component &&component, const Symbol &param)
    : u{component}, parameter{&param} {}
  TypeParamInquiry(SymbolOrComponent &&x, const Symbol &param)
    : u{x}, parameter{&param} {}
  explicit TypeParamInquiry(const Symbol &param) : parameter{&param} {}
  static constexpr int Rank() { return 0; }  // always scalar
  bool operator==(const TypeParamInquiry &) const;
  std::ostream &AsFortran(std::ostream &) const;

  SymbolOrComponent u{nullptr};
  const Symbol *parameter;
};

EXPAND_FOR_EACH_INTEGER_KIND(
    TEMPLATE_INSTANTIATION, extern template struct TypeParamInquiry)

// R921 subscript-triplet
class Triplet {
public:
  Triplet();
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Triplet)
  Triplet(std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);
  std::optional<Expr<SubscriptInteger>> lower() const;
  std::optional<Expr<SubscriptInteger>> upper() const;
  const Expr<SubscriptInteger> &stride() const;
  bool operator==(const Triplet &) const;
  bool IsStrideOne() const;
  std::ostream &AsFortran(std::ostream &) const;

private:
  std::optional<IndirectSubscriptIntegerExpr> lower_, upper_;
  IndirectSubscriptIntegerExpr stride_;
};

// R919 subscript when rank 0, R923 vector-subscript when rank 1
struct Subscript {
  EVALUATE_UNION_CLASS_BOILERPLATE(Subscript)
  explicit Subscript(Expr<SubscriptInteger> &&s)
    : u{IndirectSubscriptIntegerExpr::Make(std::move(s))} {}
  int Rank() const;
  std::ostream &AsFortran(std::ostream &) const;
  std::variant<IndirectSubscriptIntegerExpr, Triplet> u;
};

// R917 array-element, R918 array-section; however, the case of an
// array-section that is a complex-part-designator is represented here
// as a ComplexPart instead.  C919 & C925 require that at most one set of
// subscripts have rank greater than 0, but that is not explicit in
// these types.
struct ArrayRef {
  CLASS_BOILERPLATE(ArrayRef)
  ArrayRef(const Symbol &symbol, std::vector<Subscript> &&ss)
    : u{&symbol}, subscript(std::move(ss)) {}
  ArrayRef(Component &&c, std::vector<Subscript> &&ss)
    : u{std::move(c)}, subscript(std::move(ss)) {}

  int Rank() const;
  const Symbol &GetFirstSymbol() const;
  const Symbol &GetLastSymbol() const;
  Expr<SubscriptInteger> LEN() const;
  bool operator==(const ArrayRef &) const;
  std::ostream &AsFortran(std::ostream &) const;

  SymbolOrComponent u;
  std::vector<Subscript> subscript;
};

// R914 coindexed-named-object
// R924 image-selector, R926 image-selector-spec.
// C824 severely limits the usage of derived types with coarray ultimate
// components: they can't be pointers, allocatables, arrays, coarrays, or
// function results.  They can be components of other derived types.
// C930 precludes having both TEAM= and TEAM_NUMBER=.
// TODO C931 prohibits the use of a coindexed object as a stat-variable.
class CoarrayRef {
public:
  CLASS_BOILERPLATE(CoarrayRef)
  CoarrayRef(std::vector<const Symbol *> &&,
      std::vector<Expr<SubscriptInteger>> &&,
      std::vector<Expr<SubscriptInteger>> &&);

  const std::vector<const Symbol *> &base() const { return base_; }
  const std::vector<Expr<SubscriptInteger>> &subscript() const {
    return subscript_;
  }
  const std::vector<Expr<SubscriptInteger>> &cosubscript() const {
    return cosubscript_;
  }

  // These integral expressions for STAT= and TEAM= must be variables
  // (i.e., Designator or pointer-valued FunctionRef).
  std::optional<Expr<SomeInteger>> stat() const;
  CoarrayRef &set_stat(Expr<SomeInteger> &&);
  std::optional<Expr<SomeInteger>> team() const;
  bool teamIsTeamNumber() const { return teamIsTeamNumber_; }
  CoarrayRef &set_team(Expr<SomeInteger> &&, bool isTeamNumber = false);

  int Rank() const;
  const Symbol &GetFirstSymbol() const { return *base_.front(); }
  const Symbol &GetLastSymbol() const { return *base_.back(); }
  Expr<SubscriptInteger> LEN() const;
  bool operator==(const CoarrayRef &) const;
  std::ostream &AsFortran(std::ostream &) const;

private:
  std::vector<const Symbol *> base_;
  std::vector<Expr<SubscriptInteger>> subscript_, cosubscript_;
  std::optional<CopyableIndirection<Expr<SomeInteger>>> stat_, team_;
  bool teamIsTeamNumber_{false};  // false: TEAM=, true: TEAM_NUMBER=
};

// R911 data-ref is defined syntactically as a series of part-refs, which
// would be far too expressive if the constraints were ignored.  Here, the
// possible outcomes are spelled out.  Note that a data-ref cannot include
// a terminal substring range or complex component designator; use
// R901 designator for that.
struct DataRef {
  EVALUATE_UNION_CLASS_BOILERPLATE(DataRef)
  explicit DataRef(const Symbol &n) : u{&n} {}

  int Rank() const;
  const Symbol &GetFirstSymbol() const;
  const Symbol &GetLastSymbol() const;
  Expr<SubscriptInteger> LEN() const;
  std::ostream &AsFortran(std::ostream &) const;

  std::variant<const Symbol *, Component, ArrayRef, CoarrayRef> u;
};

// R908 substring, R909 parent-string, R910 substring-range.
// The base object of a substring can be a literal.
// In the F2018 standard, substrings of array sections are parsed as
// variants of sections instead.
class Substring {
public:
  CLASS_BOILERPLATE(Substring)
  Substring(DataRef &&parent, std::optional<Expr<SubscriptInteger>> &&lower,
      std::optional<Expr<SubscriptInteger>> &&upper)
    : parent_{std::move(parent)} {
    SetBounds(lower, upper);
  }
  Substring(StaticDataObject::Pointer &&parent,
      std::optional<Expr<SubscriptInteger>> &&lower,
      std::optional<Expr<SubscriptInteger>> &&upper)
    : parent_{std::move(parent)} {
    SetBounds(lower, upper);
  }

  Expr<SubscriptInteger> lower() const;
  Expr<SubscriptInteger> upper() const;
  int Rank() const;
  template<typename A> const A *GetParentIf() const {
    return std::get_if<A>(&parent_);
  }
  BaseObject GetBaseObject() const;
  const Symbol *GetLastSymbol() const;
  Expr<SubscriptInteger> LEN() const;
  bool operator==(const Substring &) const;
  std::ostream &AsFortran(std::ostream &) const;

  std::optional<Expr<SomeCharacter>> Fold(FoldingContext &);

private:
  void SetBounds(std::optional<Expr<SubscriptInteger>> &,
      std::optional<Expr<SubscriptInteger>> &);
  std::variant<DataRef, StaticDataObject::Pointer> parent_;
  std::optional<IndirectSubscriptIntegerExpr> lower_, upper_;
};

// R915 complex-part-designator
// In the F2018 standard, complex parts of array sections are parsed as
// variants of sections instead.
class ComplexPart {
public:
  ENUM_CLASS(Part, RE, IM)
  CLASS_BOILERPLATE(ComplexPart)
  ComplexPart(DataRef &&z, Part p) : complex_{std::move(z)}, part_{p} {}
  const DataRef &complex() const { return complex_; }
  Part part() const { return part_; }
  int Rank() const;
  const Symbol &GetFirstSymbol() const { return complex_.GetFirstSymbol(); }
  const Symbol &GetLastSymbol() const { return complex_.GetLastSymbol(); }
  bool operator==(const ComplexPart &) const;
  std::ostream &AsFortran(std::ostream &) const;

private:
  DataRef complex_;
  Part part_;
};

// R901 designator is the most general data reference object, apart from
// calls to pointer-valued functions.  Its variant holds everything that
// a DataRef can, and possibly also a substring reference or a
// complex component (%RE/%IM) reference.
template<typename T> class Designator {
  using DataRefs = decltype(DataRef::u);
  using MaybeSubstring =
      std::conditional_t<T::category == TypeCategory::Character,
          std::variant<Substring>, std::variant<>>;
  using MaybeComplexPart = std::conditional_t<T::category == TypeCategory::Real,
      std::variant<ComplexPart>, std::variant<>>;
  using Variant =
      common::CombineVariants<DataRefs, MaybeSubstring, MaybeComplexPart>;

public:
  using Result = T;
  static_assert(IsSpecificIntrinsicType<Result> ||
      std::is_same_v<Result, SomeKind<TypeCategory::Derived>>);
  EVALUATE_UNION_CLASS_BOILERPLATE(Designator)
  Designator(const DataRef &that) : u{common::MoveVariant<Variant>(that.u)} {}
  Designator(DataRef &&that)
    : u{common::MoveVariant<Variant>(std::move(that.u))} {}

  std::optional<DynamicType> GetType() const;
  int Rank() const;
  BaseObject GetBaseObject() const;
  const Symbol *GetLastSymbol() const;
  Expr<SubscriptInteger> LEN() const;
  std::ostream &AsFortran(std::ostream &o) const;

  Variant u;
};

FOR_EACH_CHARACTER_KIND(extern template class Designator)

class ProcedureRef {
public:
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, ActualArguments &&a)
    : proc_{std::move(p)}, arguments_(std::move(a)) {}

  ProcedureDesignator &proc() { return proc_; }
  const ProcedureDesignator &proc() const { return proc_; }
  ActualArguments &arguments() { return arguments_; }
  const ActualArguments &arguments() const { return arguments_; }

  Expr<SubscriptInteger> LEN() const;
  int Rank() const { return proc_.Rank(); }
  bool IsElemental() const { return proc_.IsElemental(); }
  bool operator==(const ProcedureRef &) const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  ProcedureDesignator proc_;
  ActualArguments arguments_;
};

template<typename A> struct FunctionRef : public ProcedureRef {
  using Result = A;
  CLASS_BOILERPLATE(FunctionRef)
  FunctionRef(ProcedureRef &&pr) : ProcedureRef{std::move(pr)} {}
  FunctionRef(ProcedureDesignator &&p, ActualArguments &&a)
    : ProcedureRef{std::move(p), std::move(a)} {}

  std::optional<DynamicType> GetType() const { return proc_.GetType(); }
  std::optional<Constant<Result>> Fold(FoldingContext &);  // for intrinsics
};

FOR_EACH_SPECIFIC_TYPE(extern template struct FunctionRef)

template<typename A> struct Variable {
  using Result = A;
  static_assert(IsSpecificIntrinsicType<Result> ||
      std::is_same_v<Result, SomeKind<TypeCategory::Derived>>);
  EVALUATE_UNION_CLASS_BOILERPLATE(Variable)
  std::optional<DynamicType> GetType() const {
    return std::visit([](const auto &x) { return x.GetType(); }, u);
  }
  int Rank() const {
    return std::visit([](const auto &x) { return x.Rank(); }, u);
  }
  std::ostream &AsFortran(std::ostream &o) const {
    std::visit([&](const auto &x) { x.AsFortran(o); }, u);
    return o;
  }
  std::variant<Designator<Result>, FunctionRef<Result>> u;
};
}
#endif  // FORTRAN_EVALUATE_VARIABLE_H_
