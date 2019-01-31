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

#ifndef FORTRAN_SEMANTICS_SCOPE_H_
#define FORTRAN_SEMANTICS_SCOPE_H_

#include "attr.h"
#include "symbol.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../parser/message.h"
#include <list>
#include <map>
#include <set>
#include <string>

namespace Fortran::evaluate {
struct FoldingContext;
}

namespace Fortran::semantics {

using namespace parser::literals;

class Scope {
  using mapType = std::map<SourceName, Symbol *>;

public:
  ENUM_CLASS(Kind, System, Global, Module, MainProgram, Subprogram, DerivedType,
      Block, Forall, ImpliedDos)
  using ImportKind = common::ImportKind;

  // Create the Global scope -- the root of the scope tree
  Scope() : Scope{*this, Kind::Global, nullptr} {}
  Scope(Scope &parent, Kind kind, Symbol *symbol)
    : parent_{parent}, kind_{kind}, symbol_{symbol} {
    if (symbol) {
      symbol->set_scope(this);
    }
  }

  bool operator==(const Scope &that) const { return this == &that; }
  bool operator!=(const Scope &that) const { return this != &that; }

  Scope &parent() {
    CHECK(kind_ != Kind::System);
    return parent_;
  }
  const Scope &parent() const {
    CHECK(kind_ != Kind::System);
    return parent_;
  }
  Kind kind() const { return kind_; }
  bool IsModule() const;  // only module, not submodule
  bool IsParameterizedDerivedType() const;
  Symbol *symbol() { return symbol_; }
  const Symbol *symbol() const { return symbol_; }

  const Symbol *GetSymbol() const;
  const Scope *GetDerivedTypeParent() const;

  const SourceName &name() const {
    const Symbol *sym{GetSymbol()};
    CHECK(sym != nullptr);
    return sym->name();
  }

  /// Make a scope nested in this one
  Scope &MakeScope(Kind kind, Symbol *symbol = nullptr);

  using size_type = mapType::size_type;
  using iterator = mapType::iterator;
  using const_iterator = mapType::const_iterator;

  iterator begin() { return symbols_.begin(); }
  iterator end() { return symbols_.end(); }
  const_iterator begin() const { return symbols_.begin(); }
  const_iterator end() const { return symbols_.end(); }
  const_iterator cbegin() const { return symbols_.cbegin(); }
  const_iterator cend() const { return symbols_.cend(); }

  iterator find(const SourceName &name);
  const_iterator find(const SourceName &name) const {
    return symbols_.find(name);
  }
  size_type erase(const SourceName &);
  size_type size() const { return symbols_.size(); }
  bool empty() const { return symbols_.empty(); }

  // Look for symbol by name in this scope and host (depending on imports).
  // Be advised: when the scope is a derived type, the search begins in its
  // enclosing scope and will not match any component or parameter of the
  // derived type; use find() instead when seeking those.
  Symbol *FindSymbol(const SourceName &) const;

  /// Make a Symbol with unknown details.
  std::pair<iterator, bool> try_emplace(
      const SourceName &name, Attrs attrs = Attrs()) {
    return try_emplace(name, attrs, UnknownDetails());
  }
  /// Make a Symbol with provided details.
  template<typename D>
  std::pair<iterator, bool> try_emplace(const SourceName &name, D &&details) {
    return try_emplace(name, Attrs(), details);
  }
  /// Make a Symbol with attrs and details
  template<typename D>
  std::pair<iterator, bool> try_emplace(
      const SourceName &name, Attrs attrs, D &&details) {
    Symbol &symbol{MakeSymbol(name, attrs, std::move(details))};
    return symbols_.emplace(name, &symbol);
  }

  /// Make a Symbol but don't add it to the scope.
  template<typename D>
  Symbol &MakeSymbol(const SourceName &name, Attrs attrs, D &&details) {
    return allSymbols.Make(*this, name, attrs, std::move(details));
  }

  std::list<Scope> &children() { return children_; }
  const std::list<Scope> &children() const { return children_; }

  // For Module scope, maintain a mapping of all submodule scopes with this
  // module as its ancestor module. AddSubmodule returns false if already there.
  Scope *FindSubmodule(const SourceName &) const;
  bool AddSubmodule(const SourceName &, Scope &);

  const DeclTypeSpec &MakeNumericType(TypeCategory, KindExpr &&kind);
  const DeclTypeSpec &MakeLogicalType(KindExpr &&kind);
  const DeclTypeSpec &MakeCharacterType(
      ParamValue &&length, KindExpr &&kind = KindExpr{0});
  const DeclTypeSpec &MakeDerivedType(
      DeclTypeSpec::Category, DerivedTypeSpec &&);
  const DeclTypeSpec &MakeDerivedType(
      DeclTypeSpec::Category, DerivedTypeSpec &&, evaluate::FoldingContext &);
  DeclTypeSpec &MakeDerivedType(const Symbol &);
  DeclTypeSpec &MakeDerivedType(DerivedTypeSpec &&, DeclTypeSpec::Category);
  const DeclTypeSpec &MakeTypeStarType();
  const DeclTypeSpec &MakeClassStarType();

  // For modules read from module files, this is the stream of characters
  // that are referenced by SourceName objects.
  void set_chars(std::string &&chars) { chars_ = std::move(chars); }

  ImportKind GetImportKind() const;
  // Names appearing in IMPORT statements in this scope
  std::set<SourceName> importNames() const { return importNames_; }

  // Set the kind of imports from host into this scope.
  // Return an error message for incompatible kinds.
  std::optional<parser::MessageFixedText> SetImportKind(ImportKind);

  void add_importName(const SourceName &);

  const DerivedTypeSpec *derivedTypeSpec() const { return derivedTypeSpec_; }
  void set_derivedTypeSpec(const DerivedTypeSpec &spec) {
    derivedTypeSpec_ = &spec;
  }

  // The range of the source of this and nested scopes.
  const parser::CharBlock &sourceRange() const { return sourceRange_; }
  void AddSourceRange(const parser::CharBlock &);
  // Find the smallest scope under this one that contains source
  const Scope *FindScope(const parser::CharBlock &) const;

  // Attempts to find a match for a derived type instance
  const DeclTypeSpec *FindInstantiatedDerivedType(
      const DerivedTypeSpec &, DeclTypeSpec::Category) const;

  // Returns a matching derived type instance if one exists, otherwise
  // creates one
  const DeclTypeSpec &FindOrInstantiateDerivedType(
      DerivedTypeSpec &&, DeclTypeSpec::Category, evaluate::FoldingContext &);

  // Clones a DerivedType scope into a new derived type instance's scope.
  void InstantiateDerivedType(Scope &, evaluate::FoldingContext &) const;

  const DeclTypeSpec &InstantiateIntrinsicType(
      const DeclTypeSpec &, evaluate::FoldingContext &);

private:
  Scope &parent_;  // this is enclosing scope, not extended derived type base
  const Kind kind_;
  parser::CharBlock sourceRange_;
  Symbol *const symbol_;  // if not null, symbol_->scope() == this
  std::list<Scope> children_;
  mapType symbols_;
  std::map<SourceName, Scope *> submodules_;
  std::list<DeclTypeSpec> declTypeSpecs_;
  std::string chars_;
  std::optional<ImportKind> importKind_;
  std::set<SourceName> importNames_;
  const DerivedTypeSpec *derivedTypeSpec_{nullptr};  // dTS->scope() == this
  // When additional data members are added to Scope, remember to
  // copy them, if appropriate, in InstantiateDerivedType().

  // Storage for all Symbols. Every Symbol is in allSymbols and every Symbol*
  // or Symbol& points to one in there.
  static Symbols<1024> allSymbols;

  bool CanImport(const SourceName &) const;
  const DeclTypeSpec &MakeLengthlessType(DeclTypeSpec &&);

  friend std::ostream &operator<<(std::ostream &, const Scope &);
};
}
#endif  // FORTRAN_SEMANTICS_SCOPE_H_
