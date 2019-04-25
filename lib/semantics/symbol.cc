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

#include "symbol.h"
#include "scope.h"
#include "semantics.h"
#include "../common/idioms.h"
#include "../evaluate/fold.h"
#include <ostream>
#include <string>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &os, const parser::CharBlock &name) {
  return os << name.ToString();
}

template<typename T>
static void DumpOptional(
    std::ostream &os, const char *label, const std::optional<T> &x) {
  if (x) {
    os << ' ' << label << ':' << *x;
  }
}

static void DumpBool(std::ostream &os, const char *label, bool x) {
  if (x) {
    os << ' ' << label;
  }
}

static void DumpSymbolList(std::ostream &os, const SymbolList &list) {
  char sep{' '};
  for (const auto *elem : list) {
    os << sep << elem->name();
    sep = ',';
  }
}

static void DumpType(std::ostream &os, const Symbol &symbol) {
  if (const auto *type{symbol.GetType()}) {
    os << *type << ' ';
  }
}
static void DumpType(std::ostream &os, const DeclTypeSpec *type) {
  if (type) {
    os << ' ' << *type;
  }
}

template<typename T>
static void DumpList(
    std::ostream &os, const char *label, const std::list<T> &list) {
  if (!list.empty()) {
    os << ' ' << label << ':';
    char sep{' '};
    for (const auto &elem : list) {
      os << sep << elem;
      sep = ',';
    }
  }
}

const Scope *ModuleDetails::parent() const {
  return isSubmodule_ && scope_ ? &scope_->parent() : nullptr;
}
const Scope *ModuleDetails::ancestor() const {
  if (!isSubmodule_ || !scope_) {
    return nullptr;
  }
  for (auto *scope{scope_};;) {
    auto *parent{&scope->parent()};
    if (parent->kind() != Scope::Kind::Module) {
      return scope;
    }
    scope = parent;
  }
}
void ModuleDetails::set_scope(const Scope *scope) {
  CHECK(!scope_);
  bool scopeIsSubmodule{scope->parent().kind() == Scope::Kind::Module};
  CHECK(isSubmodule_ == scopeIsSubmodule);
  scope_ = scope;
}

std::ostream &operator<<(std::ostream &os, const SubprogramDetails &x) {
  DumpBool(os, "isInterface", x.isInterface_);
  DumpOptional(os, "bindName", x.bindName_);
  if (x.result_) {
    os << " result:" << x.result_->name();
    if (!x.result_->attrs().empty()) {
      os << ", " << x.result_->attrs();
    }
  }
  if (x.dummyArgs_.empty()) {
    char sep{'('};
    os << ' ';
    for (const auto *arg : x.dummyArgs_) {
      os << sep << arg->name();
      sep = ',';
    }
    os << ')';
  }
  return os;
}

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = &type;
}

void EntityDetails::ReplaceType(const DeclTypeSpec &type) { type_ = &type; }

void ObjectEntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}
void ObjectEntityDetails::set_coshape(const ArraySpec &coshape) {
  CHECK(coshape_.empty());
  for (const auto &shapeSpec : coshape) {
    coshape_.push_back(shapeSpec);
  }
}

ProcEntityDetails::ProcEntityDetails(EntityDetails &&d) : EntityDetails(d) {
  if (type()) {
    interface_.set_type(*type());
  }
}

const Symbol &UseDetails::module() const {
  // owner is a module so it must have a symbol:
  return *symbol_->owner().symbol();
}

UseErrorDetails::UseErrorDetails(const UseDetails &useDetails) {
  add_occurrence(useDetails.location(), *useDetails.module().scope());
}
UseErrorDetails &UseErrorDetails::add_occurrence(
    const SourceName &location, const Scope &module) {
  occurrences_.push_back(std::make_pair(location, &module));
  return *this;
}

GenericDetails::GenericDetails(const SymbolList &specificProcs) {
  for (const auto *proc : specificProcs) {
    add_specificProc(*proc);
  }
}

void GenericDetails::set_specific(Symbol &specific) {
  CHECK(!specific_);
  specific_ = &specific;
}
void GenericDetails::set_derivedType(Symbol &derivedType) {
  CHECK(!derivedType_);
  derivedType_ = &derivedType;
}

const Symbol *GenericDetails::CheckSpecific() const {
  return const_cast<GenericDetails *>(this)->CheckSpecific();
}
Symbol *GenericDetails::CheckSpecific() {
  if (specific_) {
    for (const auto *proc : specificProcs_) {
      if (proc == specific_) {
        return nullptr;
      }
    }
    return specific_;
  } else {
    return nullptr;
  }
}

// The name of the kind of details for this symbol.
// This is primarily for debugging.
std::string DetailsToString(const Details &details) {
  return std::visit(
      common::visitors{
          [](const UnknownDetails &) { return "Unknown"; },
          [](const MainProgramDetails &) { return "MainProgram"; },
          [](const ModuleDetails &) { return "Module"; },
          [](const SubprogramDetails &) { return "Subprogram"; },
          [](const SubprogramNameDetails &) { return "SubprogramName"; },
          [](const EntityDetails &) { return "Entity"; },
          [](const ObjectEntityDetails &) { return "ObjectEntity"; },
          [](const ProcEntityDetails &) { return "ProcEntity"; },
          [](const DerivedTypeDetails &) { return "DerivedType"; },
          [](const UseDetails &) { return "Use"; },
          [](const UseErrorDetails &) { return "UseError"; },
          [](const HostAssocDetails &) { return "HostAssoc"; },
          [](const GenericDetails &) { return "Generic"; },
          [](const ProcBindingDetails &) { return "ProcBinding"; },
          [](const GenericBindingDetails &) { return "GenericBinding"; },
          [](const NamelistDetails &) { return "Namelist"; },
          [](const CommonBlockDetails &) { return "CommonBlockDetails"; },
          [](const FinalProcDetails &) { return "FinalProc"; },
          [](const TypeParamDetails &) { return "TypeParam"; },
          [](const MiscDetails &) { return "Misc"; },
          [](const AssocEntityDetails &) { return "AssocEntity"; },
      },
      details);
}

const std::string Symbol::GetDetailsName() const {
  return DetailsToString(details_);
}

void Symbol::set_details(Details &&details) {
  CHECK(CanReplaceDetails(details));
  details_ = std::move(details);
}

bool Symbol::CanReplaceDetails(const Details &details) const {
  if (has<UnknownDetails>()) {
    return true;  // can always replace UnknownDetails
  } else {
    return std::visit(
        common::visitors{
            [](const UseErrorDetails &) { return true; },
            [=](const ObjectEntityDetails &) { return has<EntityDetails>(); },
            [=](const ProcEntityDetails &) { return has<EntityDetails>(); },
            [=](const SubprogramDetails &) {
              return has<SubprogramNameDetails>();
            },
            [](const auto &) { return false; },
        },
        details);
  }
}

Symbol &Symbol::GetUltimate() {
  return const_cast<Symbol &>(const_cast<const Symbol *>(this)->GetUltimate());
}
const Symbol &Symbol::GetUltimate() const {
  if (const auto *details{detailsIf<UseDetails>()}) {
    return details->symbol().GetUltimate();
  } else if (const auto *details{detailsIf<HostAssocDetails>()}) {
    return details->symbol().GetUltimate();
  } else {
    return *this;
  }
}

void Symbol::SetType(const DeclTypeSpec &type) {
  std::visit(
      common::visitors{
          [&](EntityDetails &x) { x.set_type(type); },
          [&](ObjectEntityDetails &x) { x.set_type(type); },
          [&](AssocEntityDetails &x) { x.set_type(type); },
          [&](ProcEntityDetails &x) { x.interface().set_type(type); },
          [&](TypeParamDetails &x) { x.set_type(type); },
          [](auto &) {},
      },
      details_);
}

bool Symbol::IsDummy() const {
  return std::visit(
      common::visitors{[](const EntityDetails &x) { return x.isDummy(); },
          [](const ObjectEntityDetails &x) { return x.isDummy(); },
          [](const ProcEntityDetails &x) { return x.isDummy(); },
          [](const HostAssocDetails &x) { return x.symbol().IsDummy(); },
          [](const auto &) { return false; }},
      details_);
}

bool Symbol::IsFuncResult() const {
  return std::visit(
      common::visitors{[](const EntityDetails &x) { return x.isFuncResult(); },
          [](const ObjectEntityDetails &x) { return x.isFuncResult(); },
          [](const ProcEntityDetails &x) { return x.isFuncResult(); },
          [](const HostAssocDetails &x) { return x.symbol().IsFuncResult(); },
          [](const auto &) { return false; }},
      details_);
}

bool Symbol::IsObjectArray() const {
  const auto *details{std::get_if<ObjectEntityDetails>(&details_)};
  return details && details->IsArray();
}

bool Symbol::IsSubprogram() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const UseDetails &x) { return x.symbol().IsSubprogram(); },
          [](const auto &) { return false; },
      },
      details_);
}

bool Symbol::IsSeparateModuleProc() const {
  if (attrs().test(Attr::MODULE)) {
    if (auto *details{detailsIf<SubprogramDetails>()}) {
      return details->isInterface();
    }
  }
  return false;
}

ObjectEntityDetails::ObjectEntityDetails(EntityDetails &&d)
  : EntityDetails(d) {}

std::ostream &operator<<(std::ostream &os, const EntityDetails &x) {
  DumpBool(os, "dummy", x.isDummy());
  DumpBool(os, "funcResult", x.isFuncResult());
  if (x.type()) {
    os << " type: " << *x.type();
  }
  DumpOptional(os, "bindName", x.bindName_);
  return os;
}

std::ostream &operator<<(std::ostream &os, const ObjectEntityDetails &x) {
  os << *static_cast<const EntityDetails *>(&x);
  DumpList(os, "shape", x.shape());
  DumpList(os, "coshape", x.coshape());
  DumpOptional(os, "init", x.init_);
  return os;
}

std::ostream &operator<<(std::ostream &os, const AssocEntityDetails &x) {
  os << *static_cast<const EntityDetails *>(&x);
  DumpOptional(os, "expr", x.expr());
  return os;
}

std::ostream &operator<<(std::ostream &os, const ProcEntityDetails &x) {
  if (auto *symbol{x.interface_.symbol()}) {
    os << ' ' << symbol->name();
  } else {
    DumpType(os, x.interface_.type());
  }
  DumpOptional(os, "bindName", x.bindName());
  DumpOptional(os, "passName", x.passName());
  return os;
}

std::ostream &operator<<(std::ostream &os, const DerivedTypeDetails &x) {
  DumpBool(os, "sequence", x.sequence_);
  DumpList(os, "components", x.componentNames_);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Details &details) {
  os << DetailsToString(details);
  std::visit(
      common::visitors{
          [&](const UnknownDetails &x) {},
          [&](const MainProgramDetails &x) {},
          [&](const ModuleDetails &x) {
            if (x.isSubmodule()) {
              os << " (";
              if (x.ancestor()) {
                auto &ancestor{x.ancestor()->name()};
                os << ancestor;
                if (x.parent()) {
                  auto &parent{x.parent()->name()};
                  if (ancestor != parent) {
                    os << ':' << parent;
                  }
                }
              }
              os << ")";
            }
          },
          [&](const SubprogramDetails &x) {
            os << " (";
            int n = 0;
            for (const auto &dummy : x.dummyArgs()) {
              if (n++ > 0) os << ", ";
              DumpType(os, *dummy);
              os << dummy->name();
            }
            os << ')';
            DumpOptional(os, "bindName", x.bindName());
            if (x.isFunction()) {
              os << " result(";
              DumpType(os, x.result());
              os << x.result().name() << ')';
            }
            DumpBool(os, "interface", x.isInterface());
          },
          [&](const SubprogramNameDetails &x) {
            os << ' ' << EnumToString(x.kind());
          },
          [&](const UseDetails &x) {
            os << " from " << x.symbol().name() << " in " << x.module().name();
          },
          [&](const UseErrorDetails &x) {
            os << " uses:";
            for (const auto &[location, module] : x.occurrences()) {
              os << " from " << module->name() << " at " << location;
            }
          },
          [](const HostAssocDetails &) {},
          [&](const GenericDetails &x) {
            os << ' ' << EnumToString(x.kind());
            DumpSymbolList(os, x.specificProcs());
          },
          [&](const ProcBindingDetails &x) {
            os << " => " << x.symbol().name();
            DumpOptional(os, "passName", x.passName());
          },
          [&](const GenericBindingDetails &x) {
            os << " =>";
            DumpSymbolList(os, x.specificProcs());
          },
          [&](const NamelistDetails &x) {
            os << ':';
            DumpSymbolList(os, x.objects());
          },
          [&](const CommonBlockDetails &x) {
            os << ':';
            for (const auto *object : x.objects()) {
              os << ' ' << object->name();
            }
          },
          [&](const FinalProcDetails &) {},
          [&](const TypeParamDetails &x) {
            if (x.type()) {
              os << ' ' << *x.type();
            }
            os << ' ' << common::EnumToString(x.attr());
            DumpOptional(os, "init", x.init());
          },
          [&](const MiscDetails &x) {
            os << ' ' << MiscDetails::EnumToString(x.kind());
          },
          [&](const auto &x) { os << x; },
      },
      details);
  return os;
}

std::ostream &operator<<(std::ostream &o, Symbol::Flag flag) {
  return o << Symbol::EnumToString(flag);
}

std::ostream &operator<<(std::ostream &o, const Symbol::Flags &flags) {
  std::size_t n{flags.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Symbol::Flag flag{static_cast<Symbol::Flag>(j)};
    if (flags.test(flag)) {
      if (seen++ > 0) {
        o << ", ";
      }
      o << flag;
    }
  }
  return o;
}

std::ostream &operator<<(std::ostream &os, const Symbol &symbol) {
  os << symbol.name();
  if (!symbol.attrs().empty()) {
    os << ", " << symbol.attrs();
  }
  if (!symbol.flags().empty()) {
    os << " (" << symbol.flags() << ')';
  }
  os << ": " << symbol.details_;
  return os;
}

// Output a unique name for a scope by qualifying it with the names of
// parent scopes. For scopes without corresponding symbols, use the kind
// with an index (e.g. Block1, Block2, etc.).
static void DumpUniqueName(std::ostream &os, const Scope &scope) {
  if (scope.kind() != Scope::Kind::Global) {
    DumpUniqueName(os, scope.parent());
    os << '/';
    if (auto *scopeSymbol{scope.symbol()}) {
      os << scopeSymbol->name();
    } else {
      int index{1};
      for (auto &child : scope.parent().children()) {
        if (child == scope) {
          break;
        }
        if (child.kind() == scope.kind()) {
          ++index;
        }
      }
      os << Scope::EnumToString(scope.kind()) << index;
    }
  }
}

// Dump a symbol for UnparseWithSymbols. This will be used for tests so the
// format should be reasonably stable.
std::ostream &DumpForUnparse(
    std::ostream &os, const Symbol &symbol, bool isDef) {
  DumpUniqueName(os, symbol.owner());
  os << '/' << symbol.name();
  if (isDef) {
    if (!symbol.attrs().empty()) {
      os << ' ' << symbol.attrs();
    }
    DumpBool(os, "(implicit)", symbol.test(Symbol::Flag::Implicit));
    DumpBool(os, "(local)", symbol.test(Symbol::Flag::LocalityLocal));
    DumpBool(os, "(local_init)", symbol.test(Symbol::Flag::LocalityLocalInit));
    DumpBool(os, "(shared)", symbol.test(Symbol::Flag::LocalityShared));
    os << ' ' << symbol.GetDetailsName();
    DumpType(os, symbol.GetType());
  }
  return os;
}

Symbol &Symbol::Instantiate(
    Scope &scope, SemanticsContext &semanticsContext) const {
  evaluate::FoldingContext foldingContext{semanticsContext.foldingContext()};
  CHECK(foldingContext.pdtInstance() != nullptr);
  const DerivedTypeSpec &instanceSpec{*foldingContext.pdtInstance()};
  auto pair{scope.try_emplace(name_, attrs_)};
  Symbol &symbol{*pair.first->second};
  if (!pair.second) {
    // Symbol was already present in the scope, which can only happen
    // in the case of type parameters with actual or default values.
    CHECK(has<TypeParamDetails>());
    return symbol;
  }
  symbol.attrs_ = attrs_;
  symbol.flags_ = flags_;
  std::visit(
      common::visitors{
          [&](const ObjectEntityDetails &that) {
            symbol.details_ = that;
            ObjectEntityDetails &details{symbol.get<ObjectEntityDetails>()};
            if (DeclTypeSpec * origType{symbol.GetType()}) {
              if (const DerivedTypeSpec * derived{origType->AsDerived()}) {
                DerivedTypeSpec newSpec{*derived};
                if (test(Flag::ParentComp)) {
                  // Forward all explicit type parameter values from the
                  // derived type spec under instantiation to this parent
                  // component spec when they define type parameters that
                  // pertain to the parent component.
                  for (const auto &pair : instanceSpec.parameters()) {
                    if (scope.find(pair.first) == scope.end()) {
                      newSpec.AddParamValue(
                          pair.first, ParamValue{pair.second});
                    }
                  }
                }
                details.ReplaceType(
                    scope.FindOrInstantiateDerivedType(std::move(newSpec),
                        semanticsContext, origType->category()));
              } else if (origType->AsIntrinsic() != nullptr) {
                const DeclTypeSpec &newType{scope.InstantiateIntrinsicType(
                    *origType, semanticsContext)};
                details.ReplaceType(newType);
              } else {
                common::die("instantiated component has type that is "
                            "neither intrinsic nor derived");
              }
            }
            details.set_init(
                evaluate::Fold(foldingContext, std::move(details.init())));
            for (ShapeSpec &dim : details.shape()) {
              if (dim.lbound().isExplicit()) {
                dim.lbound().SetExplicit(Fold(
                    foldingContext, std::move(dim.lbound().GetExplicit())));
              }
              if (dim.ubound().isExplicit()) {
                dim.ubound().SetExplicit(Fold(
                    foldingContext, std::move(dim.ubound().GetExplicit())));
              }
            }
            for (ShapeSpec &dim : details.coshape()) {
              if (dim.lbound().isExplicit()) {
                dim.lbound().SetExplicit(Fold(
                    foldingContext, std::move(dim.lbound().GetExplicit())));
              }
              if (dim.ubound().isExplicit()) {
                dim.ubound().SetExplicit(Fold(
                    foldingContext, std::move(dim.ubound().GetExplicit())));
              }
            }
          },
          [&](const ProcBindingDetails &that) { symbol.details_ = that; },
          [&](const GenericBindingDetails &that) { symbol.details_ = that; },
          [&](const ProcEntityDetails &that) { symbol.details_ = that; },
          [&](const TypeParamDetails &that) {
            // LEN type parameter, or error recovery on a KIND type parameter
            // with no corresponding actual argument or default
            symbol.details_ = that;
          },
          [&](const auto &that) {
            common::die("unexpected details in Symbol::Instantiate");
          },
      },
      details_);
  return symbol;
}

const Symbol *Symbol::GetParentComponent(const Scope *scope) const {
  if (scope == nullptr) {
    CHECK(scope_ != nullptr);
    scope = scope_;
  }
  return get<DerivedTypeDetails>().GetParentComponent(*scope);
}

const DerivedTypeSpec *Symbol::GetParentTypeSpec(const Scope *scope) const {
  if (const Symbol * parentComponent{GetParentComponent(scope)}) {
    const auto &object{parentComponent->get<ObjectEntityDetails>()};
    return &object.type()->derivedTypeSpec();
  } else {
    return nullptr;
  }
}

void DerivedTypeDetails::add_component(const Symbol &symbol) {
  if (symbol.test(Symbol::Flag::ParentComp)) {
    CHECK(componentNames_.empty());
  }
  componentNames_.push_back(symbol.name());
}

std::list<SourceName> DerivedTypeDetails::OrderParameterNames(
    const Symbol &type) const {
  std::list<SourceName> result;
  if (const DerivedTypeSpec * spec{type.GetParentTypeSpec()}) {
    const DerivedTypeDetails &details{
        spec->typeSymbol().get<DerivedTypeDetails>()};
    result = details.OrderParameterNames(spec->typeSymbol());
  }
  for (const auto &name : paramNames_) {
    result.push_back(name);
  }
  return result;
}

SymbolList DerivedTypeDetails::OrderParameterDeclarations(
    const Symbol &type) const {
  SymbolList result;
  if (const DerivedTypeSpec * spec{type.GetParentTypeSpec()}) {
    const DerivedTypeDetails &details{
        spec->typeSymbol().get<DerivedTypeDetails>()};
    result = details.OrderParameterDeclarations(spec->typeSymbol());
  }
  for (const Symbol *symbol : paramDecls_) {
    result.push_back(symbol);
  }
  return result;
}

SymbolList DerivedTypeDetails::OrderComponents(const Scope &scope) const {
  SymbolList result;
  for (SourceName name : componentNames_) {
    auto iter{scope.find(name)};
    if (iter != scope.cend()) {
      const Symbol &symbol{*iter->second};
      if (symbol.test(Symbol::Flag::ParentComp)) {
        CHECK(result.empty());
        const DerivedTypeSpec &spec{
            symbol.get<ObjectEntityDetails>().type()->derivedTypeSpec()};
        result = spec.typeSymbol().get<DerivedTypeDetails>().OrderComponents(
            *spec.scope());
      }
      result.push_back(&symbol);
    }
  }
  return result;
}

const Symbol *DerivedTypeDetails::GetParentComponent(const Scope &scope) const {
  if (!componentNames_.empty()) {
    SourceName extends{componentNames_.front()};
    auto iter{scope.find(extends)};
    if (iter != scope.cend()) {
      const Symbol &symbol{*iter->second};
      if (symbol.test(Symbol::Flag::ParentComp)) {
        return &symbol;
      }
    }
  }
  return nullptr;
}

void TypeParamDetails::set_type(const DeclTypeSpec &type) {
  CHECK(type_ == nullptr);
  type_ = &type;
}

}
