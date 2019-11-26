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

#include "tools.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/indirection.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include "../parser/tools.h"
#include <algorithm>
#include <set>
#include <variant>

namespace Fortran::semantics {

static const Symbol *FindCommonBlockInScope(
    const Scope &scope, const Symbol &object) {
  for (const auto &pair : scope.commonBlocks()) {
    const Symbol &block{*pair.second};
    if (IsCommonBlockContaining(block, object)) {
      return &block;
    }
  }
  return nullptr;
}

const Symbol *FindCommonBlockContaining(const Symbol &object) {
  for (const Scope *scope{&object.owner()}; !scope->IsGlobal();
       scope = &scope->parent()) {
    if (const Symbol * block{FindCommonBlockInScope(*scope, object)}) {
      return block;
    }
  }
  return nullptr;
}

const Scope *FindProgramUnitContaining(const Scope &start) {
  const Scope *scope{&start};
  while (scope) {
    switch (scope->kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram: return scope;
    case Scope::Kind::Global: return nullptr;
    case Scope::Kind::DerivedType:
    case Scope::Kind::Block:
    case Scope::Kind::Forall:
    case Scope::Kind::ImpliedDos: scope = &scope->parent();
    }
  }
  return nullptr;
}

const Scope *FindProgramUnitContaining(const Symbol &symbol) {
  return FindProgramUnitContaining(symbol.owner());
}

const Scope *FindPureProcedureContaining(const Scope &start) {
  // N.B. We only need to examine the innermost containing program unit
  // because an internal subprogram of a PURE subprogram must also
  // be PURE (C1592).
  if (const Scope * scope{FindProgramUnitContaining(start)}) {
    if (IsPureProcedure(*scope)) {
      return scope;
    }
  }
  return nullptr;
}

Tristate IsDefinedAssignment(
    const std::optional<evaluate::DynamicType> &lhsType, int lhsRank,
    const std::optional<evaluate::DynamicType> &rhsType, int rhsRank) {
  if (!lhsType || !rhsType) {
    return Tristate::No;  // error or rhs is untyped
  }
  TypeCategory lhsCat{lhsType->category()};
  TypeCategory rhsCat{rhsType->category()};
  if (rhsRank > 0 && lhsRank != rhsRank) {
    return Tristate::Yes;
  } else if (lhsCat != TypeCategory::Derived) {
    return ToTristate(lhsCat != rhsCat &&
        (!IsNumericTypeCategory(lhsCat) || !IsNumericTypeCategory(rhsCat)));
  } else if (rhsCat == TypeCategory::Derived &&
      lhsType->GetDerivedTypeSpec() == rhsType->GetDerivedTypeSpec()) {
    return Tristate::Maybe;  // TYPE(t) = TYPE(t) can be defined or intrinsic
  } else {
    return Tristate::Yes;
  }
}

bool IsGenericDefinedOp(const Symbol &symbol) {
  const auto *details{symbol.GetUltimate().detailsIf<GenericDetails>()};
  return details && details->kind().IsDefinedOperator();
}

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object) {
  const auto &objects{block.get<CommonBlockDetails>().objects()};
  auto found{std::find(objects.begin(), objects.end(), object)};
  return found != objects.end();
}

bool IsUseAssociated(const Symbol &symbol, const Scope &scope) {
  const Scope *owner{FindProgramUnitContaining(symbol.GetUltimate().owner())};
  return owner && owner->kind() == Scope::Kind::Module &&
      owner != FindProgramUnitContaining(scope);
}

bool DoesScopeContain(
    const Scope *maybeAncestor, const Scope &maybeDescendent) {
  if (maybeAncestor) {
    const Scope *scope{&maybeDescendent};
    while (!scope->IsGlobal()) {
      scope = &scope->parent();
      if (scope == maybeAncestor) {
        return true;
      }
    }
  }
  return false;
}

bool DoesScopeContain(const Scope *maybeAncestor, const Symbol &symbol) {
  return DoesScopeContain(maybeAncestor, symbol.owner());
}

bool IsHostAssociated(const Symbol &symbol, const Scope &scope) {
  return DoesScopeContain(FindProgramUnitContaining(symbol), scope);
}

bool IsDummy(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    return details->isDummy();
  } else if (const auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    return details->isDummy();
  } else {
    return false;
  }
}

bool IsPointerDummy(const Symbol &symbol) {
  return IsPointer(symbol) && IsDummy(symbol);
}

// variable-name
bool IsVariableName(const Symbol &symbol) {
  if (const Symbol * root{GetAssociationRoot(symbol)}) {
    return root->has<ObjectEntityDetails>() && !IsNamedConstant(*root);
  } else {
    return false;
  }
}

// proc-name
bool IsProcName(const Symbol &symbol) {
  return symbol.GetUltimate().has<ProcEntityDetails>();
}

bool IsFunction(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &x) { return x.isFunction(); },
          [&](const SubprogramNameDetails &) {
            return symbol.test(Symbol::Flag::Function);
          },
          [](const ProcEntityDetails &x) {
            const auto &ifc{x.interface()};
            return ifc.type() || (ifc.symbol() && IsFunction(*ifc.symbol()));
          },
          [](const ProcBindingDetails &x) { return IsFunction(x.symbol()); },
          [](const UseDetails &x) { return IsFunction(x.symbol()); },
          [](const auto &) { return false; },
      },
      symbol.details());
}

bool IsPureProcedure(const Symbol &symbol) {
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (const Symbol * procInterface{procDetails->interface().symbol()}) {
      // procedure component with a PURE interface
      return IsPureProcedure(*procInterface);
    }
  }
  return symbol.attrs().test(Attr::PURE) && IsProcedure(symbol);
}

bool IsPureProcedure(const Scope &scope) {
  if (const Symbol * symbol{scope.GetSymbol()}) {
    return IsPureProcedure(*symbol);
  } else {
    return false;
  }
}

bool IsBindCProcedure(const Symbol &symbol) {
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (const Symbol * procInterface{procDetails->interface().symbol()}) {
      // procedure component with a BIND(C) interface
      return IsBindCProcedure(*procInterface);
    }
  }
  return symbol.attrs().test(Attr::BIND_C) && IsProcedure(symbol);
}

bool IsBindCProcedure(const Scope &scope) {
  if (const Symbol * symbol{scope.GetSymbol()}) {
    return IsBindCProcedure(*symbol);
  } else {
    return false;
  }
}

bool IsProcedure(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const ProcEntityDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const ProcBindingDetails &) { return true; },
          [](const UseDetails &x) { return IsProcedure(x.symbol()); },
          // TODO: FinalProcDetails?
          [](const auto &) { return false; },
      },
      symbol.details());
}

bool IsProcedurePointer(const Symbol &symbol) {
  return symbol.has<ProcEntityDetails>() && IsPointer(symbol);
}

static const Symbol *FindPointerComponent(
    const Scope &scope, std::set<const Scope *> &visited) {
  if (!scope.IsDerivedType()) {
    return nullptr;
  }
  if (!visited.insert(&scope).second) {
    return nullptr;
  }
  // If there's a top-level pointer component, return it for clearer error
  // messaging.
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (IsPointer(symbol)) {
      return &symbol;
    }
  }
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (const Scope * nested{derived->scope()}) {
            if (const Symbol *
                pointer{FindPointerComponent(*nested, visited)}) {
              return pointer;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

const Symbol *FindPointerComponent(const Scope &scope) {
  std::set<const Scope *> visited;
  return FindPointerComponent(scope, visited);
}

const Symbol *FindPointerComponent(const DerivedTypeSpec &derived) {
  if (const Scope * scope{derived.scope()}) {
    return FindPointerComponent(*scope);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec &type) {
  if (const DerivedTypeSpec * derived{type.AsDerived()}) {
    return FindPointerComponent(*derived);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec *type) {
  return type ? FindPointerComponent(*type) : nullptr;
}

const Symbol *FindPointerComponent(const Symbol &symbol) {
  return IsPointer(symbol) ? &symbol : FindPointerComponent(symbol.GetType());
}

// C1594 specifies several ways by which an object might be globally visible.
const Symbol *FindExternallyVisibleObject(
    const Symbol &object, const Scope &scope) {
  // TODO: Storage association with any object for which this predicate holds,
  // once EQUIVALENCE is supported.
  if (IsUseAssociated(object, scope) || IsHostAssociated(object, scope) ||
      (IsPureProcedure(scope) && IsPointerDummy(object)) ||
      (IsIntentIn(object) && IsDummy(object))) {
    return &object;
  } else if (const Symbol * block{FindCommonBlockContaining(object)}) {
    return block;
  } else {
    return nullptr;
  }
}

bool ExprHasTypeCategory(
    const SomeExpr &expr, const common::TypeCategory &type) {
  auto dynamicType{expr.GetType()};
  return dynamicType && dynamicType->category() == type;
}

bool ExprTypeKindIsDefault(
    const SomeExpr &expr, const SemanticsContext &context) {
  auto dynamicType{expr.GetType()};
  return dynamicType &&
      dynamicType->category() != common::TypeCategory::Derived &&
      dynamicType->kind() == context.GetDefaultKind(dynamicType->category());
}

const evaluate::Assignment *GetAssignment(const parser::AssignmentStmt &x) {
  const auto &typed{x.typedAssignment};
  return typed && typed->v ? &*typed->v : nullptr;
}

const Symbol *FindInterface(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const ProcEntityDetails &details) {
            return details.interface().symbol();
          },
          [](const ProcBindingDetails &details) { return &details.symbol(); },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
}

const Symbol *FindSubprogram(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [&](const ProcEntityDetails &details) -> const Symbol * {
            if (const Symbol * interface{details.interface().symbol()}) {
              return FindSubprogram(*interface);
            } else {
              return &symbol;
            }
          },
          [](const ProcBindingDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [&](const SubprogramDetails &) { return &symbol; },
          [](const UseDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [](const HostAssocDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
}

const Symbol *FindFunctionResult(const Symbol &symbol) {
  if (const Symbol * subp{FindSubprogram(symbol)}) {
    if (const auto &subpDetails{subp->detailsIf<SubprogramDetails>()}) {
      if (subpDetails->isFunction()) {
        return &subpDetails->result();
      }
    }
  }
  return nullptr;
}

const Symbol *FindOverriddenBinding(const Symbol &symbol) {
  if (symbol.has<ProcBindingDetails>()) {
    if (const DeclTypeSpec * parentType{FindParentTypeSpec(symbol.owner())}) {
      if (const DerivedTypeSpec * parentDerived{parentType->AsDerived()}) {
        if (const Scope * parentScope{parentDerived->typeSymbol().scope()}) {
          return parentScope->FindComponent(symbol.name());
        }
      }
    }
  }
  return nullptr;
}

const DeclTypeSpec *FindParentTypeSpec(const DerivedTypeSpec &derived) {
  return FindParentTypeSpec(derived.typeSymbol());
}

const DeclTypeSpec *FindParentTypeSpec(const DeclTypeSpec &decl) {
  if (const DerivedTypeSpec * derived{decl.AsDerived()}) {
    return FindParentTypeSpec(*derived);
  } else {
    return nullptr;
  }
}

const DeclTypeSpec *FindParentTypeSpec(const Scope &scope) {
  if (scope.kind() == Scope::Kind::DerivedType) {
    if (const auto *symbol{scope.symbol()}) {
      return FindParentTypeSpec(*symbol);
    }
  }
  return nullptr;
}

const DeclTypeSpec *FindParentTypeSpec(const Symbol &symbol) {
  if (const Scope * scope{symbol.scope()}) {
    if (const auto *details{symbol.detailsIf<DerivedTypeDetails>()}) {
      if (const Symbol * parent{details->GetParentComponent(*scope)}) {
        return parent->GetType();
      }
    }
  }
  return nullptr;
}

// When an construct association maps to a variable, and that variable
// is not an array with a vector-valued subscript, return the base
// Symbol of that variable, else nullptr.  Descends into other construct
// associations when one associations maps to another.
static const Symbol *GetAssociatedVariable(const AssocEntityDetails &details) {
  if (const MaybeExpr & expr{details.expr()}) {
    if (evaluate::IsVariable(*expr) && !evaluate::HasVectorSubscript(*expr)) {
      if (const Symbol * varSymbol{evaluate::GetFirstSymbol(*expr)}) {
        return GetAssociationRoot(*varSymbol);
      }
    }
  }
  return nullptr;
}

// Return the Symbol of the variable of a construct association, if it exists
// Return nullptr if the name is associated with an expression
const Symbol *GetAssociationRoot(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  if (const auto *details{ultimate.detailsIf<AssocEntityDetails>()}) {
    // We have a construct association
    return GetAssociatedVariable(*details);
  } else {
    return &ultimate;
  }
}

bool IsExtensibleType(const DerivedTypeSpec *derived) {
  return derived && !IsIsoCType(derived) &&
      !derived->typeSymbol().attrs().test(Attr::BIND_C) &&
      !derived->typeSymbol().get<DerivedTypeDetails>().sequence();
}

bool IsDerivedTypeFromModule(
    const DerivedTypeSpec *derived, const char *module, const char *name) {
  if (!derived) {
    return false;
  } else {
    const auto &symbol{derived->typeSymbol()};
    return symbol.name() == name && symbol.owner().IsModule() &&
        symbol.owner().GetName().value() == module;
  }
}

bool IsIsoCType(const DerivedTypeSpec *derived) {
  return IsDerivedTypeFromModule(derived, "iso_c_binding", "c_ptr") ||
      IsDerivedTypeFromModule(derived, "iso_c_binding", "c_funptr");
}

bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsDerivedTypeFromModule(derived, "iso_fortran_env", "team_type");
}

bool IsEventTypeOrLockType(const DerivedTypeSpec *derivedTypeSpec) {
  return IsDerivedTypeFromModule(
             derivedTypeSpec, "iso_fortran_env", "event_type") ||
      IsDerivedTypeFromModule(derivedTypeSpec, "iso_fortran_env", "lock_type");
}

bool IsOrContainsEventOrLockComponent(const Symbol &symbol) {
  if (const Symbol * root{GetAssociationRoot(symbol)}) {
    if (const auto *details{root->detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          return IsEventTypeOrLockType(derived) ||
              FindEventOrLockPotentialComponent(*derived);
        }
      }
    }
  }
  return false;
}

bool IsSaved(const Symbol &symbol) {
  auto scopeKind{symbol.owner().kind()};
  if (scopeKind == Scope::Kind::MainProgram ||
      scopeKind == Scope::Kind::Module) {
    return true;
  } else if (scopeKind == Scope::Kind::DerivedType) {
    return false;  // this is a component
  } else if (IsNamedConstant(symbol)) {
    return false;
  } else if (symbol.attrs().test(Attr::SAVE)) {
    return true;
  } else {
    if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (object->init()) {
        return true;
      }
    } else if (IsProcedurePointer(symbol)) {
      if (symbol.get<ProcEntityDetails>().init()) {
        return true;
      }
    }
    if (const Symbol * block{FindCommonBlockContaining(symbol)}) {
      if (block->attrs().test(Attr::SAVE)) {
        return true;
      }
    }
    return false;
  }
}

// Check this symbol suitable as a type-bound procedure - C769
bool CanBeTypeBoundProc(const Symbol *symbol) {
  if (!symbol || IsDummy(*symbol) || IsProcedurePointer(*symbol)) {
    return false;
  } else if (symbol->has<SubprogramNameDetails>()) {
    return symbol->owner().kind() == Scope::Kind::Module;
  } else if (auto *details{symbol->detailsIf<SubprogramDetails>()}) {
    return symbol->owner().kind() == Scope::Kind::Module ||
        details->isInterface();
  } else if (const auto *proc{symbol->detailsIf<ProcEntityDetails>()}) {
    return !symbol->attrs().test(Attr::INTRINSIC) &&
        proc->HasExplicitInterface();
  } else {
    return false;
  }
}

bool IsFinalizable(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      return IsFinalizable(*derived);
    }
  }
  return false;
}

bool IsFinalizable(const DerivedTypeSpec &derived) {
  ScopeComponentIterator components{derived};
  return std::find_if(components.begin(), components.end(),
             [](const Symbol &x) { return x.has<FinalProcDetails>(); }) !=
      components.end();
}

bool HasImpureFinal(const DerivedTypeSpec &derived) {
  ScopeComponentIterator components{derived};
  return std::find_if(
             components.begin(), components.end(), [](const Symbol &x) {
               return x.has<FinalProcDetails>() && !x.attrs().test(Attr::PURE);
             }) != components.end();
}

bool IsCoarray(const Symbol &symbol) { return symbol.Corank() > 0; }

bool IsAssumedLengthCharacter(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    return type->category() == DeclTypeSpec::Character &&
        type->characterTypeSpec().length().isAssumed();
  } else {
    return false;
  }
}

bool IsAssumedLengthCharacterFunction(const Symbol &symbol) {
  // Assumed-length character functions only appear as such in their
  // definitions; their interfaces, pointers to them, and dummy procedures
  // cannot be assumed-length.
  return symbol.has<SubprogramDetails>() && IsAssumedLengthCharacter(symbol);
}

const Symbol *IsExternalInPureContext(
    const Symbol &symbol, const Scope &scope) {
  if (const auto *pureProc{semantics::FindPureProcedureContaining(scope)}) {
    if (const Symbol * root{GetAssociationRoot(symbol)}) {
      if (const Symbol *
          visible{FindExternallyVisibleObject(*root, *pureProc)}) {
        return visible;
      }
    }
  }
  return nullptr;
}

PotentialComponentIterator::const_iterator FindPolymorphicPotentialComponent(
    const DerivedTypeSpec &derived) {
  PotentialComponentIterator potentials{derived};
  return std::find_if(
      potentials.begin(), potentials.end(), [](const Symbol &component) {
        if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
          const DeclTypeSpec *type{details->type()};
          return type && type->IsPolymorphic();
        }
        return false;
      });
}

bool IsOrContainsPolymorphicComponent(const Symbol &symbol) {
  if (const Symbol * root{GetAssociationRoot(symbol)}) {
    if (const auto *details{root->detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (type->IsPolymorphic()) {
          return true;
        }
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          return (bool)FindPolymorphicPotentialComponent(*derived);
        }
      }
    }
  }
  return false;
}

bool InProtectedContext(const Symbol &symbol, const Scope &currentScope) {
  return IsProtected(symbol) && !IsHostAssociated(symbol, currentScope);
}

// C1101 and C1158
// TODO Need to check for a coindexed object (why? C1103?)
std::optional<parser::MessageFixedText> WhyNotModifiable(
    const Symbol &symbol, const Scope &scope) {
  const Symbol *root{GetAssociationRoot(symbol)};
  if (!root) {
    return "'%s' is construct associated with an expression"_en_US;
  } else if (InProtectedContext(*root, scope)) {
    return "'%s' is protected in this scope"_en_US;
  } else if (IsExternalInPureContext(*root, scope)) {
    return "'%s' is externally visible and referenced in a PURE"
           " procedure"_en_US;
  } else if (IsOrContainsEventOrLockComponent(*root)) {
    return "'%s' is an entity with either an EVENT_TYPE or LOCK_TYPE"_en_US;
  } else if (IsIntentIn(*root)) {
    return "'%s' is an INTENT(IN) dummy argument"_en_US;
  } else if (!IsVariableName(*root)) {
    return "'%s' is not a variable"_en_US;
  } else {
    return std::nullopt;
  }
}

std::unique_ptr<parser::Message> WhyNotModifiable(parser::CharBlock at,
    const SomeExpr &expr, const Scope &scope, bool vectorSubscriptIsOk) {
  if (evaluate::IsVariable(expr)) {
    if (auto dataRef{evaluate::ExtractDataRef(expr)}) {
      if (!vectorSubscriptIsOk && evaluate::HasVectorSubscript(expr)) {
        return std::make_unique<parser::Message>(
            at, "variable has a vector subscript"_en_US);
      } else {
        const Symbol &symbol{dataRef->GetFirstSymbol()};
        if (auto maybeWhy{WhyNotModifiable(symbol, scope)}) {
          return std::make_unique<parser::Message>(symbol.name(),
              parser::MessageFormattedText{
                  std::move(*maybeWhy), symbol.name()});
        }
      }
    } else {
      // reference to function returning POINTER
    }
  } else {
    return std::make_unique<parser::Message>(
        at, "expression is not a variable"_en_US);
  }
  return {};
}

class ImageControlStmtHelper {
  using ImageControlStmts = std::variant<parser::ChangeTeamConstruct,
      parser::CriticalConstruct, parser::EventPostStmt, parser::EventWaitStmt,
      parser::FormTeamStmt, parser::LockStmt, parser::StopStmt,
      parser::SyncAllStmt, parser::SyncImagesStmt, parser::SyncMemoryStmt,
      parser::SyncTeamStmt, parser::UnlockStmt>;

public:
  template<typename T> bool operator()(const T &) {
    return common::HasMember<T, ImageControlStmts>;
  }
  template<typename T> bool operator()(const common::Indirection<T> &x) {
    return (*this)(x.value());
  }
  bool operator()(const parser::AllocateStmt &stmt) {
    const auto &allocationList{std::get<std::list<parser::Allocation>>(stmt.t)};
    for (const auto &allocation : allocationList) {
      const auto &allocateObject{
          std::get<parser::AllocateObject>(allocation.t)};
      if (IsCoarrayObject(allocateObject)) {
        return true;
      }
    }
    return false;
  }
  bool operator()(const parser::DeallocateStmt &stmt) {
    const auto &allocateObjectList{
        std::get<std::list<parser::AllocateObject>>(stmt.t)};
    for (const auto &allocateObject : allocateObjectList) {
      if (IsCoarrayObject(allocateObject)) {
        return true;
      }
    }
    return false;
  }
  bool operator()(const parser::CallStmt &stmt) {
    const auto &procedureDesignator{
        std::get<parser::ProcedureDesignator>(stmt.v.t)};
    if (auto *name{std::get_if<parser::Name>(&procedureDesignator.u)}) {
      // TODO: also ensure that the procedure is, in fact, an intrinsic
      if (name->source == "move_alloc") {
        const auto &args{std::get<std::list<parser::ActualArgSpec>>(stmt.v.t)};
        if (!args.empty()) {
          const parser::ActualArg &actualArg{
              std::get<parser::ActualArg>(args.front().t)};
          if (const auto *argExpr{
                  std::get_if<common::Indirection<parser::Expr>>(
                      &actualArg.u)}) {
            return HasCoarray(argExpr->value());
          }
        }
      }
    }
    return false;
  }
  bool operator()(const parser::Statement<parser::ActionStmt> &stmt) {
    return std::visit(*this, stmt.statement.u);
  }

private:
  bool IsCoarrayObject(const parser::AllocateObject &allocateObject) {
    const parser::Name &name{GetLastName(allocateObject)};
    return name.symbol && IsCoarray(*name.symbol);
  }
};

bool IsImageControlStmt(const parser::ExecutableConstruct &construct) {
  return std::visit(ImageControlStmtHelper{}, construct.u);
}

std::optional<parser::MessageFixedText> GetImageControlStmtCoarrayMsg(
    const parser::ExecutableConstruct &construct) {
  if (const auto *actionStmt{
          std::get_if<parser::Statement<parser::ActionStmt>>(&construct.u)}) {
    return std::visit(
        common::visitors{
            [](const common::Indirection<parser::AllocateStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "ALLOCATE of a coarray is an image control"
                     " statement"_en_US;
            },
            [](const common::Indirection<parser::DeallocateStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "DEALLOCATE of a coarray is an image control"
                     " statement"_en_US;
            },
            [](const common::Indirection<parser::CallStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "MOVE_ALLOC of a coarray is an image control"
                     " statement "_en_US;
            },
            [](const auto &) -> std::optional<parser::MessageFixedText> {
              return std::nullopt;
            },
        },
        actionStmt->statement.u);
  }
  return std::nullopt;
}

parser::CharBlock GetImageControlStmtLocation(
    const parser::ExecutableConstruct &executableConstruct) {
  return std::visit(
      common::visitors{
          [](const common::Indirection<parser::ChangeTeamConstruct>
                  &construct) {
            return std::get<parser::Statement<parser::ChangeTeamStmt>>(
                construct.value().t)
                .source;
          },
          [](const common::Indirection<parser::CriticalConstruct> &construct) {
            return std::get<parser::Statement<parser::CriticalStmt>>(
                construct.value().t)
                .source;
          },
          [](const parser::Statement<parser::ActionStmt> &actionStmt) {
            return actionStmt.source;
          },
          [](const auto &) { return parser::CharBlock{}; },
      },
      executableConstruct.u);
}

bool HasCoarray(const parser::Expr &expression) {
  if (const auto *expr{GetExpr(expression)}) {
    for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
      if (const Symbol * root{GetAssociationRoot(symbol)}) {
        if (IsCoarray(*root)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool IsPolymorphic(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    return type->IsPolymorphic();
  }
  return false;
}

bool IsPolymorphicAllocatable(const Symbol &symbol) {
  return IsAllocatable(symbol) && IsPolymorphic(symbol);
}

static const DeclTypeSpec &InstantiateIntrinsicType(Scope &scope,
    const DeclTypeSpec &spec, SemanticsContext &semanticsContext) {
  const IntrinsicTypeSpec *intrinsic{spec.AsIntrinsic()};
  CHECK(intrinsic);
  if (evaluate::ToInt64(intrinsic->kind())) {
    return spec;  // KIND is already a known constant
  }
  // The expression was not originally constant, but now it must be so
  // in the context of a parameterized derived type instantiation.
  KindExpr copy{intrinsic->kind()};
  evaluate::FoldingContext &foldingContext{semanticsContext.foldingContext()};
  copy = evaluate::Fold(foldingContext, std::move(copy));
  int kind{semanticsContext.GetDefaultKind(intrinsic->category())};
  if (auto value{evaluate::ToInt64(copy)}) {
    if (evaluate::IsValidKindOfIntrinsicType(intrinsic->category(), *value)) {
      kind = *value;
    } else {
      foldingContext.messages().Say(
          "KIND parameter value (%jd) of intrinsic type %s "
          "did not resolve to a supported value"_err_en_US,
          static_cast<std::intmax_t>(*value),
          parser::ToUpperCaseLetters(
              common::EnumToString(intrinsic->category())));
    }
  }
  switch (spec.category()) {
  case DeclTypeSpec::Numeric:
    return scope.MakeNumericType(intrinsic->category(), KindExpr{kind});
  case DeclTypeSpec::Logical:  //
    return scope.MakeLogicalType(KindExpr{kind});
  case DeclTypeSpec::Character:
    return scope.MakeCharacterType(
        ParamValue{spec.characterTypeSpec().length()}, KindExpr{kind});
  default: CRASH_NO_CASE;
  }
}

static const DeclTypeSpec *FindInstantiatedDerivedType(const Scope &scope,
    const DerivedTypeSpec &spec, DeclTypeSpec::Category category) {
  DeclTypeSpec type{category, spec};
  if (const auto *found{scope.FindType(type)}) {
    return found;
  } else if (scope.IsGlobal()) {
    return nullptr;
  } else {
    return FindInstantiatedDerivedType(scope.parent(), spec, category);
  }
}

static Symbol &InstantiateSymbol(const Symbol &, Scope &, SemanticsContext &);

std::list<SourceName> OrderParameterNames(const Symbol &typeSymbol) {
  std::list<SourceName> result;
  if (const DerivedTypeSpec * spec{typeSymbol.GetParentTypeSpec()}) {
    result = OrderParameterNames(spec->typeSymbol());
  }
  const auto &paramNames{typeSymbol.get<DerivedTypeDetails>().paramNames()};
  result.insert(result.end(), paramNames.begin(), paramNames.end());
  return result;
}

SymbolVector OrderParameterDeclarations(const Symbol &typeSymbol) {
  SymbolVector result;
  if (const DerivedTypeSpec * spec{typeSymbol.GetParentTypeSpec()}) {
    result = OrderParameterDeclarations(spec->typeSymbol());
  }
  const auto &paramDecls{typeSymbol.get<DerivedTypeDetails>().paramDecls()};
  result.insert(result.end(), paramDecls.begin(), paramDecls.end());
  return result;
}

void InstantiateDerivedType(DerivedTypeSpec &spec, Scope &containingScope,
    SemanticsContext &semanticsContext) {
  Scope &newScope{containingScope.MakeScope(Scope::Kind::DerivedType)};
  newScope.set_derivedTypeSpec(spec);
  spec.ReplaceScope(newScope);
  const Symbol &typeSymbol{spec.typeSymbol()};
  const Scope *typeScope{typeSymbol.scope()};
  CHECK(typeScope);
  for (const Symbol &symbol : OrderParameterDeclarations(typeSymbol)) {
    const SourceName &name{symbol.name()};
    if (typeScope->find(symbol.name()) != typeScope->end()) {
      // This type parameter belongs to the derived type itself, not to
      // one of its parents.  Put the type parameter expression value
      // into the new scope as the initialization value for the parameter.
      if (ParamValue * paramValue{spec.FindParameter(name)}) {
        const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
        paramValue->set_attr(details.attr());
        if (MaybeIntExpr expr{paramValue->GetExplicit()}) {
          // Ensure that any kind type parameters with values are
          // constant by now.
          if (details.attr() == common::TypeParamAttr::Kind) {
            // Any errors in rank and type will have already elicited
            // messages, so don't pile on by complaining further here.
            if (auto maybeDynamicType{expr->GetType()}) {
              if (expr->Rank() == 0 &&
                  maybeDynamicType->category() == TypeCategory::Integer) {
                if (!evaluate::ToInt64(*expr)) {
                  std::stringstream fortran;
                  fortran << *expr;
                  if (auto *msg{
                          semanticsContext.foldingContext().messages().Say(
                              "Value of kind type parameter '%s' (%s) is not "
                              "a scalar INTEGER constant"_err_en_US,
                              name, fortran.str())}) {
                    msg->Attach(name, "declared here"_en_US);
                  }
                }
              }
            }
          }
          TypeParamDetails instanceDetails{details.attr()};
          if (const DeclTypeSpec * type{details.type()}) {
            instanceDetails.set_type(*type);
          }
          instanceDetails.set_init(std::move(*expr));
          newScope.try_emplace(name, std::move(instanceDetails));
        }
      }
    }
  }
  // Instantiate every non-parameter symbol from the original derived
  // type's scope into the new instance.
  auto restorer{semanticsContext.foldingContext().WithPDTInstance(spec)};
  newScope.AddSourceRange(typeScope->sourceRange());
  for (const auto &pair : *typeScope) {
    const Symbol &symbol{*pair.second};
    InstantiateSymbol(symbol, newScope, semanticsContext);
  }
}

void ProcessParameterExpressions(
    DerivedTypeSpec &spec, evaluate::FoldingContext &foldingContext) {
  auto paramDecls{OrderParameterDeclarations(spec.typeSymbol())};
  // Fold the explicit type parameter value expressions first.  Do not
  // fold them within the scope of the derived type being instantiated;
  // these expressions cannot use its type parameters.  Convert the values
  // of the expressions to the declared types of the type parameters.
  for (const Symbol &symbol : paramDecls) {
    const SourceName &name{symbol.name()};
    if (ParamValue * paramValue{spec.FindParameter(name)}) {
      if (const MaybeIntExpr & expr{paramValue->GetExplicit()}) {
        if (auto converted{evaluate::ConvertToType(symbol, SomeExpr{*expr})}) {
          SomeExpr folded{
              evaluate::Fold(foldingContext, std::move(*converted))};
          if (auto *intExpr{std::get_if<SomeIntExpr>(&folded.u)}) {
            paramValue->SetExplicit(std::move(*intExpr));
            continue;
          }
        }
        std::stringstream fortran;
        fortran << *expr;
        if (auto *msg{foldingContext.messages().Say(
                "Value of type parameter '%s' (%s) is not "
                "convertible to its type"_err_en_US,
                name, fortran.str())}) {
          msg->Attach(name, "declared here"_en_US);
        }
      }
    }
  }
  // Type parameter default value expressions are folded in declaration order
  // within the scope of the derived type so that the values of earlier type
  // parameters are available for use in the default initialization
  // expressions of later parameters.
  auto restorer{foldingContext.WithPDTInstance(spec)};
  for (const Symbol &symbol : paramDecls) {
    const SourceName &name{symbol.name()};
    const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
    MaybeIntExpr expr;
    ParamValue *paramValue{spec.FindParameter(name)};
    if (!paramValue) {
      expr = evaluate::Fold(foldingContext, common::Clone(details.init()));
    } else if (paramValue->isExplicit()) {
      expr = paramValue->GetExplicit();
    }
    if (expr) {
      if (paramValue) {
        paramValue->SetExplicit(std::move(*expr));
      } else {
        spec.AddParamValue(
            symbol.name(), ParamValue{std::move(*expr), details.attr()});
      }
    }
  }
}

const DeclTypeSpec &FindOrInstantiateDerivedType(Scope &scope,
    DerivedTypeSpec &&spec, SemanticsContext &semanticsContext,
    DeclTypeSpec::Category category) {
  ProcessParameterExpressions(spec, semanticsContext.foldingContext());
  if (const DeclTypeSpec *
      type{FindInstantiatedDerivedType(scope, spec, category)}) {
    return *type;
  }
  // Create a new instantiation of this parameterized derived type
  // for this particular distinct set of actual parameter values.
  DeclTypeSpec &type{scope.MakeDerivedType(category, std::move(spec))};
  InstantiateDerivedType(type.derivedTypeSpec(), scope, semanticsContext);
  return type;
}

// Clone a Symbol in the context of a parameterized derived type instance
static Symbol &InstantiateSymbol(
    const Symbol &symbol, Scope &scope, SemanticsContext &semanticsContext) {
  evaluate::FoldingContext foldingContext{semanticsContext.foldingContext()};
  const DerivedTypeSpec &instanceSpec{DEREF(foldingContext.pdtInstance())};
  auto pair{scope.try_emplace(symbol.name(), symbol.attrs())};
  Symbol &result{*pair.first->second};
  if (!pair.second) {
    // Symbol was already present in the scope, which can only happen
    // in the case of type parameters.
    CHECK(symbol.has<TypeParamDetails>());
    return result;
  }
  result.attrs() = symbol.attrs();
  result.flags() = symbol.flags();
  result.set_details(common::Clone(symbol.details()));
  if (auto *details{result.detailsIf<ObjectEntityDetails>()}) {
    if (DeclTypeSpec * origType{result.GetType()}) {
      if (const DerivedTypeSpec * derived{origType->AsDerived()}) {
        DerivedTypeSpec newSpec{*derived};
        if (symbol.test(Symbol::Flag::ParentComp)) {
          // Forward any explicit type parameter values from the
          // derived type spec under instantiation to its parent
          // component derived type spec that define type parameters
          // of the parent component.
          for (const auto &pair : instanceSpec.parameters()) {
            if (scope.find(pair.first) == scope.end()) {
              newSpec.AddParamValue(pair.first, ParamValue{pair.second});
            }
          }
        }
        details->ReplaceType(FindOrInstantiateDerivedType(
            scope, std::move(newSpec), semanticsContext, origType->category()));
      } else if (origType->AsIntrinsic()) {
        details->ReplaceType(
            InstantiateIntrinsicType(scope, *origType, semanticsContext));
      } else if (origType->category() != DeclTypeSpec::ClassStar) {
        DIE("instantiated component has type that is "
            "neither intrinsic, derived, nor CLASS(*)");
      }
    }
    details->set_init(
        evaluate::Fold(foldingContext, std::move(details->init())));
    for (ShapeSpec &dim : details->shape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(
            Fold(foldingContext, std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(
            Fold(foldingContext, std::move(dim.ubound().GetExplicit())));
      }
    }
    for (ShapeSpec &dim : details->coshape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(
            Fold(foldingContext, std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(
            Fold(foldingContext, std::move(dim.ubound().GetExplicit())));
      }
    }
  }
  return result;
}

// ComponentIterator implementation

template<ComponentKind componentKind>
typename ComponentIterator<componentKind>::const_iterator
ComponentIterator<componentKind>::const_iterator::Create(
    const DerivedTypeSpec &derived) {
  const_iterator it{};
  it.componentPath_.emplace_back(derived);
  it.Increment();  // cue up first relevant component, if any
  return it;
}

template<ComponentKind componentKind>
const DerivedTypeSpec *
ComponentIterator<componentKind>::const_iterator::PlanComponentTraversal(
    const Symbol &component) const {
  if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
    if (const DeclTypeSpec * type{details->type()}) {
      if (const auto *derived{type->AsDerived()}) {
        bool traverse{false};
        if constexpr (componentKind == ComponentKind::Ordered) {
          // Order Component (only visit parents)
          traverse = component.test(Symbol::Flag::ParentComp);
        } else if constexpr (componentKind == ComponentKind::Direct) {
          traverse = !IsAllocatableOrPointer(component);
        } else if constexpr (componentKind == ComponentKind::Ultimate) {
          traverse = !IsAllocatableOrPointer(component);
        } else if constexpr (componentKind == ComponentKind::Potential) {
          traverse = !IsPointer(component);
        } else if constexpr (componentKind == ComponentKind::Scope) {
          traverse = !IsAllocatableOrPointer(component);
        }
        if (traverse) {
          const Symbol &newTypeSymbol{derived->typeSymbol()};
          // Avoid infinite loop if the type is already part of the types
          // being visited. It is possible to have "loops in type" because
          // C744 does not forbid to use not yet declared type for
          // ALLOCATABLE or POINTER components.
          for (const auto &node : componentPath_) {
            if (&newTypeSymbol == &node.GetTypeSymbol()) {
              return nullptr;
            }
          }
          return derived;
        }
      }
    }  // intrinsic & unlimited polymorphic not traversable
  }
  return nullptr;
}

template<ComponentKind componentKind>
static bool StopAtComponentPre(const Symbol &component) {
  if constexpr (componentKind == ComponentKind::Ordered) {
    // Parent components need to be iterated upon after their
    // sub-components in structure constructor analysis.
    return !component.test(Symbol::Flag::ParentComp);
  } else if constexpr (componentKind == ComponentKind::Direct) {
    return true;
  } else if constexpr (componentKind == ComponentKind::Ultimate) {
    return component.has<ProcEntityDetails>() ||
        IsAllocatableOrPointer(component) ||
        (component.get<ObjectEntityDetails>().type() &&
            component.get<ObjectEntityDetails>().type()->AsIntrinsic());
  } else if constexpr (componentKind == ComponentKind::Potential) {
    return !IsPointer(component);
  }
}

template<ComponentKind componentKind>
static bool StopAtComponentPost(const Symbol &component) {
  return componentKind == ComponentKind::Ordered &&
      component.test(Symbol::Flag::ParentComp);
}

template<ComponentKind componentKind>
void ComponentIterator<componentKind>::const_iterator::Increment() {
  while (!componentPath_.empty()) {
    ComponentPathNode &deepest{componentPath_.back()};
    if (deepest.component()) {
      if (!deepest.descended()) {
        deepest.set_descended(true);
        if (const DerivedTypeSpec *
            derived{PlanComponentTraversal(*deepest.component())}) {
          componentPath_.emplace_back(*derived);
          continue;
        }
      } else if (!deepest.visited()) {
        deepest.set_visited(true);
        return;  // this is the next component to visit, after descending
      }
    }
    auto &nameIterator{deepest.nameIterator()};
    if (nameIterator == deepest.nameEnd()) {
      componentPath_.pop_back();
    } else if constexpr (componentKind == ComponentKind::Scope) {
      deepest.set_component(*nameIterator++->second);
      deepest.set_descended(false);
      deepest.set_visited(true);
      return;  // this is the next component to visit, before descending
    } else {
      const Scope &scope{deepest.GetScope()};
      auto scopeIter{scope.find(*nameIterator++)};
      if (scopeIter != scope.cend()) {
        const Symbol &component{*scopeIter->second};
        deepest.set_component(component);
        deepest.set_descended(false);
        if (StopAtComponentPre<componentKind>(component)) {
          deepest.set_visited(true);
          return;  // this is the next component to visit, before descending
        } else {
          deepest.set_visited(!StopAtComponentPost<componentKind>(component));
        }
      }
    }
  }
}

template<ComponentKind componentKind>
std::string
ComponentIterator<componentKind>::const_iterator::BuildResultDesignatorName()
    const {
  std::string designator{""};
  for (const auto &node : componentPath_) {
    designator += "%" + DEREF(node.component()).name().ToString();
  }
  return designator;
}

template class ComponentIterator<ComponentKind::Ordered>;
template class ComponentIterator<ComponentKind::Direct>;
template class ComponentIterator<ComponentKind::Ultimate>;
template class ComponentIterator<ComponentKind::Potential>;
template class ComponentIterator<ComponentKind::Scope>;

UltimateComponentIterator::const_iterator FindCoarrayUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsCoarray);
}

UltimateComponentIterator::const_iterator FindPointerUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsPointer);
}

PotentialComponentIterator::const_iterator FindEventOrLockPotentialComponent(
    const DerivedTypeSpec &derived) {
  PotentialComponentIterator potentials{derived};
  return std::find_if(
      potentials.begin(), potentials.end(), [](const Symbol &component) {
        if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
          const DeclTypeSpec *type{details->type()};
          return type && IsEventTypeOrLockType(type->AsDerived());
        }
        return false;
      });
}

UltimateComponentIterator::const_iterator FindAllocatableUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsAllocatable);
}

UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableUltimateComponent(const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(
      ultimates.begin(), ultimates.end(), IsPolymorphicAllocatable);
}

UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableNonCoarrayUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), [](const Symbol &x) {
    return IsPolymorphicAllocatable(x) && !IsCoarray(x);
  });
}

const Symbol *FindUltimateComponent(const DerivedTypeSpec &derived,
    const std::function<bool(const Symbol &)> &predicate) {
  UltimateComponentIterator ultimates{derived};
  if (auto it{std::find_if(ultimates.begin(), ultimates.end(),
          [&predicate](const Symbol &component) -> bool {
            return predicate(component);
          })}) {
    return &*it;
  }
  return nullptr;
}

const Symbol *FindUltimateComponent(const Symbol &symbol,
    const std::function<bool(const Symbol &)> &predicate) {
  if (predicate(symbol)) {
    return &symbol;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (const auto *type{object->type()}) {
      if (const auto *derived{type->AsDerived()}) {
        return FindUltimateComponent(*derived, predicate);
      }
    }
  }
  return nullptr;
}

const Symbol *FindImmediateComponent(const DerivedTypeSpec &type,
    const std::function<bool(const Symbol &)> &predicate) {
  if (const Scope * scope{type.scope()}) {
    const Symbol *parent{nullptr};
    for (const auto &pair : *scope) {
      const Symbol *symbol{&*pair.second};
      if (predicate(*symbol)) {
        return symbol;
      }
      if (symbol->test(Symbol::Flag::ParentComp)) {
        parent = symbol;
      }
    }
    if (parent) {
      if (const auto *object{parent->detailsIf<ObjectEntityDetails>()}) {
        if (const auto *type{object->type()}) {
          if (const auto *derived{type->AsDerived()}) {
            return FindImmediateComponent(*derived, predicate);
          }
        }
      }
    }
  }
  return nullptr;
}

bool IsFunctionResult(const Symbol &symbol) {
  return (symbol.has<semantics::ObjectEntityDetails>() &&
             symbol.get<semantics::ObjectEntityDetails>().isFuncResult()) ||
      (symbol.has<semantics::ProcEntityDetails>() &&
          symbol.get<semantics::ProcEntityDetails>().isFuncResult());
}

bool IsFunctionResultWithSameNameAsFunction(const Symbol &symbol) {
  if (IsFunctionResult(symbol)) {
    if (const Symbol * function{symbol.owner().symbol()}) {
      return symbol.name() == function->name();
    }
  }
  return false;
}
}
