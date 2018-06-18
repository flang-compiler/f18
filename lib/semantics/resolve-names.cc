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

#include "resolve-names.h"
#include "attr.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "type.h"
#include "../common/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>
#include <memory>
#include <ostream>
#include <set>
#include <stack>

namespace Fortran::semantics {

using namespace parser::literals;

class MessageHandler;

static GenericSpec MapGenericSpec(const parser::GenericSpec &);

// ImplicitRules maps initial character of identifier to the DeclTypeSpec*
// representing the implicit type; nullptr if none.
class ImplicitRules {
public:
  ImplicitRules(MessageHandler &messages);
  bool isImplicitNoneType() const { return isImplicitNoneType_; }
  bool isImplicitNoneExternal() const { return isImplicitNoneExternal_; }
  void set_isImplicitNoneType(bool x) { isImplicitNoneType_ = x; }
  void set_isImplicitNoneExternal(bool x) { isImplicitNoneExternal_ = x; }
  // Get the implicit type for identifiers starting with ch. May be null.
  const DeclTypeSpec *GetType(char ch) const;
  // Record the implicit type for this range of characters.
  void SetType(const DeclTypeSpec &type, parser::Location lo, parser::Location,
      bool isDefault = false);
  // Apply the default implicit rules (if no IMPLICIT NONE).
  void AddDefaultRules();

private:
  static char Incr(char ch);

  MessageHandler &messages_;
  bool isImplicitNoneType_{false};
  bool isImplicitNoneExternal_{false};
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec> map_;
  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor {
public:
  void BeginAttrs();
  Attrs GetAttrs();
  Attrs EndAttrs();
  void Post(const parser::LanguageBindingSpec &);
  bool Pre(const parser::AccessSpec &);
  bool Pre(const parser::IntentSpec &);

// Simple case: encountering CLASSNAME causes ATTRNAME to be set.
#define HANDLE_ATTR_CLASS(CLASSNAME, ATTRNAME) \
  bool Pre(const parser::CLASSNAME &) { \
    attrs_->set(Attr::ATTRNAME); \
    return false; \
  }
  HANDLE_ATTR_CLASS(PrefixSpec::Elemental, ELEMENTAL)
  HANDLE_ATTR_CLASS(PrefixSpec::Impure, IMPURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Module, MODULE)
  HANDLE_ATTR_CLASS(PrefixSpec::Non_Recursive, NON_RECURSIVE)
  HANDLE_ATTR_CLASS(PrefixSpec::Pure, PURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Recursive, RECURSIVE)
  HANDLE_ATTR_CLASS(TypeAttrSpec::BindC, BIND_C)
  HANDLE_ATTR_CLASS(Abstract, ABSTRACT)
  HANDLE_ATTR_CLASS(Allocatable, ALLOCATABLE)
  HANDLE_ATTR_CLASS(Asynchronous, ASYNCHRONOUS)
  HANDLE_ATTR_CLASS(Contiguous, CONTIGUOUS)
  HANDLE_ATTR_CLASS(External, EXTERNAL)
  HANDLE_ATTR_CLASS(Intrinsic, INTRINSIC)
  HANDLE_ATTR_CLASS(NoPass, NOPASS)
  HANDLE_ATTR_CLASS(Optional, OPTIONAL)
  HANDLE_ATTR_CLASS(Parameter, PARAMETER)
  HANDLE_ATTR_CLASS(Pass, PASS)
  HANDLE_ATTR_CLASS(Pointer, POINTER)
  HANDLE_ATTR_CLASS(Protected, PROTECTED)
  HANDLE_ATTR_CLASS(Save, SAVE)
  HANDLE_ATTR_CLASS(Target, TARGET)
  HANDLE_ATTR_CLASS(Value, VALUE)
  HANDLE_ATTR_CLASS(Volatile, VOLATILE)
#undef HANDLE_ATTR_CLASS

protected:
  std::optional<Attrs> attrs_;
  std::string langBindingName_{""};

  Attr AccessSpecToAttr(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public: return Attr::PUBLIC;
    case parser::AccessSpec::Kind::Private: return Attr::PRIVATE;
    }
    // unnecessary but g++ warns "control reaches end of non-void function"
    common::die("unreachable");
  }
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  bool Pre(const parser::IntegerTypeSpec &);
  bool Pre(const parser::IntrinsicTypeSpec::Logical &);
  bool Pre(const parser::IntrinsicTypeSpec::Real &);
  bool Pre(const parser::IntrinsicTypeSpec::Complex &);
  bool Pre(const parser::IntrinsicTypeSpec::DoublePrecision &);
  bool Pre(const parser::DeclarationTypeSpec::ClassStar &);
  bool Pre(const parser::DeclarationTypeSpec::TypeStar &);
  void Post(const parser::DeclarationTypeSpec::Type &);
  void Post(const parser::DeclarationTypeSpec::Class &);
  bool Pre(const parser::DeclarationTypeSpec::Record &);
  bool Pre(const parser::DerivedTypeSpec &);
  void Post(const parser::TypeParamSpec &);
  bool Pre(const parser::TypeParamValue &);
  void Post(const parser::StructureConstructor &);
  bool Pre(const parser::AllocateStmt &);
  void Post(const parser::AllocateStmt &);
  bool Pre(const parser::TypeGuardStmt &);
  void Post(const parser::TypeGuardStmt &);

protected:
  std::unique_ptr<DeclTypeSpec> &GetDeclTypeSpec();
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();

  std::unique_ptr<DerivedTypeSpec> derivedTypeSpec_;
  std::unique_ptr<ParamValue> typeParamValue_;

private:
  bool expectDeclTypeSpec_{false};  // should only see decl-type-spec when true
  std::unique_ptr<DeclTypeSpec> declTypeSpec_;
  void MakeIntrinsic(const IntrinsicTypeSpec &intrinsicTypeSpec);
  void SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec);
  static KindParamValue GetKindParamValue(
      const std::optional<parser::KindSelector> &kind);
};

// Track statement source locations and save messages.
class MessageHandler {
public:
  using Message = parser::Message;
  using MessageFixedText = parser::MessageFixedText;
  using MessageFormattedText = parser::MessageFormattedText;

  const parser::Messages &messages() const { return messages_; }

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    currStmtSource_ = &x.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmtSource_ = nullptr;
  }

  const SourceName *currStmtSource() { return currStmtSource_; }

  // Add a message to the messages to be emitted.
  Message &Say(Message &&);
  // Emit a message associated with the current statement source.
  Message &Say(MessageFixedText &&);
  // Emit a message about a SourceName or parser::Name
  Message &Say(const SourceName &, MessageFixedText &&);
  Message &Say(const parser::Name &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  Message &Say(const SourceName &, MessageFixedText &&, const std::string &);
  Message &Say(const SourceName &, MessageFixedText &&, const SourceName &,
      const SourceName &);
  void SayAlreadyDeclared(const SourceName &, const Symbol &);
  // Emit a message and attached message with two names and locations.
  void Say2(const SourceName &, MessageFixedText &&, const SourceName &,
      MessageFixedText &&);

private:
  // Where messages are emitted:
  parser::Messages messages_;
  // Source location of current statement; null if not in a statement
  const SourceName *currStmtSource_{nullptr};
};

// Visit ImplicitStmt and related parse tree nodes and updates implicit rules.
class ImplicitRulesVisitor : public DeclTypeSpecVisitor,
                             public virtual MessageHandler {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;
  using MessageHandler::Post;
  using MessageHandler::Pre;
  using ImplicitNoneNameSpec = parser::ImplicitStmt::ImplicitNoneNameSpec;

  void Post(const parser::ParameterStmt &);
  bool Pre(const parser::ImplicitStmt &);
  bool Pre(const parser::LetterSpec &);
  bool Pre(const parser::ImplicitSpec &);
  void Post(const parser::ImplicitSpec &);

  ImplicitRules &implicitRules() { return implicitRules_.top(); }
  const ImplicitRules &implicitRules() const { return implicitRules_.top(); }
  bool isImplicitNoneType() const {
    return implicitRules().isImplicitNoneType();
  }
  bool isImplicitNoneExternal() const {
    return implicitRules().isImplicitNoneExternal();
  }

protected:
  void PushScope();
  void PopScope();
  void CopyImplicitRules();  // copy from parent into this scope

private:
  // implicit rules in effect for current scope
  std::stack<ImplicitRules, std::list<ImplicitRules>> implicitRules_;
  // previous occurrence of these kinds of statements:
  const SourceName *prevImplicit_{nullptr};
  const SourceName *prevImplicitNone_{nullptr};
  const SourceName *prevImplicitNoneType_{nullptr};
  const SourceName *prevParameterStmt_{nullptr};

  bool HandleImplicitNone(const std::list<ImplicitNoneNameSpec> &nameSpecs);
};

// Track array specifications. They can occur in AttrSpec, EntityDecl,
// ObjectDecl, DimensionStmt, CommonBlockObject, or BasedPointerStmt.
// 1. INTEGER, DIMENSION(10) :: x
// 2. INTEGER :: x(10)
// 3. ALLOCATABLE :: x(:)
// 4. DIMENSION :: x(10)
// 5. TODO: COMMON x(10)
// 6. TODO: BasedPointerStmt
class ArraySpecVisitor {
public:
  bool Pre(const parser::ArraySpec &);
  void Post(const parser::AttrSpec &) { PostAttrSpec(); }
  void Post(const parser::ComponentAttrSpec &) { PostAttrSpec(); }
  bool Pre(const parser::DeferredShapeSpecList &);
  bool Pre(const parser::AssumedShapeSpec &);
  bool Pre(const parser::ExplicitShapeSpec &);
  bool Pre(const parser::AssumedImpliedSpec &);
  bool Pre(const parser::AssumedRankSpec &);

protected:
  const ArraySpec &arraySpec();
  void BeginArraySpec();
  void EndArraySpec();
  void ClearArraySpec() { arraySpec_.clear(); }

private:
  // arraySpec_ is populated by any ArraySpec
  ArraySpec arraySpec_;
  // When an ArraySpec is under an AttrSpec or ComponentAttrSpec, it is moved
  // into attrArraySpec_
  ArraySpec attrArraySpec_;

  void PostAttrSpec();
  Bound GetBound(const parser::SpecificationExpr &);
};

// Manage a stack of Scopes
class ScopeHandler : public virtual ImplicitRulesVisitor {
public:
  ScopeHandler() { PushScope(Scope::globalScope); }
  Scope &CurrScope() { return *scopes_.top(); }
  void PushScope(Scope &scope);
  void PopScope();

  Symbol *FindSymbol(const SourceName &name);
  void EraseSymbol(const SourceName &name);

  // Helpers to make a Symbol in the current scope
  template<typename D>
  Symbol &MakeSymbol(const SourceName &name, const Attrs &attrs, D &&details) {
    auto *symbol = FindSymbol(name);
    if (!symbol) {
      const auto pair = CurrScope().try_emplace(name, attrs, details);
      CHECK(pair.second);  // name was not found, so must be able to add
      return pair.first->second;
    }
    symbol->add_occurrence(name);
    if (symbol->CanReplaceDetails(details)) {
      // update the existing symbol
      symbol->attrs() |= attrs;
      symbol->set_details(details);
      return *symbol;
    } else if (std::is_same<UnknownDetails, D>::value) {
      symbol->attrs() |= attrs;
      return *symbol;
    } else {
      SayAlreadyDeclared(name, *symbol);
      // replace the old symbols with a new one with correct details
      EraseSymbol(symbol->name());
      return MakeSymbol(name, attrs, details);
    }
  }
  template<typename D>
  Symbol &MakeSymbol(
      const parser::Name &name, const Attrs &attrs, D &&details) {
    return MakeSymbol(name.source, attrs, std::move(details));
  }
  template<typename D>
  Symbol &MakeSymbol(const parser::Name &name, D &&details) {
    return MakeSymbol(name, Attrs(), details);
  }
  Symbol &MakeSymbol(const SourceName &name, Attrs attrs = Attrs{}) {
    return MakeSymbol(name, attrs, UnknownDetails());
  }

protected:
  // When subpNamesOnly_ is set we are only collecting procedure names.
  // Create symbols with SubprogramNameDetails of the given kind.
  std::optional<SubprogramKind> subpNamesOnly_;

private:
  // Stack of containing scopes; memory referenced is owned by parent scopes
  std::stack<Scope *, std::list<Scope *>> scopes_;

  // On leaving a scope, add implicit types if appropriate.
  void ApplyImplicitRules();
};

class ModuleVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::Module &);
  void Post(const parser::Module &);
  bool Pre(const parser::AccessStmt &);
  bool Pre(const parser::Only &);
  bool Pre(const parser::Rename::Names &);
  bool Pre(const parser::UseStmt &);
  void Post(const parser::UseStmt &);

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  const SourceName *prevAccessStmt_{nullptr};
  // The scope of the module during a UseStmt
  const Scope *useModuleScope_{nullptr};
  void SetAccess(const parser::Name &, Attr);
  void ApplyDefaultAccess();

  void AddUse(const parser::Rename::Names &);
  void AddUse(const parser::Name &);
  // Record a use from useModuleScope_ of useName as localName. location is
  // where it occurred (either the module or the rename) for error reporting.
  void AddUse(const SourceName &location, const SourceName &localName,
      const SourceName &useName);
};

class InterfaceVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::InterfaceStmt &);
  void Post(const parser::InterfaceStmt &);
  void Post(const parser::EndInterfaceStmt &);
  bool Pre(const parser::GenericSpec &);
  bool Pre(const parser::TypeBoundGenericStmt &);
  void Post(const parser::TypeBoundGenericStmt &);
  bool Pre(const parser::ProcedureStmt &);
  void Post(const parser::GenericStmt &);

  bool inInterfaceBlock() const { return inInterfaceBlock_; }
  bool isGeneric() const { return genericSymbol_ != nullptr; }
  bool isAbstract() const { return isAbstract_; }

protected:
  // Add name or symbol to the generic we are currently processing
  void AddToGeneric(const parser::Name &name, bool expectModuleProc = false);
  void AddToGeneric(const Symbol &symbol);
  // Add to generic the symbol for the subprogram with the same name
  void SetSpecificInGeneric(Symbol &&symbol);

private:
  bool inInterfaceBlock_{false};  // set when in interface block
  bool isAbstract_{false};  // set when in abstract interface block
  Symbol *genericSymbol_{nullptr};  // set when in generic interface block
};

class SubprogramVisitor : public InterfaceVisitor {
public:
  bool Pre(const parser::StmtFunctionStmt &);
  void Post(const parser::StmtFunctionStmt &);
  void Post(const parser::SubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  bool Pre(const parser::SubroutineSubprogram &);
  void Post(const parser::SubroutineSubprogram &);
  bool Pre(const parser::FunctionSubprogram &);
  void Post(const parser::FunctionSubprogram &);
  bool Pre(const parser::InterfaceBody::Subroutine &);
  void Post(const parser::InterfaceBody::Subroutine &);
  bool Pre(const parser::InterfaceBody::Function &);
  void Post(const parser::InterfaceBody::Function &);
  bool Pre(const parser::Suffix &);

protected:
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

private:
  // Function result name from parser::Suffix, if any.
  const parser::Name *funcResultName_{nullptr};

  bool BeginSubprogram(const parser::Name &, Symbol::Flag,
      const std::optional<parser::InternalSubprogramPart> &);
  void EndSubprogram();
  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &, Symbol::Flag);
  Symbol *GetSpecificFromGeneric(const parser::Name &);
};

class DeclarationVisitor : public ArraySpecVisitor,
                           public virtual ScopeHandler {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;

  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
  bool Pre(const parser::AsynchronousStmt &);
  bool Pre(const parser::ContiguousStmt &);
  bool Pre(const parser::ExternalStmt &);
  bool Pre(const parser::IntrinsicStmt &);
  bool Pre(const parser::OptionalStmt &);
  bool Pre(const parser::ProtectedStmt &);
  bool Pre(const parser::ValueStmt &);
  bool Pre(const parser::VolatileStmt &);
  bool Pre(const parser::AllocatableStmt &) {
    objectDeclAttr_ = Attr::ALLOCATABLE;
    return true;
  }
  void Post(const parser::AllocatableStmt &) { objectDeclAttr_ = std::nullopt; }
  bool Pre(const parser::TargetStmt &x) {
    objectDeclAttr_ = Attr::TARGET;
    return true;
  }
  void Post(const parser::TargetStmt &) { objectDeclAttr_ = std::nullopt; }
  void Post(const parser::DimensionStmt::Declaration &);
  bool Pre(const parser::TypeDeclarationStmt &) { return BeginDecl(); }
  void Post(const parser::TypeDeclarationStmt &) { EndDecl(); }
  bool Pre(const parser::DerivedTypeDef &x);
  void Post(const parser::DerivedTypeDef &x);
  bool Pre(const parser::DerivedTypeStmt &x);
  void Post(const parser::DerivedTypeStmt &x);
  bool Pre(const parser::TypeAttrSpec::Extends &x);
  bool Pre(const parser::PrivateStmt &x);
  bool Pre(const parser::SequenceStmt &x);
  bool Pre(const parser::ComponentDefStmt &) { return BeginDecl(); }
  void Post(const parser::ComponentDefStmt &) { EndDecl(); }
  void Post(const parser::ComponentDecl &x);
  bool Pre(const parser::ProcedureDeclarationStmt &);
  void Post(const parser::ProcedureDeclarationStmt &);
  bool Pre(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcInterface &x);
  void Post(const parser::ProcDecl &x);
  bool Pre(const parser::FinalProcedureStmt &x);

protected:
  bool BeginDecl();
  void EndDecl();

private:
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // In a DerivedTypeDef, this is data collected for it
  std::unique_ptr<DerivedTypeDef::Data> derivedTypeData_;
  // In a ProcedureDeclarationStmt or ProcComponentDefStmt, this is
  // the interface name, if any.
  const SourceName *interfaceName_{nullptr};

  // Handle a statement that sets an attribute on a list of names.
  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);
  void DeclareObjectEntity(const parser::Name &, Attrs);
  void DeclareProcEntity(const parser::Name &, Attrs, ProcInterface &&);

  // Set the type of an entity or report an error.
  void SetType(
      const SourceName &name, Symbol &symbol, const DeclTypeSpec &type);

  // Declare an object or procedure entity.
  template<typename T>
  Symbol &DeclareEntity(const parser::Name &name, Attrs attrs) {
    Symbol &symbol{MakeSymbol(name.source, attrs)};
    if (symbol.has<UnknownDetails>()) {
      symbol.set_details(T{});
    } else if (auto *details = symbol.detailsIf<EntityDetails>()) {
      if (!std::is_same<EntityDetails, T>::value) {
        symbol.set_details(T(*details));
      }
    }
    if (T *details = symbol.detailsIf<T>()) {
      // OK
    } else if (std::is_same<EntityDetails, T>::value &&
        (symbol.has<ObjectEntityDetails>() ||
            symbol.has<ProcEntityDetails>())) {
      // OK
    } else if (UseDetails *details = symbol.detailsIf<UseDetails>()) {
      Say(name.source,
          "'%s' is use-associated from module '%s' and cannot be re-declared"_err_en_US,
          name.source, details->module().name());
    } else if (auto *details = symbol.detailsIf<SubprogramNameDetails>()) {
      if (details->kind() == SubprogramKind::Module) {
        Say2(name.source,
            "Declaration of '%s' conflicts with its use as module procedure"_err_en_US,
            symbol.name(), "Module procedure definition"_en_US);
      } else if (details->kind() == SubprogramKind::Internal) {
        Say2(name.source,
            "Declaration of '%s' conflicts with its use as internal procedure"_err_en_US,
            symbol.name(), "Internal procedure definition"_en_US);
      } else {
        CHECK(!"unexpected kind");
      }
    } else {
      SayAlreadyDeclared(name.source, symbol);
    }
    return symbol;
  }
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public ModuleVisitor,
                            public SubprogramVisitor,
                            public DeclarationVisitor {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;
  using DeclarationVisitor::Post;
  using DeclarationVisitor::Pre;
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;
  using InterfaceVisitor::Post;
  using InterfaceVisitor::Pre;
  using ModuleVisitor::Post;
  using ModuleVisitor::Pre;
  using SubprogramVisitor::Post;
  using SubprogramVisitor::Pre;

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  bool Pre(const parser::CommonBlockObject &);
  void Post(const parser::CommonBlockObject &);
  bool Pre(const parser::TypeParamDefStmt &);
  void Post(const parser::TypeParamDefStmt &);
  bool Pre(const parser::TypeDeclarationStmt &) { return BeginDecl(); }
  void Post(const parser::TypeDeclarationStmt &) { EndDecl(); }
  void Post(const parser::ComponentDecl &);
  bool Pre(const parser::PrefixSpec &);
  void Post(const parser::SpecificationPart &);
  bool Pre(const parser::MainProgram &);
  void Post(const parser::EndProgramStmt &);
  void Post(const parser::Program &);

  void Post(const parser::Expr &x) { CheckImplicitSymbol(GetVariableName(x)); }
  void Post(const parser::Variable &x) {
    CheckImplicitSymbol(GetVariableName(x));
  }

  void Post(const parser::ProcedureDesignator &);
  bool Pre(const parser::FunctionReference &);
  void Post(const parser::FunctionReference &);
  bool Pre(const parser::CallStmt &);
  void Post(const parser::CallStmt &);

private:
  // Kind of procedure we are expecting to see in a ProcedureDesignator
  std::optional<Symbol::Flag> expectedProcFlag_;

  const parser::Name *GetVariableName(const parser::DataRef &);
  const parser::Name *GetVariableName(const parser::Designator &);
  const parser::Name *GetVariableName(const parser::Expr &);
  const parser::Name *GetVariableName(const parser::Variable &);
  void CheckImplicitSymbol(const parser::Name *);
  bool CheckUseError(const SourceName &, const Symbol &);
};

// ImplicitRules implementation

ImplicitRules::ImplicitRules(MessageHandler &messages) : messages_{messages} {}

const DeclTypeSpec *ImplicitRules::GetType(char ch) const {
  auto it = map_.find(ch);
  return it != map_.end() ? &it->second : nullptr;
}

// isDefault is set when we are applying the default rules, so it is not
// an error if the type is already set.
void ImplicitRules::SetType(const DeclTypeSpec &type, parser::Location lo,
    parser::Location hi, bool isDefault) {
  for (char ch = *lo; ch; ch = ImplicitRules::Incr(ch)) {
    auto res = map_.emplace(ch, type);
    if (!res.second && !isDefault) {
      messages_.Say(lo,
          "More than one implicit type specified for '%s'"_err_en_US,
          std::string(1, ch));
    }
    if (ch == *hi) {
      break;
    }
  }
}

void ImplicitRules::AddDefaultRules() {
  SetType(DeclTypeSpec::MakeIntrinsic(IntegerTypeSpec::Make()), "i", "n", true);
  SetType(DeclTypeSpec::MakeIntrinsic(RealTypeSpec::Make()), "a", "z", true);
}

// Return the next char after ch in a way that works for ASCII or EBCDIC.
// Return '\0' for the char after 'z'.
char ImplicitRules::Incr(char ch) {
  switch (ch) {
  case 'i': return 'j';
  case 'r': return 's';
  case 'z': return '\0';
  default: return ch + 1;
  }
}

std::ostream &operator<<(std::ostream &o, const ImplicitRules &implicitRules) {
  o << "ImplicitRules:\n";
  for (char ch = 'a'; ch; ch = ImplicitRules::Incr(ch)) {
    ShowImplicitRule(o, implicitRules, ch);
  }
  ShowImplicitRule(o, implicitRules, '_');
  ShowImplicitRule(o, implicitRules, '$');
  ShowImplicitRule(o, implicitRules, '@');
  return o;
}
void ShowImplicitRule(
    std::ostream &o, const ImplicitRules &implicitRules, char ch) {
  auto it = implicitRules.map_.find(ch);
  if (it != implicitRules.map_.end()) {
    o << "  " << ch << ": " << it->second << '\n';
  }
}

// AttrsVisitor implementation

void AttrsVisitor::BeginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_optional<Attrs>();
}
Attrs AttrsVisitor::GetAttrs() {
  CHECK(attrs_);
  return *attrs_;
}
Attrs AttrsVisitor::EndAttrs() {
  CHECK(attrs_);
  Attrs result{*attrs_};
  attrs_.reset();
  return result;
}
void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  attrs_->set(Attr::BIND_C);
  if (x.v) {
    // TODO: set langBindingName_ from ScalarDefaultCharConstantExpr
  }
}
bool AttrsVisitor::Pre(const parser::AccessSpec &x) {
  attrs_->set(AccessSpecToAttr(x));
  return false;
}
bool AttrsVisitor::Pre(const parser::IntentSpec &x) {
  switch (x.v) {
  case parser::IntentSpec::Intent::In: attrs_->set(Attr::INTENT_IN); break;
  case parser::IntentSpec::Intent::Out: attrs_->set(Attr::INTENT_OUT); break;
  case parser::IntentSpec::Intent::InOut:
    attrs_->set(Attr::INTENT_IN);
    attrs_->set(Attr::INTENT_OUT);
    break;
  }
  return false;
}

// DeclTypeSpecVisitor implementation

std::unique_ptr<DeclTypeSpec> &DeclTypeSpecVisitor::GetDeclTypeSpec() {
  return declTypeSpec_;
}
void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!expectDeclTypeSpec_);
  expectDeclTypeSpec_ = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(expectDeclTypeSpec_);
  expectDeclTypeSpec_ = false;
  declTypeSpec_.reset();
}

bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::ClassStar &x) {
  SetDeclTypeSpec(DeclTypeSpec::MakeClassStar());
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::TypeStar &x) {
  SetDeclTypeSpec(DeclTypeSpec::MakeTypeStar());
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::DerivedTypeSpec &x) {
  CHECK(!derivedTypeSpec_);
  derivedTypeSpec_ =
      std::make_unique<DerivedTypeSpec>(std::get<parser::Name>(x.t).ToString());
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeParamSpec &x) {
  if (const auto &keyword = std::get<std::optional<parser::Keyword>>(x.t)) {
    derivedTypeSpec_->AddParamValue(keyword->v.ToString(), *typeParamValue_);
  } else {
    derivedTypeSpec_->AddParamValue(*typeParamValue_);
  }
  typeParamValue_.reset();
}
bool DeclTypeSpecVisitor::Pre(const parser::TypeParamValue &x) {
  typeParamValue_ = std::make_unique<ParamValue>(std::visit(
      common::visitors{
          // TODO: create IntExpr from ScalarIntExpr
          [&](const parser::ScalarIntExpr &x) { return Bound{IntExpr{}}; },
          [&](const parser::Star &x) { return Bound::ASSUMED; },
          [&](const parser::TypeParamValue::Deferred &x) {
            return Bound::DEFERRED;
          },
      },
      x.u));
  return false;
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Type &) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeTypeDerivedType(std::move(derivedTypeSpec_)));
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Class &) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeClassDerivedType(std::move(derivedTypeSpec_)));
}
bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::Record &x) {
  // TODO
  return true;
}

void DeclTypeSpecVisitor::Post(const parser::StructureConstructor &) {
  // TODO: StructureConstructor
  derivedTypeSpec_.reset();
}
bool DeclTypeSpecVisitor::Pre(const parser::AllocateStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::AllocateStmt &) {
  // TODO: AllocateStmt
  EndDeclTypeSpec();
  derivedTypeSpec_.reset();
}
bool DeclTypeSpecVisitor::Pre(const parser::TypeGuardStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeGuardStmt &) {
  // TODO: TypeGuardStmt
  EndDeclTypeSpec();
  derivedTypeSpec_.reset();
}

bool DeclTypeSpecVisitor::Pre(const parser::IntegerTypeSpec &x) {
  MakeIntrinsic(IntegerTypeSpec::Make(GetKindParamValue(x.v)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Logical &x) {
  MakeIntrinsic(LogicalTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Real &x) {
  MakeIntrinsic(RealTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Complex &x) {
  MakeIntrinsic(ComplexTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(
    const parser::IntrinsicTypeSpec::DoublePrecision &) {
  CHECK(!"TODO: double precision");
  return false;
}
void DeclTypeSpecVisitor::MakeIntrinsic(
    const IntrinsicTypeSpec &intrinsicTypeSpec) {
  SetDeclTypeSpec(DeclTypeSpec::MakeIntrinsic(intrinsicTypeSpec));
}
// Check that we're expecting to see a DeclTypeSpec (and haven't seen one yet)
// and save it in declTypeSpec_.
void DeclTypeSpecVisitor::SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec) {
  CHECK(expectDeclTypeSpec_);
  CHECK(!declTypeSpec_);
  declTypeSpec_ = std::make_unique<DeclTypeSpec>(declTypeSpec);
}

KindParamValue DeclTypeSpecVisitor::GetKindParamValue(
    const std::optional<parser::KindSelector> &kind) {
  if (kind) {
    if (auto *intExpr = std::get_if<parser::ScalarIntConstantExpr>(&kind->u)) {
      const parser::Expr &expr{*intExpr->thing.thing.thing};
      if (auto *lit = std::get_if<parser::LiteralConstant>(&expr.u)) {
        if (auto *intLit = std::get_if<parser::IntLiteralConstant>(&lit->u)) {
          return KindParamValue{
              IntConst::Make(std::get<std::uint64_t>(intLit->t))};
        }
      }
      CHECK(!"TODO: constant evaluation");
    } else {
      CHECK(!"TODO: translate star-size to kind");
    }
  }
  return KindParamValue{};
}

// MessageHandler implementation

MessageHandler::Message &MessageHandler::Say(Message &&msg) {
  return messages_.Put(std::move(msg));
}
MessageHandler::Message &MessageHandler::Say(MessageFixedText &&msg) {
  CHECK(currStmtSource_);
  return Say(Message{*currStmtSource_, std::move(msg)});
}
MessageHandler::Message &MessageHandler::Say(
    const SourceName &name, MessageFixedText &&msg) {
  return Say(name, std::move(msg), name.ToString());
}
MessageHandler::Message &MessageHandler::Say(
    const parser::Name &name, MessageFixedText &&msg) {
  return Say(name.source, std::move(msg), name.ToString());
}
MessageHandler::Message &MessageHandler::Say(const SourceName &location,
    MessageFixedText &&msg, const std::string &arg1) {
  return Say(Message{location, MessageFormattedText{msg, arg1.c_str()}});
}
MessageHandler::Message &MessageHandler::Say(const SourceName &location,
    MessageFixedText &&msg, const SourceName &arg1, const SourceName &arg2) {
  return Say(Message{location,
      MessageFormattedText{
          msg, arg1.ToString().c_str(), arg2.ToString().c_str()}});
}
void MessageHandler::SayAlreadyDeclared(
    const SourceName &name, const Symbol &prev) {
  Say2(name, "'%s' is already declared in this scoping unit"_err_en_US,
      prev.name(), "Previous declaration of '%s'"_en_US);
}
void MessageHandler::Say2(const SourceName &name1, MessageFixedText &&msg1,
    const SourceName &name2, MessageFixedText &&msg2) {
  Say(name1, std::move(msg1))
      .Attach(name2, MessageFormattedText{msg2, name2.ToString().data()});
}

// ImplicitRulesVisitor implementation

void ImplicitRulesVisitor::Post(const parser::ParameterStmt &x) {
  prevParameterStmt_ = currStmtSource();
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitStmt &x) {
  bool res = std::visit(
      common::visitors{
          [&](const std::list<ImplicitNoneNameSpec> &x) {
            return HandleImplicitNone(x);
          },
          [&](const std::list<parser::ImplicitSpec> &x) {
            if (prevImplicitNoneType_) {
              Say("IMPLICIT statement after IMPLICIT NONE or "
                  "IMPLICIT NONE(TYPE) statement"_err_en_US);
              return false;
            }
            return true;
          },
      },
      x.u);
  prevImplicit_ = currStmtSource();
  return res;
}

bool ImplicitRulesVisitor::Pre(const parser::LetterSpec &x) {
  auto loLoc = std::get<parser::Location>(x.t);
  auto hiLoc = loLoc;
  if (auto hiLocOpt = std::get<std::optional<parser::Location>>(x.t)) {
    hiLoc = *hiLocOpt;
    if (*hiLoc < *loLoc) {
      Say(hiLoc, "'%s' does not follow '%s' alphabetically"_err_en_US,
          std::string(hiLoc, 1), std::string(loLoc, 1));
      return false;
    }
  }
  implicitRules().SetType(*GetDeclTypeSpec(), loLoc, hiLoc);
  return false;
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitSpec &) {
  BeginDeclTypeSpec();
  return true;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitSpec &) {
  EndDeclTypeSpec();
}

void ImplicitRulesVisitor::PushScope() {
  implicitRules_.push(ImplicitRules(*this));
  prevImplicit_ = nullptr;
  prevImplicitNone_ = nullptr;
  prevImplicitNoneType_ = nullptr;
  prevParameterStmt_ = nullptr;
}

void ImplicitRulesVisitor::CopyImplicitRules() {
  implicitRules_.pop();
  implicitRules_.push(ImplicitRules(implicitRules_.top()));
}

void ImplicitRulesVisitor::PopScope() { implicitRules_.pop(); }

// TODO: for all of these errors, reference previous statement too
bool ImplicitRulesVisitor::HandleImplicitNone(
    const std::list<ImplicitNoneNameSpec> &nameSpecs) {
  if (prevImplicitNone_ != nullptr) {
    Say("More than one IMPLICIT NONE statement"_err_en_US);
    Say(*prevImplicitNone_, "Previous IMPLICIT NONE statement"_en_US);
    return false;
  }
  if (prevParameterStmt_ != nullptr) {
    Say("IMPLICIT NONE statement after PARAMETER statement"_err_en_US);
    return false;
  }
  prevImplicitNone_ = currStmtSource();
  if (nameSpecs.empty()) {
    prevImplicitNoneType_ = currStmtSource();
    implicitRules().set_isImplicitNoneType(true);
    if (prevImplicit_) {
      Say("IMPLICIT NONE statement after IMPLICIT statement"_err_en_US);
      return false;
    }
  } else {
    int sawType{0};
    int sawExternal{0};
    for (const auto noneSpec : nameSpecs) {
      switch (noneSpec) {
      case ImplicitNoneNameSpec::External:
        implicitRules().set_isImplicitNoneExternal(true);
        ++sawExternal;
        break;
      case ImplicitNoneNameSpec::Type:
        prevImplicitNoneType_ = currStmtSource();
        implicitRules().set_isImplicitNoneType(true);
        if (prevImplicit_) {
          Say("IMPLICIT NONE(TYPE) after IMPLICIT statement"_err_en_US);
          return false;
        }
        ++sawType;
        break;
      }
    }
    if (sawType > 1) {
      Say("TYPE specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
    if (sawExternal > 1) {
      Say("EXTERNAL specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
  }
  return true;
}

// ArraySpecVisitor implementation

bool ArraySpecVisitor::Pre(const parser::ArraySpec &x) {
  CHECK(arraySpec_.empty());
  return true;
}

bool ArraySpecVisitor::Pre(const parser::DeferredShapeSpecList &x) {
  for (int i = 0; i < x.v; ++i) {
    arraySpec_.push_back(ShapeSpec::MakeDeferred());
  }
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedShapeSpec &x) {
  const auto &lb = x.v;
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeAssumed(GetBound(*lb)) : ShapeSpec::MakeAssumed());
  return false;
}

bool ArraySpecVisitor::Pre(const parser::ExplicitShapeSpec &x) {
  const auto &lb = std::get<std::optional<parser::SpecificationExpr>>(x.t);
  const auto &ub = GetBound(std::get<parser::SpecificationExpr>(x.t));
  arraySpec_.push_back(lb ? ShapeSpec::MakeExplicit(GetBound(*lb), ub)
                          : ShapeSpec::MakeExplicit(ub));
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedImpliedSpec &x) {
  const auto &lb = x.v;
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeImplied(GetBound(*lb)) : ShapeSpec::MakeImplied());
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedRankSpec &) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
  return false;
}

const ArraySpec &ArraySpecVisitor::arraySpec() {
  return !arraySpec_.empty() ? arraySpec_ : attrArraySpec_;
}
void ArraySpecVisitor::BeginArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(attrArraySpec_.empty());
}
void ArraySpecVisitor::EndArraySpec() {
  CHECK(arraySpec_.empty());
  attrArraySpec_.clear();
}
void ArraySpecVisitor::PostAttrSpec() {
  if (!arraySpec_.empty()) {
    // Example: integer, dimension(<1>) :: x(<2>)
    // This saves <1> in attrArraySpec_ so we can process <2> into arraySpec_
    CHECK(attrArraySpec_.empty());
    attrArraySpec_.splice(attrArraySpec_.cbegin(), arraySpec_);
    CHECK(arraySpec_.empty());
  }
}

Bound ArraySpecVisitor::GetBound(const parser::SpecificationExpr &x) {
  return Bound(IntExpr{});  // TODO: convert x.v to IntExpr
}

// ScopeHandler implementation

void ScopeHandler::PushScope(Scope &scope) {
  scopes_.push(&scope);
  ImplicitRulesVisitor::PushScope();
}
void ScopeHandler::PopScope() {
  ApplyImplicitRules();
  scopes_.pop();
  ImplicitRulesVisitor::PopScope();
}
Symbol *ScopeHandler::FindSymbol(const SourceName &name) {
  const auto &it = CurrScope().find(name);
  if (it == CurrScope().end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}
void ScopeHandler::EraseSymbol(const SourceName &name) {
  CurrScope().erase(name);
}

void ScopeHandler::ApplyImplicitRules() {
  if (!isImplicitNoneType()) {
    implicitRules().AddDefaultRules();
    for (auto &pair : CurrScope()) {
      Symbol &symbol = pair.second;
      if (symbol.has<UnknownDetails>()) {
        symbol.set_details(ObjectEntityDetails{});
      } else if (auto *details = symbol.detailsIf<EntityDetails>()) {
        symbol.set_details(ObjectEntityDetails{*details});
      }
      if (auto *details = symbol.detailsIf<ObjectEntityDetails>()) {
        if (!details->type()) {
          const auto &name = pair.first;
          if (const auto *type = implicitRules().GetType(name.begin()[0])) {
            details->set_type(*type);
          } else {
            Say(name, "No explicit type declared for '%s'"_err_en_US);
          }
        }
      }
    }
  }
}

// ModuleVisitor implementation

bool ModuleVisitor::Pre(const parser::Only &x) {
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::GenericSpec> &generic) {
            std::visit(
                common::visitors{
                    [&](const parser::Name &name) { AddUse(name); },
                    [](const auto &) { common::die("TODO: GenericSpec"); },
                },
                generic->u);
          },
          [&](const parser::Name &name) { AddUse(name); },
          [&](const parser::Rename &rename) {
            std::visit(
                common::visitors{
                    [&](const parser::Rename::Names &names) { AddUse(names); },
                    [&](const parser::Rename::Operators &ops) {
                      common::die("TODO: Rename::Operators");
                    },
                },
                rename.u);
          },
      },
      x.u);
  return false;
}

bool ModuleVisitor::Pre(const parser::Rename::Names &x) {
  AddUse(x);
  return false;
}

// Set useModuleScope_ to the Scope of the module being used.
bool ModuleVisitor::Pre(const parser::UseStmt &x) {
  // x.nature = UseStmt::ModuleNature::Intrinsic or Non_Intrinsic
  const auto it = Scope::globalScope.find(x.moduleName.source);
  if (it == Scope::globalScope.end()) {
    Say(x.moduleName, "Module '%s' not found"_err_en_US);
    return false;
  }
  const auto *details = it->second.detailsIf<ModuleDetails>();
  if (!details) {
    Say(x.moduleName, "'%s' is not a module"_err_en_US);
    return false;
  }
  useModuleScope_ = details->scope();
  CHECK(useModuleScope_);
  return true;
}
void ModuleVisitor::Post(const parser::UseStmt &x) {
  if (const auto *list = std::get_if<std::list<parser::Rename>>(&x.u)) {
    // Not a use-only: collect the names that were used in renames,
    // then add a use for each public name that was not renamed.
    std::set<SourceName> useNames;
    for (const auto &rename : *list) {
      std::visit(
          common::visitors{
              [&](const parser::Rename::Names &names) {
                useNames.insert(std::get<1>(names.t).source);
              },
              [&](const parser::Rename::Operators &ops) {
                CHECK(!"TODO: Rename::Operators");
              },
          },
          rename.u);
    }
    const SourceName &moduleName{x.moduleName.source};
    for (const auto &pair : *useModuleScope_) {
      const Symbol &symbol{pair.second};
      if (symbol.attrs().test(Attr::PUBLIC) &&
          !symbol.detailsIf<ModuleDetails>()) {
        const SourceName &name{symbol.name()};
        if (useNames.count(name) == 0) {
          AddUse(moduleName, name, name);
        }
      }
    }
  }
  useModuleScope_ = nullptr;
}

void ModuleVisitor::AddUse(const parser::Rename::Names &names) {
  const SourceName &useName{std::get<0>(names.t).source};
  const SourceName &localName{std::get<1>(names.t).source};
  AddUse(useName, useName, localName);
}
void ModuleVisitor::AddUse(const parser::Name &useName) {
  AddUse(useName.source, useName.source, useName.source);
}
void ModuleVisitor::AddUse(const SourceName &location,
    const SourceName &localName, const SourceName &useName) {
  if (!useModuleScope_) {
    return;  // error occurred finding module
  }
  const auto it = useModuleScope_->find(useName);
  if (it == useModuleScope_->end()) {
    Say(useName, "'%s' not found in module '%s'"_err_en_US, useName,
        useModuleScope_->name());
    return;
  }
  const Symbol &useSymbol{it->second};
  if (useSymbol.attrs().test(Attr::PRIVATE)) {
    Say(useName, "'%s' is PRIVATE in '%s'"_err_en_US, useName,
        useModuleScope_->name());
    return;
  }
  Symbol &localSymbol{MakeSymbol(localName, useSymbol.attrs())};
  localSymbol.attrs() &= ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
  localSymbol.flags() |= useSymbol.flags();
  if (auto *details = localSymbol.detailsIf<UseDetails>()) {
    // check for importing the same symbol again:
    if (localSymbol.GetUltimate() != useSymbol.GetUltimate()) {
      localSymbol.set_details(
          UseErrorDetails{details->location(), *useModuleScope_});
    }
  } else if (auto *details = localSymbol.detailsIf<UseErrorDetails>()) {
    details->add_occurrence(location, *useModuleScope_);
  } else if (localSymbol.has<UnknownDetails>()) {
    localSymbol.set_details(UseDetails{location, useSymbol});
  } else {
    localSymbol.set_details(
        UseErrorDetails{useSymbol.name(), *useModuleScope_});
  }
}

bool ModuleVisitor::Pre(const parser::Module &x) {
  // Make a symbol and push a scope for this module
  const auto &name =
      std::get<parser::Statement<parser::ModuleStmt>>(x.t).statement.v;
  auto &symbol = MakeSymbol(name, ModuleDetails{});
  ModuleDetails &details{symbol.details<ModuleDetails>()};
  Scope &modScope = CurrScope().MakeScope(Scope::Kind::Module, &symbol);
  details.set_scope(&modScope);
  PushScope(modScope);
  MakeSymbol(name, ModuleDetails{details});
  // collect module subprogram names
  if (const auto &subpPart =
          std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)) {
    subpNamesOnly_ = SubprogramKind::Module;
    parser::Walk(*subpPart, *static_cast<ResolveNamesVisitor *>(this));
    subpNamesOnly_ = std::nullopt;
  }
  return true;
}

void ModuleVisitor::Post(const parser::Module &) {
  ApplyDefaultAccess();
  PopScope();
  prevAccessStmt_ = nullptr;
}

void ModuleVisitor::ApplyDefaultAccess() {
  for (auto &pair : CurrScope()) {
    Symbol &symbol = pair.second;
    if (!symbol.attrs().HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
      symbol.attrs().set(defaultAccess_);
    }
  }
}

// InterfaceVistor implementation

bool InterfaceVisitor::Pre(const parser::InterfaceStmt &x) {
  inInterfaceBlock_ = true;
  isAbstract_ = std::holds_alternative<parser::Abstract>(x.u);
  return true;
}
void InterfaceVisitor::Post(const parser::InterfaceStmt &) {}

void InterfaceVisitor::Post(const parser::EndInterfaceStmt &) {
  if (genericSymbol_) {
    if (const auto *proc =
            genericSymbol_->details<GenericDetails>().CheckSpecific()) {
      SayAlreadyDeclared(genericSymbol_->name(), *proc);
    }
    genericSymbol_ = nullptr;
  }
  inInterfaceBlock_ = false;
  isAbstract_ = false;
}

// Create a symbol for the generic in genericSymbol_
bool InterfaceVisitor::Pre(const parser::GenericSpec &x) {
  const SourceName *genericName{nullptr};
  GenericSpec genericSpec{MapGenericSpec(x)};
  switch (genericSpec.kind()) {
  case GenericSpec::Kind::GENERIC_NAME:
    genericName = &genericSpec.genericName();
    break;
  case GenericSpec::Kind::OP_DEFINED:
    genericName = &genericSpec.definedOp();
    break;
  default: CHECK(!"TODO: intrinsic ops");
  }
  genericSymbol_ = FindSymbol(*genericName);
  if (genericSymbol_) {
    if (!genericSymbol_->isSubprogram()) {
      SayAlreadyDeclared(*genericName, *genericSymbol_);
      EraseSymbol(*genericName);
      genericSymbol_ = nullptr;
    } else if (genericSymbol_->has<UseDetails>()) {
      // copy the USEd symbol into this scope so we can modify it
      const Symbol &ultimate{genericSymbol_->GetUltimate()};
      EraseSymbol(*genericName);
      genericSymbol_ = &MakeSymbol(ultimate.name(), ultimate.attrs());
      if (const auto *details = ultimate.detailsIf<GenericDetails>()) {
        genericSymbol_->set_details(GenericDetails{details->specificProcs()});
      } else if (const auto *details =
                     ultimate.detailsIf<SubprogramDetails>()) {
        genericSymbol_->set_details(SubprogramDetails{*details});
      } else {
        CHECK(!"can't happen");
      }
    }
  }
  if (!genericSymbol_) {
    genericSymbol_ = &MakeSymbol(*genericName);
    genericSymbol_->set_details(GenericDetails{});
  }
  if (genericSymbol_->has<GenericDetails>()) {
    // okay
  } else if (genericSymbol_->has<SubprogramDetails>() ||
      genericSymbol_->has<SubprogramNameDetails>()) {
    Details details;
    if (auto *d = genericSymbol_->detailsIf<SubprogramNameDetails>()) {
      details = *d;
    } else if (auto *d = genericSymbol_->detailsIf<SubprogramDetails>()) {
      details = *d;
    } else {
      CHECK(!"can't happen");
    }
    Symbol symbol{CurrScope(), genericSymbol_->name(), genericSymbol_->attrs(),
        std::move(details)};
    EraseSymbol(*genericName);
    genericSymbol_ = &MakeSymbol(*genericName);
    genericSymbol_->set_details(GenericDetails{std::move(symbol)});
  }
  CHECK(genericSymbol_->has<GenericDetails>());
  return false;
}

bool InterfaceVisitor::Pre(const parser::TypeBoundGenericStmt &) {
  return true;
}
void InterfaceVisitor::Post(const parser::TypeBoundGenericStmt &) {
  // TODO: TypeBoundGenericStmt
}

bool InterfaceVisitor::Pre(const parser::ProcedureStmt &x) {
  if (!isGeneric()) {
    Say("A PROCEDURE statement is only allowed in a generic interface block"_err_en_US);
    return false;
  }
  bool expectModuleProc = std::get<parser::ProcedureStmt::Kind>(x.t) ==
      parser::ProcedureStmt::Kind::ModuleProcedure;
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    AddToGeneric(name, expectModuleProc);
  }
  return false;
}

void InterfaceVisitor::Post(const parser::GenericStmt &x) {
  if (auto &accessSpec = std::get<std::optional<parser::AccessSpec>>(x.t)) {
    genericSymbol_->attrs().set(AccessSpecToAttr(*accessSpec));
  }
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    AddToGeneric(name);
  }
}

void InterfaceVisitor::AddToGeneric(
    const parser::Name &name, bool expectModuleProc) {
  const auto *symbol = FindSymbol(name.source);
  if (!symbol) {
    Say(name, "Procedure '%s' not found"_err_en_US);
    return;
  }
  if (symbol == genericSymbol_) {
    if (auto *specific =
            genericSymbol_->details<GenericDetails>().specific().get()) {
      symbol = specific;
    }
  }
  if (!symbol->has<SubprogramDetails>() &&
      !symbol->has<SubprogramNameDetails>()) {
    Say(name, "'%s' is not a subprogram"_err_en_US);
    return;
  }
  if (expectModuleProc) {
    const auto *details = symbol->detailsIf<SubprogramNameDetails>();
    if (!details || details->kind() != SubprogramKind::Module) {
      Say(name, "'%s' is not a module procedure"_en_US);
    }
  }
  AddToGeneric(*symbol);
}
void InterfaceVisitor::AddToGeneric(const Symbol &symbol) {
  genericSymbol_->details<GenericDetails>().add_specificProc(&symbol);
}
void InterfaceVisitor::SetSpecificInGeneric(Symbol &&symbol) {
  genericSymbol_->details<GenericDetails>().set_specific(std::move(symbol));
}

// SubprogramVisitor implementation

bool SubprogramVisitor::Pre(const parser::StmtFunctionStmt &x) {
  const auto &name = std::get<parser::Name>(x.t);
  std::optional<SourceName> occurrence;
  std::optional<DeclTypeSpec> resultType;
  // Look up name: provides return type or tells us if it's an array
  if (auto *symbol = FindSymbol(name.source)) {
    if (auto *details = symbol->detailsIf<EntityDetails>()) {
      // TODO: check that attrs are compatible with stmt func
      resultType = details->type();
      occurrence = symbol->name();
      EraseSymbol(symbol->name());
    } else if (symbol->has<ObjectEntityDetails>()) {
      // not a stmt-func at all but an array; do nothing
      symbol->add_occurrence(name.source);
      badStmtFuncFound_ = true;
      return true;
    }
  }
  if (badStmtFuncFound_) {
    Say(name, "'%s' has not been declared as an array"_err_en_US);
    return true;
  }
  auto &symbol = PushSubprogramScope(name, Symbol::Flag::Function);
  CopyImplicitRules();
  if (occurrence) {
    symbol.add_occurrence(*occurrence);
  }
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyName : std::get<std::list<parser::Name>>(x.t)) {
    EntityDetails dummyDetails{true};
    auto it = CurrScope().parent().find(dummyName.source);
    if (it != CurrScope().parent().end()) {
      if (auto *d = it->second.detailsIf<EntityDetails>()) {
        if (d->type()) {
          dummyDetails.set_type(*d->type());
        }
      }
    }
    details.add_dummyArg(MakeSymbol(dummyName, std::move(dummyDetails)));
  }
  EraseSymbol(name.source);  // added by PushSubprogramScope
  EntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  details.set_result(MakeSymbol(name, resultDetails));
  return true;
}

void SubprogramVisitor::Post(const parser::StmtFunctionStmt &x) {
  if (badStmtFuncFound_) {
    return;  // This wasn't really a stmt function so no scope was created
  }
  PopScope();
}

bool SubprogramVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName) {
    funcResultName_ = &suffix.resultName.value();
  }
  return true;
}

bool SubprogramVisitor::Pre(const parser::SubroutineSubprogram &x) {
  const auto &name = std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t);
  const auto &subpPart =
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t);
  return BeginSubprogram(name, Symbol::Flag::Subroutine, subpPart);
}
void SubprogramVisitor::Post(const parser::SubroutineSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::FunctionSubprogram &x) {
  const auto &name = std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t);
  const auto &subpPart =
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t);
  return BeginSubprogram(name, Symbol::Flag::Function, subpPart);
}
void SubprogramVisitor::Post(const parser::FunctionSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::InterfaceBody::Subroutine &x) {
  const auto &name = std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t);
  return BeginSubprogram(name, Symbol::Flag::Subroutine, std::nullopt);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Subroutine &) {
  EndSubprogram();
}
bool SubprogramVisitor::Pre(const parser::InterfaceBody::Function &x) {
  const auto &name = std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t);
  return BeginSubprogram(name, Symbol::Flag::Function, std::nullopt);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Function &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::FunctionStmt &stmt) {
  if (!subpNamesOnly_) {
    BeginDeclTypeSpec();
    CHECK(!funcResultName_);
  }
  return true;
}

void SubprogramVisitor::Post(const parser::SubroutineStmt &stmt) {
  const auto &name = std::get<parser::Name>(stmt.t);
  Symbol &symbol{*CurrScope().symbol()};
  CHECK(name.source == symbol.name());
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    const parser::Name *dummyName = std::get_if<parser::Name>(&dummyArg.u);
    CHECK(dummyName != nullptr && "TODO: alternate return indicator");
    Symbol &dummy{MakeSymbol(*dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
}

void SubprogramVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &name = std::get<parser::Name>(stmt.t);
  Symbol &symbol{*CurrScope().symbol()};
  CHECK(name.source == symbol.name());
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
  // add function result to function scope
  EntityDetails funcResultDetails;
  if (auto &type = GetDeclTypeSpec()) {
    funcResultDetails.set_type(*type);
  }
  EndDeclTypeSpec();

  const parser::Name *funcResultName;
  if (funcResultName_ && funcResultName_->source != name.source) {
    funcResultName = funcResultName_;
  } else {
    EraseSymbol(name.source);  // was added by PushSubprogramScope
    funcResultName = &name;
  }
  details.set_result(MakeSymbol(*funcResultName, funcResultDetails));
  funcResultName_ = nullptr;
}

bool SubprogramVisitor::BeginSubprogram(const parser::Name &name,
    Symbol::Flag subpFlag,
    const std::optional<parser::InternalSubprogramPart> &subpPart) {
  if (subpNamesOnly_) {
    auto &symbol = MakeSymbol(name, SubprogramNameDetails{*subpNamesOnly_});
    symbol.set(subpFlag);
    return false;
  }
  PushSubprogramScope(name, subpFlag);
  if (subpPart) {
    subpNamesOnly_ = SubprogramKind::Internal;
    parser::Walk(*subpPart, *static_cast<ResolveNamesVisitor *>(this));
    subpNamesOnly_ = std::nullopt;
  }
  return true;
}
void SubprogramVisitor::EndSubprogram() {
  if (!subpNamesOnly_) {
    PopScope();
  }
}

Symbol &SubprogramVisitor::PushSubprogramScope(
    const parser::Name &name, Symbol::Flag subpFlag) {
  Symbol *symbol = GetSpecificFromGeneric(name);
  if (!symbol) {
    symbol = &MakeSymbol(name, SubprogramDetails{});
    symbol->set(subpFlag);
  }
  auto &details = symbol->details<SubprogramDetails>();
  if (inInterfaceBlock()) {
    details.set_isInterface();
    if (!isAbstract()) {
      symbol->attrs().set(Attr::EXTERNAL);
    }
    if (isGeneric()) {
      AddToGeneric(*symbol);
    }
  }
  Scope &subpScope = CurrScope().MakeScope(Scope::Kind::Subprogram, symbol);
  PushScope(subpScope);
  // can't reuse this name inside subprogram:
  MakeSymbol(name, SubprogramDetails(details)).set(subpFlag);
  return *symbol;
}

// If name is a generic, look for the specific subprogram with the same
// name. Return that subprogram symbol or nullptr.
Symbol *SubprogramVisitor::GetSpecificFromGeneric(const parser::Name &name) {
  if (Symbol *symbol = FindSymbol(name.source)) {
    if (auto *details = symbol->detailsIf<GenericDetails>()) {
      // found generic, want subprogram
      auto *specific = details->specific().get();
      if (isGeneric()) {
        if (specific) {
          SayAlreadyDeclared(name.source, *specific);
        } else {
          SetSpecificInGeneric(
              Symbol{CurrScope(), name.source, Attrs{}, SubprogramDetails{}});
          specific = details->specific().get();
        }
      }
      if (specific) {
        if (!specific->has<SubprogramDetails>()) {
          specific->set_details(SubprogramDetails{});
        }
        return specific;
      }
    }
  }
  return nullptr;
}

// DeclarationVisitor implementation

bool DeclarationVisitor::BeginDecl() {
  BeginDeclTypeSpec();
  BeginAttrs();
  BeginArraySpec();
  return true;
}
void DeclarationVisitor::EndDecl() {
  EndDeclTypeSpec();
  EndAttrs();
  EndArraySpec();
}

void DeclarationVisitor::Post(const parser::DimensionStmt::Declaration &x) {
  const auto &name = std::get<parser::Name>(x.t);
  DeclareObjectEntity(name, Attrs{});
}

void DeclarationVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name{std::get<parser::ObjectName>(x.t)};
  // TODO: CoarraySpec, CharLength, Initialization
  Attrs attrs{attrs_ ? *attrs_ : Attrs{}};
  if (!arraySpec().empty()) {
    DeclareObjectEntity(name, attrs);
  } else {
    Symbol &symbol{DeclareEntity<EntityDetails>(name, attrs)};
    if (auto &type = GetDeclTypeSpec()) {
      SetType(name.source, symbol, *type);
    }
  }
}

bool DeclarationVisitor::Pre(const parser::AsynchronousStmt &x) {
  return HandleAttributeStmt(Attr::ASYNCHRONOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ContiguousStmt &x) {
  return HandleAttributeStmt(Attr::CONTIGUOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ExternalStmt &x) {
  HandleAttributeStmt(Attr::EXTERNAL, x.v);
  for (const auto &name : x.v) {
    auto *symbol = FindSymbol(name.source);
    if (symbol->has<ProcEntityDetails>()) {
      // nothing to do
    } else if (symbol->has<UnknownDetails>()) {
      symbol->set_details(ProcEntityDetails{});
    } else if (auto *details = symbol->detailsIf<EntityDetails>()) {
      symbol->set_details(ProcEntityDetails(*details));
      symbol->set(Symbol::Flag::Function);
    } else {
      Say2(name.source, "EXTERNAL attribute not allowed on '%s'"_err_en_US,
          symbol->name(), "Declaration of '%s'"_en_US);
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::IntrinsicStmt &x) {
  return HandleAttributeStmt(Attr::INTRINSIC, x.v);
}
bool DeclarationVisitor::Pre(const parser::OptionalStmt &x) {
  return HandleAttributeStmt(Attr::OPTIONAL, x.v);
}
bool DeclarationVisitor::Pre(const parser::ProtectedStmt &x) {
  return HandleAttributeStmt(Attr::PROTECTED, x.v);
}
bool DeclarationVisitor::Pre(const parser::ValueStmt &x) {
  return HandleAttributeStmt(Attr::VALUE, x.v);
}
bool DeclarationVisitor::Pre(const parser::VolatileStmt &x) {
  return HandleAttributeStmt(Attr::VOLATILE, x.v);
}
bool DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const std::list<parser::Name> &names) {
  for (const auto &name : names) {
    const auto pair = CurrScope().try_emplace(name.source, Attrs{attr});
    if (!pair.second) {
      // symbol was already there: set attribute on it
      Symbol &symbol{pair.first->second};
      if (attr != Attr::ASYNCHRONOUS && attr != Attr::VOLATILE &&
          symbol.has<UseDetails>()) {
        Say(*currStmtSource(),
            "Cannot change %s attribute on use-associated '%s'"_err_en_US,
            EnumToString(attr), name.source);
      }
      symbol.attrs().set(attr);
    }
  }
  return false;
}

void DeclarationVisitor::Post(const parser::ObjectDecl &x) {
  CHECK(objectDeclAttr_.has_value());
  const auto &name = std::get<parser::ObjectName>(x.t);
  DeclareObjectEntity(name, Attrs{*objectDeclAttr_});
}

void DeclarationVisitor::DeclareProcEntity(
    const parser::Name &name, Attrs attrs, ProcInterface &&interface) {
  Symbol &symbol{DeclareEntity<ProcEntityDetails>(name, attrs)};
  if (auto *details = symbol.detailsIf<ProcEntityDetails>()) {
    if (interface.type()) {
      symbol.set(Symbol::Flag::Function);
    } else if (interface.symbol()) {
      symbol.set(interface.symbol()->test(Symbol::Flag::Function)
              ? Symbol::Flag::Function
              : Symbol::Flag::Subroutine);
    }
    details->set_interface(std::move(interface));
    symbol.attrs().set(Attr::EXTERNAL);
  }
}

void DeclarationVisitor::DeclareObjectEntity(
    const parser::Name &name, Attrs attrs) {
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, attrs)};
  if (auto *details = symbol.detailsIf<ObjectEntityDetails>()) {
    if (auto &type = GetDeclTypeSpec()) {
      if (details->type().has_value()) {
        Say(name, "The type of '%s' has already been declared"_err_en_US);
      } else {
        details->set_type(*type);
      }
    }
    if (!arraySpec().empty()) {
      if (!details->shape().empty()) {
        Say(name,
            "The dimensions of '%s' have already been declared"_err_en_US);
      } else {
        details->set_shape(arraySpec());
      }
      ClearArraySpec();
    }
  }
}

bool DeclarationVisitor::Pre(const parser::DerivedTypeDef &x) {
  CHECK(!derivedTypeData_);
  derivedTypeData_ = std::make_unique<DerivedTypeDef::Data>();
  return true;
}
void DeclarationVisitor::Post(const parser::DerivedTypeDef &x) {
  DerivedTypeDef derivedType{*derivedTypeData_};
  // TODO: do something with derivedType
  derivedTypeData_.reset();
}
bool DeclarationVisitor::Pre(const parser::DerivedTypeStmt &x) {
  derivedTypeData_->name = std::get<parser::Name>(x.t).source;
  BeginAttrs();
  return true;
}
void DeclarationVisitor::Post(const parser::DerivedTypeStmt &x) {
  derivedTypeData_->attrs = GetAttrs();
  EndAttrs();
}
bool DeclarationVisitor::Pre(const parser::TypeAttrSpec::Extends &x) {
  derivedTypeData_->extends = x.v.source;
  return false;
}
bool DeclarationVisitor::Pre(const parser::PrivateStmt &x) {
  derivedTypeData_->Private = true;
  return false;
}
bool DeclarationVisitor::Pre(const parser::SequenceStmt &x) {
  derivedTypeData_->sequence = true;
  return false;
}
void DeclarationVisitor::Post(const parser::ComponentDecl &x) {
  const auto &name = std::get<parser::Name>(x.t).source;
  derivedTypeData_->dataComps.emplace_back(
      *GetDeclTypeSpec(), name, GetAttrs(), arraySpec());
  ClearArraySpec();
}
bool DeclarationVisitor::Pre(const parser::ProcedureDeclarationStmt &) {
  CHECK(!interfaceName_);
  return BeginDecl();
}
void DeclarationVisitor::Post(const parser::ProcedureDeclarationStmt &) {
  interfaceName_ = nullptr;
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::ProcComponentDefStmt &) {
  CHECK(!interfaceName_);
  return true;
}
void DeclarationVisitor::Post(const parser::ProcComponentDefStmt &) {
  interfaceName_ = nullptr;
}
void DeclarationVisitor::Post(const parser::ProcInterface &x) {
  if (auto *name = std::get_if<parser::Name>(&x.u)) {
    interfaceName_ = &name->source;
  }
}

void DeclarationVisitor::Post(const parser::ProcDecl &x) {
  const auto &name = std::get<parser::Name>(x.t);
  ProcInterface interface;
  if (interfaceName_) {
    auto *symbol = FindSymbol(*interfaceName_);
    if (!symbol) {
      Say(*interfaceName_, "Explicit interface '%s' not found"_err_en_US);
    } else if (!symbol->HasExplicitInterface()) {
      Say2(*interfaceName_,
          "'%s' is not an abstract interface or a procedure with an explicit interface"_err_en_US,
          symbol->name(), "Declaration of '%s'"_en_US);
    } else {
      interface.set_symbol(*symbol);
    }
  } else if (auto &type = GetDeclTypeSpec()) {
    interface.set_type(*type);
  }
  if (derivedTypeData_) {
    derivedTypeData_->procComps.emplace_back(
        ProcDecl{name.source}, GetAttrs(), std::move(interface));
  } else {
    DeclareProcEntity(name, GetAttrs(), std::move(interface));
  }
}

bool DeclarationVisitor::Pre(const parser::FinalProcedureStmt &x) {
  for (const parser::Name &name : x.v) {
    derivedTypeData_->finalProcs.push_back(name.source);
  }
  return false;
}

void DeclarationVisitor::SetType(
    const SourceName &name, Symbol &symbol, const DeclTypeSpec &type) {
  if (auto *details = symbol.detailsIf<EntityDetails>()) {
    if (!details->type().has_value()) {
      details->set_type(type);
      return;
    }
  } else if (auto *details = symbol.detailsIf<ObjectEntityDetails>()) {
    if (!details->type().has_value()) {
      details->set_type(type);
      return;
    }
  } else if (auto *details = symbol.detailsIf<ProcEntityDetails>()) {
    if (!details->interface().type()) {
      details->interface().set_type(type);
      return;
    }
  } else {
    return;
  }
  Say(name, "The type of '%s' has already been declared"_err_en_US);
}

// ResolveNamesVisitor implementation

bool ResolveNamesVisitor::Pre(const parser::TypeParamDefStmt &x) {
  BeginDeclTypeSpec();
  return true;
}
void ResolveNamesVisitor::Post(const parser::TypeParamDefStmt &x) {
  EndDeclTypeSpec();
  // TODO: TypeParamDefStmt
}

bool ResolveNamesVisitor::Pre(const parser::CommonBlockObject &x) {
  BeginArraySpec();
  return true;
}
void ResolveNamesVisitor::Post(const parser::CommonBlockObject &x) {
  ClearArraySpec();
  // TODO: CommonBlockObject
}

void ResolveNamesVisitor::Post(const parser::ComponentDecl &) {
  ClearArraySpec();
}

bool ResolveNamesVisitor::Pre(const parser::PrefixSpec &x) {
  return true;  // TODO
}

bool ResolveNamesVisitor::Pre(const parser::FunctionReference &) {
  expectedProcFlag_ = Symbol::Flag::Function;
  return true;
}
void ResolveNamesVisitor::Post(const parser::FunctionReference &) {
  expectedProcFlag_ = std::nullopt;
}
bool ResolveNamesVisitor::Pre(const parser::CallStmt &) {
  expectedProcFlag_ = Symbol::Flag::Subroutine;
  return true;
}
void ResolveNamesVisitor::Post(const parser::CallStmt &) {
  expectedProcFlag_ = std::nullopt;
}

bool ResolveNamesVisitor::CheckUseError(
    const SourceName &name, const Symbol &symbol) {
  const auto *details = symbol.detailsIf<UseErrorDetails>();
  if (!details) {
    return false;
  }
  Message &msg{Say(name, "Reference to '%s' is ambiguous"_err_en_US)};
  for (const auto &pair : details->occurrences()) {
    const SourceName &location{*pair.first};
    const SourceName &moduleName{pair.second->name()};
    msg.Attach(location,
        MessageFormattedText{"'%s' was use-associated from module '%s'"_en_US,
            name.ToString().data(), moduleName.ToString().data()});
  }
  return true;
}

void ResolveNamesVisitor::Post(const parser::ProcedureDesignator &x) {
  if (const auto *name = std::get_if<parser::Name>(&x.u)) {
    Symbol &symbol{MakeSymbol(name->source)};
    if (symbol.has<UnknownDetails>()) {
      if (isImplicitNoneExternal() && !symbol.attrs().test(Attr::EXTERNAL)) {
        Say(*name,
            "'%s' is an external procedure without the EXTERNAL"
            " attribute in a scope with IMPLICIT NONE(EXTERNAL)"_err_en_US);
      }
      symbol.attrs().set(Attr::EXTERNAL);
      symbol.set_details(ProcEntityDetails{});
      CHECK(expectedProcFlag_);
      symbol.set(*expectedProcFlag_);
    } else if (CheckUseError(name->source, symbol)) {
      // error was reported
    } else {
      if (auto *details = symbol.detailsIf<EntityDetails>()) {
        symbol.set_details(ProcEntityDetails(*details));
        symbol.set(Symbol::Flag::Function);
      }
      if (symbol.test(Symbol::Flag::Function) &&
          expectedProcFlag_ == Symbol::Flag::Subroutine) {
        Say2(name->source,
            "Cannot call function '%s' like a subroutine"_err_en_US,
            symbol.name(), "Declaration of '%s'"_en_US);
      } else if (symbol.test(Symbol::Flag::Subroutine) &&
          expectedProcFlag_ == Symbol::Flag::Function) {
        Say2(name->source,
            "Cannot call subroutine '%s' like a function"_err_en_US,
            symbol.name(), "Declaration of '%s'"_en_US);
      } else if (symbol.detailsIf<ProcEntityDetails>()) {
        symbol.set(*expectedProcFlag_);  // in case it hasn't been set yet
      } else {
        Say2(name->source,
            "Use of '%s' as a procedure conflicts with its declaration"_err_en_US,
            symbol.name(), "Declaration of '%s'"_en_US);
      }
    }
  }
}

bool ModuleVisitor::Pre(const parser::AccessStmt &x) {
  Attr accessAttr = AccessSpecToAttr(std::get<parser::AccessSpec>(x.t));
  if (CurrScope().kind() != Scope::Kind::Module) {
    Say(*currStmtSource(),
        "%s statement may only appear in the specification part of a module"_err_en_US,
        EnumToString(accessAttr));
    return false;
  }
  const auto &accessIds = std::get<std::list<parser::AccessId>>(x.t);
  if (accessIds.empty()) {
    if (prevAccessStmt_) {
      Say("The default accessibility of this module has already been declared"_err_en_US)
          .Attach(*prevAccessStmt_, "Previous declaration"_en_US);
    }
    prevAccessStmt_ = currStmtSource();
    defaultAccess_ = accessAttr;
  } else {
    for (const auto &accessId : accessIds) {
      std::visit(
          common::visitors{
              [=](const parser::Name &y) { SetAccess(y, accessAttr); },
              [=](const common::Indirection<parser::GenericSpec> &y) {
                std::visit(
                    common::visitors{
                        [=](const parser::Name &z) {
                          SetAccess(z, accessAttr);
                        },
                        [](const auto &) { common::die("TODO: GenericSpec"); },
                    },
                    y->u);
              },
          },
          accessId.u);
    }
  }
  return false;
}

// Set the access specification for this name.
void ModuleVisitor::SetAccess(const parser::Name &name, Attr attr) {
  Symbol &symbol{MakeSymbol(name.source)};
  Attrs &attrs{symbol.attrs()};
  if (attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    // PUBLIC/PRIVATE already set: make it a fatal error if it changed
    Attr prev = attrs.test(Attr::PUBLIC) ? Attr::PUBLIC : Attr::PRIVATE;
    Say(name.source,
        attr == prev
            ? "The accessibility of '%s' has already been specified as %s"_en_US
            : "The accessibility of '%s' has already been specified as %s"_err_en_US,
        name.source, EnumToString(prev));
  } else {
    attrs.set(attr);
  }
}

static bool NeedsExplicitType(const Symbol &symbol) {
  if (symbol.has<UnknownDetails>()) {
    return true;
  } else if (const auto *details = symbol.detailsIf<EntityDetails>()) {
    return !details->type().has_value();
  } else if (const auto *details = symbol.detailsIf<ObjectEntityDetails>()) {
    return !details->type().has_value();
  } else if (const auto *details = symbol.detailsIf<ProcEntityDetails>()) {
    return details->interface().symbol() == nullptr &&
        details->interface().type() == nullptr;
  } else {
    return false;
  }
}

void ResolveNamesVisitor::Post(const parser::SpecificationPart &s) {
  badStmtFuncFound_ = false;
  if (isImplicitNoneType()) {
    // Check that every name referenced has an explicit type
    for (const auto &pair : CurrScope()) {
      const auto &name = pair.first;
      const auto &symbol = pair.second;
      if (NeedsExplicitType(symbol)) {
        Say(name, "No explicit type declared for '%s'"_err_en_US);
      }
    }
  }
}

bool ResolveNamesVisitor::Pre(const parser::MainProgram &x) {
  Scope &scope = CurrScope().MakeScope(Scope::Kind::MainProgram);
  PushScope(scope);
  using stmtType = std::optional<parser::Statement<parser::ProgramStmt>>;
  if (const stmtType &stmt = std::get<stmtType>(x.t)) {
    const parser::Name &name{stmt->statement.v};
    MakeSymbol(name, MainProgramDetails());
  }
  return true;
}

void ResolveNamesVisitor::Post(const parser::EndProgramStmt &) { PopScope(); }

const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::DataRef &x) {
  return std::get_if<parser::Name>(&x.u);
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Designator &x) {
  return std::visit(
      common::visitors{
          [&](const parser::ObjectName &x) { return &x; },
          [&](const parser::DataRef &x) { return GetVariableName(x); },
          [&](const auto &) {
            return static_cast<const parser::Name *>(nullptr);
          },
      },
      x.u);
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Expr &x) {
  if (const auto *designator =
          std::get_if<common::Indirection<parser::Designator>>(&x.u)) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Variable &x) {
  if (const auto *designator =
          std::get_if<common::Indirection<parser::Designator>>(&x.u)) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}

// If implicit types are allowed, ensure name is in the symbol table.
// Otherwise, report an error if it hasn't been declared.
void ResolveNamesVisitor::CheckImplicitSymbol(const parser::Name *name) {
  if (name) {
    if (const auto *symbol = FindSymbol(name->source)) {
      if (CheckUseError(name->source, *symbol) ||
          !symbol->has<UnknownDetails>()) {
        return;  // reported an error or symbol is declared
      }
    }
    if (isImplicitNoneType()) {
      Say(*name, "No explicit type declared for '%s'"_err_en_US);
    } else {
      CurrScope().try_emplace(name->source);
    }
  }
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!GetDeclTypeSpec());
}

void ResolveNames(
    parser::Program &program, const parser::CookedSource &cookedSource) {
  ResolveNamesVisitor visitor;
  parser::Walk(static_cast<const parser::Program &>(program), visitor);
  if (!visitor.messages().empty()) {
    visitor.messages().Emit(std::cerr, cookedSource);
    return;
  }
  RewriteParseTree(program);
}

// Map the enum in the parser to the one in GenericSpec
static GenericSpec::Kind MapIntrinsicOperator(
    parser::DefinedOperator::IntrinsicOperator x) {
  switch (x) {
  case parser::DefinedOperator::IntrinsicOperator::Add:
    return GenericSpec::OP_ADD;
  case parser::DefinedOperator::IntrinsicOperator::AND:
    return GenericSpec::OP_AND;
  case parser::DefinedOperator::IntrinsicOperator::Concat:
    return GenericSpec::OP_CONCAT;
  case parser::DefinedOperator::IntrinsicOperator::Divide:
    return GenericSpec::OP_DIVIDE;
  case parser::DefinedOperator::IntrinsicOperator::EQ:
    return GenericSpec::OP_EQ;
  case parser::DefinedOperator::IntrinsicOperator::EQV:
    return GenericSpec::OP_EQV;
  case parser::DefinedOperator::IntrinsicOperator::GE:
    return GenericSpec::OP_GE;
  case parser::DefinedOperator::IntrinsicOperator::GT:
    return GenericSpec::OP_GT;
  case parser::DefinedOperator::IntrinsicOperator::LE:
    return GenericSpec::OP_LE;
  case parser::DefinedOperator::IntrinsicOperator::LT:
    return GenericSpec::OP_LT;
  case parser::DefinedOperator::IntrinsicOperator::Multiply:
    return GenericSpec::OP_MULTIPLY;
  case parser::DefinedOperator::IntrinsicOperator::NE:
    return GenericSpec::OP_NE;
  case parser::DefinedOperator::IntrinsicOperator::NEQV:
    return GenericSpec::OP_NEQV;
  case parser::DefinedOperator::IntrinsicOperator::NOT:
    return GenericSpec::OP_NOT;
  case parser::DefinedOperator::IntrinsicOperator::OR:
    return GenericSpec::OP_OR;
  case parser::DefinedOperator::IntrinsicOperator::Power:
    return GenericSpec::OP_POWER;
  case parser::DefinedOperator::IntrinsicOperator::Subtract:
    return GenericSpec::OP_SUBTRACT;
  case parser::DefinedOperator::IntrinsicOperator::XOR:
    return GenericSpec::OP_XOR;
  default: CRASH_NO_CASE;
  }
}

// Map a parser::GenericSpec to a semantics::GenericSpec
static GenericSpec MapGenericSpec(const parser::GenericSpec &genericSpec) {
  return std::visit(
      common::visitors{
          [](const parser::Name &x) {
            return GenericSpec::GenericName(x.source);
          },
          [](const parser::DefinedOperator &x) {
            return std::visit(
                common::visitors{
                    [](const parser::DefinedOpName &name) {
                      return GenericSpec::DefinedOp(name.v.source);
                    },
                    [](const parser::DefinedOperator::IntrinsicOperator &x) {
                      return GenericSpec::IntrinsicOp(MapIntrinsicOperator(x));
                    },
                },
                x.u);
          },
          [](const parser::GenericSpec::Assignment &) {
            return GenericSpec::IntrinsicOp(GenericSpec::ASSIGNMENT);
          },
          [](const parser::GenericSpec::ReadFormatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::READ_FORMATTED);
          },
          [](const parser::GenericSpec::ReadUnformatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::READ_UNFORMATTED);
          },
          [](const parser::GenericSpec::WriteFormatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::WRITE_FORMATTED);
          },
          [](const parser::GenericSpec::WriteUnformatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::WRITE_UNFORMATTED);
          },
      },
      genericSpec.u);
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

static void DumpSymbols(std::ostream &os, const Scope &scope, int indent = 0) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol = scope.symbol()) {
    os << ' ' << symbol->name().ToString();
  }
  os << '\n';
  ++indent;
  for (const auto &symbol : scope) {
    PutIndent(os, indent);
    os << symbol.second << "\n";
  }
  for (const auto &child : scope.children()) {
    DumpSymbols(os, child, indent);
  }
  --indent;
}

void DumpSymbols(std::ostream &os) { DumpSymbols(os, Scope::globalScope); }

}  // namespace Fortran::semantics
