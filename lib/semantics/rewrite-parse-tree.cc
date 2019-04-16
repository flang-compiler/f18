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

#include "rewrite-parse-tree.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "../common/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>

namespace Fortran::semantics {

using namespace parser::literals;

/// Convert mis-identified statement functions to array element assignments.
/// Convert mis-identified format expressions to namelist group names.
/// Convert mis-identified character variables in I/O units to integer
/// unit number expressions.
class RewriteMutator {
public:
  RewriteMutator(parser::Messages &messages) : messages_{messages} {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}

  void Post(parser::Name &);
  void Post(parser::SpecificationPart &);
  bool Pre(parser::ExecutionPart &);
  void Post(parser::IoUnit &);
  void Post(parser::ReadStmt &);
  void Post(parser::WriteStmt &);

  // Name resolution yet implemented:
  bool Pre(parser::EquivalenceStmt &) { return false; }
  bool Pre(parser::Keyword &) { return false; }
  bool Pre(parser::EntryStmt &) { return false; }
  bool Pre(parser::CompilerDirective &) { return false; }

  // Don't bother resolving names in end statements.
  bool Pre(parser::EndBlockDataStmt &) { return false; }
  bool Pre(parser::EndFunctionStmt &) { return false; }
  bool Pre(parser::EndModuleStmt &) { return false; }
  bool Pre(parser::EndMpSubprogramStmt &) { return false; }
  bool Pre(parser::EndProgramStmt &) { return false; }
  bool Pre(parser::EndSubmoduleStmt &) { return false; }
  bool Pre(parser::EndSubroutineStmt &) { return false; }
  bool Pre(parser::EndTypeStmt &) { return false; }

private:
  using stmtFuncType =
      parser::Statement<common::Indirection<parser::StmtFunctionStmt>>;
  bool errorOnUnresolvedName_{true};
  parser::Messages &messages_;
  std::list<stmtFuncType> stmtFuncsToConvert_;
};

// Check that name has been resolved to a symbol
void RewriteMutator::Post(parser::Name &name) {
  if (name.symbol == nullptr && errorOnUnresolvedName_) {
    messages_.Say(name.source, "Internal: no symbol found for '%s'"_err_en_US,
        name.ToString().c_str());
  }
}

// Find mis-parsed statement functions and move to stmtFuncsToConvert_ list.
void RewriteMutator::Post(parser::SpecificationPart &x) {
  auto &list{std::get<std::list<parser::DeclarationConstruct>>(x.t)};
  for (auto it{list.begin()}; it != list.end();) {
    if (auto stmt{std::get_if<stmtFuncType>(&it->u)}) {
      Symbol *symbol{std::get<parser::Name>(stmt->statement.value().t).symbol};
      if (symbol && symbol->has<ObjectEntityDetails>()) {
        // not a stmt func: remove it here and add to ones to convert
        stmtFuncsToConvert_.push_back(std::move(*stmt));
        it = list.erase(it);
        continue;
      }
    }
    ++it;
  }
}

// Insert converted assignments at start of ExecutionPart.
bool RewriteMutator::Pre(parser::ExecutionPart &x) {
  auto origFirst{x.v.begin()};  // insert each elem before origFirst
  for (stmtFuncType &sf : stmtFuncsToConvert_) {
    auto stmt{sf.statement.value().ConvertToAssignment()};
    stmt.source = sf.source;
    x.v.insert(origFirst,
        parser::ExecutionPartConstruct{
            parser::ExecutableConstruct{std::move(stmt)}});
  }
  stmtFuncsToConvert_.clear();
  return true;
}

static DeclTypeSpec *GetType(const parser::Name &x) {
  if (x.symbol != nullptr) {
    return x.symbol->GetType();
  } else {
    return nullptr;
  }
}
static DeclTypeSpec *GetType(const parser::StructureComponent &x) {
  return GetType(x.component);
}
static DeclTypeSpec *GetType(const parser::DataRef &x) {
  return std::visit(
      common::visitors{
          [](const parser::Name &name) { return GetType(name); },
          [](const common::Indirection<parser::StructureComponent> &sc) {
            return GetType(sc.value());
          },
          [](const common::Indirection<parser::ArrayElement> &sc) {
            return GetType(sc.value().base);
          },
          [](const common::Indirection<parser::CoindexedNamedObject> &ci) {
            return GetType(ci.value().base);
          },
      },
      x.u);
}
static DeclTypeSpec *GetType(const parser::Substring &x) {
  return GetType(std::get<parser::DataRef>(x.t));
}
static DeclTypeSpec *GetType(const parser::Designator &x) {
  return std::visit([](const auto &y) { return GetType(y); }, x.u);
}
static DeclTypeSpec *GetType(const parser::ProcComponentRef &x) {
  return GetType(x.v.thing);
}
static DeclTypeSpec *GetType(const parser::ProcedureDesignator &x) {
  return std::visit([](const auto &y) { return GetType(y); }, x.u);
}
static DeclTypeSpec *GetType(const parser::Call &x) {
  return GetType(std::get<parser::ProcedureDesignator>(x.t));
}
static DeclTypeSpec *GetType(const parser::FunctionReference &x) {
  return GetType(x.v);
}
static DeclTypeSpec *GetType(const parser::Variable &x) {
  return std::visit(
      [](const auto &indirection) { return GetType(indirection.value()); },
      x.u);
}

void RewriteMutator::Post(parser::IoUnit &x) {
  if (auto *var{std::get_if<parser::Variable>(&x.u)}) {
    DeclTypeSpec *type{GetType(*var)};
    if (type == nullptr || type->category() != DeclTypeSpec::Character) {
      // If the Variable is not known to be character (any kind), transform
      // the I/O unit in situ to a FileUnitNumber so that automatic expression
      // constraint checking will be applied.
      auto expr{std::visit(
          [](auto &&indirection) {
            return parser::Expr{std::move(indirection)};
          },
          std::move(var->u))};
      x.u = parser::FileUnitNumber{
          parser::ScalarIntExpr{parser::IntExpr{std::move(expr)}}};
    }
  }
}

// When a namelist group name appears (without NML=) in a READ or WRITE
// statement in such a way that it can be misparsed as a format expression,
// rewrite the I/O statement's parse tree node as if the namelist group
// name had appeared with NML=.
template<typename READ_OR_WRITE>
void FixMisparsedUntaggedNamelistName(READ_OR_WRITE &x) {
  if (x.iounit.has_value() && x.format.has_value()) {
    if (auto *charExpr{
            std::get_if<parser::DefaultCharExpr>(&x.format.value().u)}) {
      parser::Expr &expr{charExpr->thing.value()};
      parser::Name *name{GetSimpleName(expr)};
      if (name != nullptr && name->symbol != nullptr &&
          name->symbol->has<NamelistDetails>()) {
        x.controls.emplace_front(parser::IoControlSpec{std::move(*name)});
        x.format.reset();
      }
    }
  }
}

void RewriteMutator::Post(parser::ReadStmt &x) {
  FixMisparsedUntaggedNamelistName(x);
}

void RewriteMutator::Post(parser::WriteStmt &x) {
  FixMisparsedUntaggedNamelistName(x);
}

bool RewriteParseTree(SemanticsContext &context, parser::Program &program) {
  RewriteMutator mutator{context.messages()};
  parser::Walk(program, mutator);
  return !context.AnyFatalError();
}

}
