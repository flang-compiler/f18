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

#include "rewrite-parse-tree.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "../common/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>

namespace Fortran::semantics {

using namespace parser::literals;

// Symbols collected during name resolution that are added to parse tree.
using symbolMap = std::map<const char *, Symbol *>;

/// Walk the parse tree and add symbols from the symbolMap in Name nodes.
/// Convert mis-identified statement functions to array element assignments.
class RewriteMutator {
public:
  RewriteMutator(parser::Messages &messages, const symbolMap &symbols)
    : messages_{messages}, symbols_{symbols} {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}

  void Post(parser::Name &);
  void Post(parser::SpecificationPart &);
  bool Pre(parser::ExecutionPart &);
  void Post(parser::Variable &x) { ConvertFunctionRef(x); }
  void Post(parser::Expr &x) { ConvertFunctionRef(x); }

  // Name resolution yet implemented:
  bool Pre(parser::CommonStmt &) { return false; }
  bool Pre(parser::NamelistStmt &) { return false; }
  bool Pre(parser::EquivalenceStmt &) { return false; }
  bool Pre(parser::BindEntity &) { return false; }
  bool Pre(parser::Keyword &) { return false; }
  bool Pre(parser::SavedEntity &) { return false; }
  bool Pre(parser::EntryStmt &) { return false; }

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
  const symbolMap &symbols_;
  std::list<stmtFuncType> stmtFuncsToConvert_;

  // For T = Variable or Expr, if x has a function reference that really
  // should be an array element reference (i.e. the name occurs in an
  // entity declaration, convert it.
  template<typename T> void ConvertFunctionRef(T &x) {
    auto *funcRef{
        std::get_if<common::Indirection<parser::FunctionReference>>(&x.u)};
    if (!funcRef) {
      return;
    }
    parser::Name *name{std::get_if<parser::Name>(
        &std::get<parser::ProcedureDesignator>((*funcRef)->v.t).u)};
    if (!name || !name->symbol ||
        !name->symbol->GetUltimate().has<ObjectEntityDetails>()) {
      return;
    }
    x.u = common::Indirection{(*funcRef)->ConvertToArrayElementRef()};
  }
};

// Fill in name.symbol if there is a corresponding symbol
void RewriteMutator::Post(parser::Name &name) {
  const auto it{symbols_.find(name.source.begin())};
  if (it != symbols_.end()) {
    name.symbol = it->second;
  } else if (errorOnUnresolvedName_) {
    messages_.Say(name.source, "Internal: no symbol found for '%s'"_en_US,
        name.ToString().c_str());
  }
}

// Find mis-parsed statement functions and move to stmtFuncsToConvert_ list.
void RewriteMutator::Post(parser::SpecificationPart &x) {
  auto &list{std::get<std::list<parser::DeclarationConstruct>>(x.t)};
  for (auto it{list.begin()}; it != list.end();) {
    if (auto stmt{std::get_if<stmtFuncType>(&it->u)}) {
      Symbol *symbol{std::get<parser::Name>(stmt->statement->t).symbol};
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
    auto &&stmt = sf.statement->ConvertToAssignment();
    stmt.source = sf.source;
    x.v.insert(origFirst,
        parser::ExecutionPartConstruct{parser::ExecutableConstruct{stmt}});
  }
  stmtFuncsToConvert_.clear();
  return true;
}

static void CollectSymbol(Symbol &symbol, symbolMap &symbols) {
  for (const auto &name : symbol.occurrences()) {
    symbols.emplace(name.begin(), &symbol);
  }
}

static void CollectSymbols(const Scope &scope, symbolMap &symbols) {
  for (auto &pair : scope) {
    Symbol *symbol{pair.second};
    CollectSymbol(*symbol, symbols);
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      if (details->derivedType()) {
        CollectSymbol(*details->derivedType(), symbols);
      }
    }
  }
  for (auto &child : scope.children()) {
    CollectSymbols(child, symbols);
  }
}

void RewriteParseTree(SemanticsContext &context, parser::Program &program) {
  symbolMap symbols;
  CollectSymbols(context.globalScope(), symbols);
  RewriteMutator mutator{context.messages(), symbols};
  parser::Walk(program, mutator);
}

}  // namespace Fortran::semantics
