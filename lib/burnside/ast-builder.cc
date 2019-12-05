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

#include "ast-builder.h"
#include "../parser/parse-tree-visitor.h"
#include <cassert>
#include <utility>

/// Build an light-weight AST to help with lowering to FIR.  The AST will
/// capture pointers back into the parse tree, so the parse tree data structure
/// may <em>not</em> be changed between the construction of the AST and all of
/// its uses.
///
/// The AST captures a structured view of the program.  The program is a list of
/// units.  Function like units will contain lists of evaluations.  Evaluations
/// are either statements or constructs, where a construct contains a list of
/// evaluations.  The resulting AST structure can then be used to create FIR.

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Pa = Fortran::parser;

using namespace Fortran;
using namespace Br;

namespace {

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time, so one goal here is to limit
/// the bridge to one such instantiation.
class ASTBuilder {
public:
  ASTBuilder() {
    pgm = new AST::Program;
    parents.push_back(pgm);
  }

  /// Get the result
  AST::Program *result() { return pgm; }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  // Module like

  bool Pre(const Pa::Module &x) { return enterModule(x); }
  bool Pre(const Pa::Submodule &x) { return enterModule(x); }

  void Post(const Pa::Module &) { exitModule(); }
  void Post(const Pa::Submodule &) { exitModule(); }

  // Function like

  bool Pre(const Pa::MainProgram &x) { return enterFunc(x); }
  bool Pre(const Pa::FunctionSubprogram &x) { return enterFunc(x); }
  bool Pre(const Pa::SubroutineSubprogram &x) { return enterFunc(x); }
  bool Pre(const Pa::SeparateModuleSubprogram &x) { return enterFunc(x); }

  void Post(const Pa::MainProgram &) { exitFunc(); }
  void Post(const Pa::FunctionSubprogram &) { exitFunc(); }
  void Post(const Pa::SubroutineSubprogram &) { exitFunc(); }
  void Post(const Pa::SeparateModuleSubprogram &) { exitFunc(); }

  // Block data

  void Post(const Pa::BlockData &x) {
    AST::BlockDataUnit unit{x, parents.back()};
    addUnit(unit);
  }

  //
  // Action statements
  //

  void Post(const Pa::Statement<Pa::ActionStmt> &s) {
    addEval(makeEvalAction(s));
  }
  void Post(const Pa::UnlabeledStatement<Pa::ActionStmt> &s) {
    addEval(makeEvalAction(s));
  }

  //
  // Non-executable statements
  //

  void Post(const Pa::Statement<Co::Indirection<Pa::FormatStmt>> &s) {
    addEval(makeEvalIndirect(s));
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::EntryStmt>> &s) {
    addEval(makeEvalIndirect(s));
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::DataStmt>> &s) {
    addEval(makeEvalIndirect(s));
  }
  void Post(const Pa::Statement<Co::Indirection<Pa::NamelistStmt>> &s) {
    addEval(makeEvalIndirect(s));
  }

  //
  // Construct statements
  //

  void Post(const Pa::Statement<parser::AssociateStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndAssociateStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::BlockStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndBlockStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::SelectCaseStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::CaseStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndSelectStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::ChangeTeamStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndChangeTeamStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::CriticalStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndCriticalStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::NonLabelDoStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndDoStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::IfThenStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::ElseIfStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::ElseStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndIfStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::SelectRankStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::SelectRankCaseStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::SelectTypeStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::TypeGuardStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::WhereConstructStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::MaskedElsewhereStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::ElsewhereStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndWhereStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::ForallConstructStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  void Post(const Pa::Statement<parser::EndForallStmt> &s) {
    addEval(makeEvalDirect(s));
  }
  // Get rid of production wrapper
  void Post(const Pa::UnlabeledStatement<parser::ForallAssignmentStmt> &s) {
    addEval(std::visit(
        [&](const auto &x) {
          return AST::Evaluation{x, s.source, {}, parents.back()};
        },
        s.statement.u));
  }
  void Post(const Pa::Statement<parser::ForallAssignmentStmt> &s) {
    addEval(std::visit(
        [&](const auto &x) {
          return AST::Evaluation{x, s.source, s.label, parents.back()};
        },
        s.statement.u));
  }

  //
  // Constructs (enter and exit)
  //

  bool Pre(const Pa::AssociateConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::BlockConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CaseConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::ChangeTeamConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CriticalConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::DoConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::IfConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::SelectRankConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::SelectTypeConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::WhereConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::ForallConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::CompilerDirective &c) { return enterConstruct(c); }
  bool Pre(const Pa::OpenMPConstruct &c) { return enterConstruct(c); }
  bool Pre(const Pa::OmpEndLoopDirective &c) { return enterConstruct(c); }

  void Post(const Pa::AssociateConstruct &) { exitConstruct(); }
  void Post(const Pa::BlockConstruct &) { exitConstruct(); }
  void Post(const Pa::CaseConstruct &) { exitConstruct(); }
  void Post(const Pa::ChangeTeamConstruct &) { exitConstruct(); }
  void Post(const Pa::CriticalConstruct &) { exitConstruct(); }
  void Post(const Pa::DoConstruct &) { exitConstruct(); }
  void Post(const Pa::IfConstruct &) { exitConstruct(); }
  void Post(const Pa::SelectRankConstruct &) { exitConstruct(); }
  void Post(const Pa::SelectTypeConstruct &) { exitConstruct(); }
  void Post(const Pa::WhereConstruct &) { exitConstruct(); }
  void Post(const Pa::ForallConstruct &) { exitConstruct(); }
  void Post(const Pa::CompilerDirective &) { exitConstruct(); }
  void Post(const Pa::OpenMPConstruct &) { exitConstruct(); }
  void Post(const Pa::OmpEndLoopDirective &) { exitConstruct(); }

private:
  // ActionStmt has a couple of non-conforming cases, which get handled
  // explicitly here.  The other cases use an Indirection, which we discard in
  // the AST.
  AST::Evaluation makeEvalAction(const Pa::Statement<Pa::ActionStmt> &s) {
    return std::visit(
        common::visitors{
            [&](const Pa::ContinueStmt &x) {
              return AST::Evaluation{x, s.source, s.label, parents.back()};
            },
            [&](const Pa::FailImageStmt &x) {
              return AST::Evaluation{x, s.source, s.label, parents.back()};
            },
            [&](const auto &x) {
              return AST::Evaluation{
                  x.value(), s.source, s.label, parents.back()};
            },
        },
        s.statement.u);
  }
  AST::Evaluation makeEvalAction(
      const Pa::UnlabeledStatement<Pa::ActionStmt> &s) {
    return std::visit(
        common::visitors{
            [&](const Pa::ContinueStmt &x) {
              return AST::Evaluation{x, s.source, {}, parents.back()};
            },
            [&](const Pa::FailImageStmt &x) {
              return AST::Evaluation{x, s.source, {}, parents.back()};
            },
            [&](const auto &x) {
              return AST::Evaluation{x.value(), s.source, {}, parents.back()};
            },
        },
        s.statement.u);
  }

  template<typename A>
  AST::Evaluation makeEvalIndirect(const Pa::Statement<Co::Indirection<A>> &s) {
    return AST::Evaluation{
        s.statement.value(), s.source, s.label, parents.back()};
  }

  template<typename A>
  AST::Evaluation makeEvalDirect(const Pa::Statement<A> &s) {
    return AST::Evaluation{s.statement, s.source, s.label, parents.back()};
  }

  // When we enter a function-like structure, we want to build a new unit and
  // set the builder's cursors to point to it.
  template<typename A> bool enterFunc(const A &f) {
    auto &unit = addFunc(AST::FunctionLikeUnit{f, parents.back()});
    funclist = &unit.funcs;
    pushEval(&unit.evals);
    parents.emplace_back(&unit);
    return true;
  }

  void exitFunc() {
    popEval();
    funclist = nullptr;
    parents.pop_back();
  }

  // When we enter a construct structure, we want to build a new construct and
  // set the builder's evaluation cursor to point to it.
  template<typename A> bool enterConstruct(const A &c) {
    auto &con = addEval(AST::Evaluation{c, parents.back()});
    con.subs = new std::list<AST::Evaluation>();
    pushEval(con.subs);
    parents.emplace_back(&con);
    return true;
  }

  void exitConstruct() {
    popEval();
    parents.pop_back();
  }

  // When we enter a module structure, we want to build a new module and
  // set the builder's function cursor to point to it.
  template<typename A> bool enterModule(const A &f) {
    auto &unit = addUnit(AST::ModuleLikeUnit{f, parents.back()});
    funclist = &unit.funcs;
    parents.emplace_back(&unit);
    return true;
  }

  void exitModule() {
    funclist = nullptr;
    parents.pop_back();
  }

  template<typename A> A &addUnit(const A &unit) {
    pgm->getUnits().emplace_back(unit);
    return std::get<A>(pgm->getUnits().back());
  }

  template<typename A> A &addFunc(const A &func) {
    if (funclist) {
      funclist->emplace_back(func);
      return funclist->back();
    }
    return addUnit(func);
  }

  /// move the Evaluation to the end of the current list
  AST::Evaluation &addEval(AST::Evaluation &&eval) {
    assert(funclist && "not in a function");
    assert(evallist.size() > 0);
    evallist.back()->emplace_back(std::move(eval));
    return evallist.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEval(std::list<AST::Evaluation> *eval) {
    assert(funclist && "not in a function");
    assert(eval && eval->empty() && "evaluation list isn't correct");
    evallist.emplace_back(eval);
  }

  /// pop the current list and return to the last Evaluation list
  void popEval() {
    assert(funclist && "not in a function");
    evallist.pop_back();
  }

  AST::Program *pgm;
  std::list<AST::FunctionLikeUnit> *funclist{nullptr};
  std::vector<std::list<AST::Evaluation> *> evallist;
  std::vector<AST::ParentType> parents;
};

template<typename A> constexpr bool hasErrLabel(const A &stmt) {
  if constexpr (std::is_same_v<A, Pa::ReadStmt> ||
      std::is_same_v<A, Pa::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<Pa::ErrLabel>(control.u)) {
        return true;
      }
    }
  }
  if constexpr (std::is_same_v<A, Pa::WaitStmt> ||
      std::is_same_v<A, Pa::OpenStmt> || std::is_same_v<A, Pa::CloseStmt> ||
      std::is_same_v<A, Pa::BackspaceStmt> ||
      std::is_same_v<A, Pa::EndfileStmt> || std::is_same_v<A, Pa::RewindStmt> ||
      std::is_same_v<A, Pa::FlushStmt>) {
    for (const auto &spec : stmt.v) {
      if (std::holds_alternative<Pa::ErrLabel>(spec.u)) {
        return true;
      }
    }
  }
  if constexpr (std::is_same_v<A, Pa::InquireStmt>) {
    for (const auto &spec : std::get<std::list<Pa::InquireSpec>>(stmt.u)) {
      if (std::holds_alternative<Pa::ErrLabel>(spec.u)) {
        return true;
      }
    }
  }
  return false;
}

template<typename A> constexpr bool hasEorLabel(const A &stmt) {
  if constexpr (std::is_same_v<A, Pa::ReadStmt> ||
      std::is_same_v<A, Pa::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<Pa::EorLabel>(control.u)) {
        return true;
      }
    }
  }
  if constexpr (std::is_same_v<A, Pa::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<Pa::EorLabel>(waitSpec.u)) {
        return true;
      }
    }
  }
  return false;
}

template<typename A> constexpr bool hasEndLabel(const A &stmt) {
  if constexpr (std::is_same_v<A, Pa::ReadStmt> ||
      std::is_same_v<A, Pa::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<Pa::EndLabel>(control.u)) {
        return true;
      }
    }
  }
  if constexpr (std::is_same_v<A, Pa::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<Pa::EndLabel>(waitSpec.u)) {
        return true;
      }
    }
  }
  return false;
}

bool hasAltReturns(const Pa::CallStmt &callStmt) {
  const auto &args{std::get<std::list<Pa::ActualArgSpec>>(callStmt.v.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<Pa::ActualArg>(arg.t)};
    if (std::holds_alternative<Pa::AltReturnSpec>(actual.u)) {
      return true;
    }
  }
  return false;
}

/// Determine if `callStmt` has alternate returns and if so set `e` to be the
/// origin of a switch-like control flow
void altRet(
    AST::Evaluation &e, const Pa::CallStmt *callStmt, AST::Evaluation *cstr) {
  if (hasAltReturns(*callStmt)) {
    e.setCFG(AST::CFGAnnotation::Switch, cstr);
  }
}

template<typename A>
void ioLabel(AST::Evaluation &e, const A *s, AST::Evaluation *cstr) {
  if (hasErrLabel(*s) || hasEorLabel(*s) || hasEndLabel(*s)) {
    e.setCFG(AST::CFGAnnotation::IoSwitch, cstr);
  }
}

void annotateEvalListCFG(
    std::list<AST::Evaluation> &evals, AST::Evaluation *cstr) {
  bool nextIsTarget = false;
  for (auto &e : evals) {
    e.isTarget = nextIsTarget;
    nextIsTarget = false;
    if (e.isConstruct()) {
      annotateEvalListCFG(*e.getConstructEvals(), &e);
      // assume that the entry and exit are both possible branch targets
      nextIsTarget = true;
    }
    if (e.isActionStmt() && e.lab.has_value()) {
      e.isTarget = true;
    }
    std::visit(
        common::visitors{
            [&](const Pa::BackspaceStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::CallStmt *s) { altRet(e, s, cstr); },
            [&](const Pa::CloseStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::CycleStmt *) {
              e.setCFG(AST::CFGAnnotation::Goto, cstr);
            },
            [&](const Pa::EndfileStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::ExitStmt *) {
              e.setCFG(AST::CFGAnnotation::Goto, cstr);
            },
            [&](const Pa::FailImageStmt *) {
              e.setCFG(AST::CFGAnnotation::Terminate, cstr);
            },
            [&](const Pa::FlushStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::GotoStmt *) {
              e.setCFG(AST::CFGAnnotation::Goto, cstr);
            },
            [&](const Pa::IfStmt *) {
              e.setCFG(AST::CFGAnnotation::CondGoto, cstr);
            },
            [&](const Pa::InquireStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::OpenStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::ReadStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::ReturnStmt *) {
              e.setCFG(AST::CFGAnnotation::Return, cstr);
            },
            [&](const Pa::RewindStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::StopStmt *) {
              e.setCFG(AST::CFGAnnotation::Terminate, cstr);
            },
            [&](const Pa::WaitStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::WriteStmt *s) { ioLabel(e, s, cstr); },
            [&](const Pa::ArithmeticIfStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch, cstr);
            },
            [&](const Pa::AssignedGotoStmt *) {
              e.setCFG(AST::CFGAnnotation::IndGoto, cstr);
            },
            [&](const Pa::ComputedGotoStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch, cstr);
            },
            [&](const Pa::WhereStmt *) {
              // fir.loop + fir.where around the next stmt
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Iterative, cstr);
            },
            [&](const Pa::ForallStmt *) {
              // fir.loop around the next stmt
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Iterative, cstr);
            },
            [&](AST::CGJump &) { e.setCFG(AST::CFGAnnotation::Goto, cstr); },
            [&](const Pa::EndAssociateStmt *) { e.isTarget = true; },
            [&](const Pa::EndBlockStmt *) { e.isTarget = true; },
            [&](const Pa::SelectCaseStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch, cstr);
            },
            [&](const Pa::CaseStmt *) { e.isTarget = true; },
            [&](const Pa::EndSelectStmt *) { e.isTarget = true; },
            [&](const Pa::EndChangeTeamStmt *) { e.isTarget = true; },
            [&](const Pa::EndCriticalStmt *) { e.isTarget = true; },
            [&](const Pa::NonLabelDoStmt *) {
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Iterative, cstr);
            },
            [&](const Pa::EndDoStmt *) {
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Goto, cstr);
            },
            [&](const Pa::IfThenStmt *) {
              e.setCFG(AST::CFGAnnotation::CondGoto, cstr);
            },
            [&](const Pa::ElseIfStmt *) {
              e.setCFG(AST::CFGAnnotation::CondGoto, cstr);
            },
            [&](const Pa::ElseStmt *) { e.isTarget = true; },
            [&](const Pa::EndIfStmt *) { e.isTarget = true; },
            [&](const Pa::SelectRankStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch, cstr);
            },
            [&](const Pa::SelectRankCaseStmt *) { e.isTarget = true; },
            [&](const Pa::SelectTypeStmt *) {
              e.setCFG(AST::CFGAnnotation::Switch, cstr);
            },
            [&](const Pa::TypeGuardStmt *) { e.isTarget = true; },
            [&](const Pa::WhereConstruct *) {
              // mark the WHERE as if it were a DO loop
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Iterative, cstr);
            },
            [&](const Pa::WhereConstructStmt *) {
              e.setCFG(AST::CFGAnnotation::CondGoto, cstr);
            },
            [&](const Pa::MaskedElsewhereStmt *) {
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::CondGoto, cstr);
            },
            [&](const Pa::ElsewhereStmt *) { e.isTarget = true; },
            [&](const Pa::EndWhereStmt *) { e.isTarget = true; },
            [&](const Pa::ForallConstructStmt *) {
              e.isTarget = true;
              e.setCFG(AST::CFGAnnotation::Iterative, cstr);
            },
            [&](const Pa::EndForallStmt *) { e.isTarget = true; },
            [](const auto *) { /* do nothing */ },
        },
        e.u);
  }
}

/// Annotate the AST with CFG source decorations (see CFGAnnotation) and mark
/// potential branch targets
inline void annotateFuncCFG(AST::FunctionLikeUnit &flu) {
  annotateEvalListCFG(flu.evals, nullptr);
}

}  // namespace

Br::AST::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::MainProgram &f, const AST::ParentType &parent)
  : ProgramUnit{&f, parent} {
  auto &ps{std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(f.t)};
  if (ps.has_value()) {
    const Pa::Statement<Pa::ProgramStmt> &s{ps.value()};
    funStmts.push_back(&s);
  }
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndProgramStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::FunctionSubprogram &f, const AST::ParentType &parent)
  : ProgramUnit{&f, parent} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::FunctionStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndFunctionStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::SubroutineSubprogram &f, const AST::ParentType &parent)
  : ProgramUnit{&f, parent} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::SubroutineStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndSubroutineStmt>>(f.t));
}

Br::AST::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::SeparateModuleSubprogram &f, const AST::ParentType &parent)
  : ProgramUnit{&f, parent} {
  funStmts.push_back(&std::get<Pa::Statement<Pa::MpSubprogramStmt>>(f.t));
  funStmts.push_back(&std::get<Pa::Statement<Pa::EndMpSubprogramStmt>>(f.t));
}

Br::AST::ModuleLikeUnit::ModuleLikeUnit(
    const Pa::Module &m, const AST::ParentType &parent)
  : ProgramUnit{&m, parent} {
  modStmts.push_back(&std::get<Pa::Statement<Pa::ModuleStmt>>(m.t));
  modStmts.push_back(&std::get<Pa::Statement<Pa::EndModuleStmt>>(m.t));
}

Br::AST::ModuleLikeUnit::ModuleLikeUnit(
    const Pa::Submodule &m, const AST::ParentType &parent)
  : ProgramUnit{&m, parent} {
  modStmts.push_back(&std::get<Pa::Statement<Pa::SubmoduleStmt>>(m.t));
  modStmts.push_back(&std::get<Pa::Statement<Pa::EndSubmoduleStmt>>(m.t));
}

Br::AST::BlockDataUnit::BlockDataUnit(
    const Pa::BlockData &db, const AST::ParentType &parent)
  : ProgramUnit{&db, parent} {}

AST::Program *Br::createAST(const Pa::Program &root) {
  ASTBuilder walker;
  Walk(root, walker);
  return walker.result();
}

void Br::annotateControl(AST::Program &ast) {
  for (auto &unit : ast.getUnits()) {
    std::visit(common::visitors{
                   [](AST::BlockDataUnit &) {},
                   [](AST::FunctionLikeUnit &f) {
                     annotateFuncCFG(f);
                     for (auto &s : f.funcs) {
                       annotateFuncCFG(s);
                     }
                   },
                   [](AST::ModuleLikeUnit &u) {
                     for (auto &f : u.funcs) {
                       annotateFuncCFG(f);
                     }
                   },
               },
        unit);
  }
}
