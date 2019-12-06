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

#include "bridge.h"
#include "ast-builder.h"
#include "builder.h"
#include "convert-expr.h"
#include "convert-type.h"
#include "intrinsics.h"
#include "io.h"
#include "runtime.h"
#include "../parser/parse-tree.h"
#include "../semantics/tools.h"
#include "fir/FIRDialect.h"
#include "fir/FIROps.h"
#include "fir/FIRType.h"
#include "fir/InternalNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"

#undef TODO
#define TODO() assert(false && "not yet implemented")

#undef SOFT_TODO
#define SOFT_TODO() \
  llvm::errs() << __FILE__ << ":" << __LINE__ << " not yet implemented\n";

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

using SelectCaseConstruct = Pa::CaseConstruct;
using SelectRankConstruct = Pa::SelectRankConstruct;
using SelectTypeConstruct = Pa::SelectTypeConstruct;

using CFGSinkListType = L::SmallVector<AST::Evaluation *, 2>;
using CFGMapType = L::DenseMap<AST::Evaluation *, CFGSinkListType *>;

constexpr static bool isStopStmt(const Pa::StopStmt &stm) {
  return std::get<Pa::StopStmt::Kind>(stm.t) == Pa::StopStmt::Kind::Stop;
}

/// Traverse the AST and complete the CFG by drawing the arcs, pruning unused
/// potential targets, making implied jumps explicit, etc.
class CfgBuilder {

  AST::Evaluation *getEvalByLabel(const Pa::Label &label) {
    auto iter = labels.find(label);
    if (iter != labels.end()) {
      return iter->second;
    }
    return nullptr;
  }

  /// Collect all the potential targets and initialize them to unreferenced
  void resetPotentialTargets(std::list<AST::Evaluation> &evals) {
    for (auto &e : evals) {
      if (e.isTarget) {
        e.isTarget = false;
      }
      if (e.lab.has_value()) {
        labels.try_emplace(*e.lab, &e);
      }
      if (e.subs) {
        resetPotentialTargets(*e.subs);
      }
    }
  }

  /// cache ASSIGN statements that may yield a live branch target
  void cacheAssigns(std::list<AST::Evaluation> &evals) {
    for (auto &e : evals) {
      std::visit(Co::visitors{
                     [&](const Pa::AssignStmt *stmt) {
                       auto *trg = getEvalByLabel(std::get<Pa::Label>(stmt->t));
                       auto *sym = std::get<Pa::Name>(stmt->t).symbol;
                       assert(sym);
                       auto jter = assignedGotoMap.find(sym);
                       if (jter == assignedGotoMap.end()) {
                         std::list<AST::Evaluation *> lst = {trg};
                         assignedGotoMap.try_emplace(sym, lst);
                       } else {
                         jter->second.emplace_back(trg);
                       }
                     },
                     [](auto) { /* do nothing */ },
                 },
          e.u);
      if (e.subs) {
        cacheAssigns(*e.subs);
      }
    }
  }

  void deannotate(std::list<AST::Evaluation> &evals) {
    for (auto &e : evals) {
      e.cfg = AST::CFGAnnotation::None;
      if (e.subs) {
        deannotate(*e.subs);
      }
    }
  }

  bool structuredLoop(const std::optional<Pa::LoopControl> &optLoopCtrl) {
    if (optLoopCtrl.has_value())
      return std::visit(
          Co::visitors{
              [](const Pa::LoopControl::Bounds &) { return true; },
              [](const Pa::ScalarLogicalExpr &) { return false; },
              [](const Pa::LoopControl::Concurrent &) { return true; },
          },
          optLoopCtrl->u);
    return false;
  }

  bool structuredCheck(std::list<AST::Evaluation> &evals) {
    for (auto &e : evals) {
      if (auto **s = std::get_if<const Pa::DoConstruct *>(&e.u)) {
        if (!structuredLoop(std::get<std::optional<Pa::LoopControl>>(
                std::get<Pa::Statement<Pa::NonLabelDoStmt>>((*s)->t)
                    .statement.t)))
          return false;
        return structuredCheck(*e.subs);
      }
      if (std::holds_alternative<const Pa::IfConstruct *>(e.u)) {
        return structuredCheck(*e.subs);
      }
      if (e.subs) {
        return false;
      }
      switch (e.cfg) {
      case AST::CFGAnnotation::None: break;
      case AST::CFGAnnotation::Goto: break;
      case AST::CFGAnnotation::Iterative: break;
      case AST::CFGAnnotation::FirStructuredOp: break;
      case AST::CFGAnnotation::CondGoto:
        if (!std::holds_alternative<const Pa::EndDoStmt *>(e.u)) {
          return false;
        }
        break;
      case AST::CFGAnnotation::IndGoto: return false;
      case AST::CFGAnnotation::IoSwitch: return false;
      case AST::CFGAnnotation::Switch: return false;
      case AST::CFGAnnotation::Return: return false;
      case AST::CFGAnnotation::Terminate: return false;
      }
    }
    return true;
  }

  void wrapIterationSpaces(std::list<AST::Evaluation> &evals) {
    for (auto &e : evals) {
      if (std::holds_alternative<const Pa::DoConstruct *>(e.u))
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = AST::CFGAnnotation::FirStructuredOp;
          continue;
        }
      if (std::holds_alternative<const Pa::IfConstruct *>(e.u))
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = AST::CFGAnnotation::FirStructuredOp;
          continue;
        }
      // FIXME: ForallConstruct? WhereConstruct?
      if (e.subs) {
        wrapIterationSpaces(*e.subs);
      }
    }
  }

  /// Add source->sink edge to CFG map
  void addSourceToSink(AST::Evaluation *src, AST::Evaluation *snk) {
    auto iter = cfgMap.find(src);
    if (iter == cfgMap.end()) {
      CFGSinkListType sink{snk};
      cfgEdgeSetPool.emplace_back(std::move(sink));
      auto rc{cfgMap.try_emplace(src, &cfgEdgeSetPool.back())};
      assert(rc.second && "insert failed unexpectedly");
      (void)rc;  // for release build
      return;
    }
    for (auto *s : *iter->second)
      if (s == snk) {
        return;
      }
    iter->second->push_back(snk);
  }

  void addSourceToSink(AST::Evaluation *src, const Pa::Label &label) {
    auto iter = labels.find(label);
    assert(iter != labels.end());
    addSourceToSink(src, iter->second);
  }

  /// Find the next ELSE IF, ELSE or END IF statement in the list
  template<typename A> A nextFalseTarget(A iter, const A &endi) {
    for (; iter != endi; ++iter)
      if (std::visit(Co::visitors{
                         [&](const Pa::ElseIfStmt *) { return true; },
                         [&](const Pa::ElseStmt *) { return true; },
                         [&](const Pa::EndIfStmt *) { return true; },
                         [](auto) { return false; },
                     },
              iter->u)) {
        break;
      }
    return iter;
  }

  /// Add branches for this IF block like construct.
  /// Branch to the "true block", the "false block", and from the end of the
  /// true block to the end of the construct.
  template<typename A>
  void doNextIfBlock(std::list<AST::Evaluation> &evals, AST::Evaluation &e,
      const A &iter, const A &endif) {
    A i{iter};
    A j{nextFalseTarget(++i, endif)};
    auto *cstr = std::get<AST::Evaluation *>(e.parent);
    AST::CGJump jump{&*endif};
    A k{evals.insert(j, AST::Evaluation{std::move(jump), j->parent})};
    if (i == j) {
      // block was empty, so adjust "true" target
      i = k;
    }
    addSourceToSink(&*k, cstr);
    addSourceToSink(&e, &*i);
    addSourceToSink(&e, &*j);
  }

  /// Determine which branch targets are reachable. The target map must
  /// already be initialized.
  void reachabilityAnalysis(std::list<AST::Evaluation> &evals) {
    for (auto iter = evals.begin(); iter != evals.end(); ++iter) {
      auto &e = *iter;
      switch (e.cfg) {
      case AST::CFGAnnotation::None:
        // do nothing - does not impart control flow
        break;
      case AST::CFGAnnotation::Goto:
        std::visit(
            Co::visitors{
                [&](const Pa::CycleStmt *) {
                  // FIXME: deal with construct name
                  auto *cstr = std::get<AST::Evaluation *>(e.parent);
                  addSourceToSink(&e, &cstr->subs->front());
                },
                [&](const Pa::ExitStmt *) {
                  // FIXME: deal with construct name
                  auto *cstr = std::get<AST::Evaluation *>(e.parent);
                  addSourceToSink(&e, &cstr->subs->back());
                },
                [&](const Pa::GotoStmt *stmt) { addSourceToSink(&e, stmt->v); },
                [&](const Pa::EndDoStmt *) {
                  // the END DO is the loop exit landing pad
                  // insert a JUMP as the backedge right before the END DO
                  auto *cstr = std::get<AST::Evaluation *>(e.parent);
                  AST::CGJump jump{&cstr->subs->front()};
                  AST::Evaluation jumpEval{std::move(jump), iter->parent};
                  evals.insert(iter, std::move(jumpEval));
                  addSourceToSink(&e, &cstr->subs->front());
                },
                [](auto) { assert(false); },
            },
            e.u);
        break;
      case AST::CFGAnnotation::CondGoto:
        std::visit(Co::visitors{
                       [&](const Pa::IfStmt *) {
                         // check if these are marked; they must targets here
                         auto i{iter};
                         addSourceToSink(&e, &*(++i));
                         addSourceToSink(&e, &*(++i));
                       },
                       [&](const Pa::IfThenStmt *) {
                         doNextIfBlock(evals, e, iter, evals.end());
                       },
                       [&](const Pa::ElseIfStmt *) {
                         doNextIfBlock(evals, e, iter, evals.end());
                       },
                       [](const Pa::WhereConstructStmt *stmt) { TODO(); },
                       [](const Pa::MaskedElsewhereStmt *stmt) { TODO(); },
                       [](auto) { assert(false); },
                   },
            e.u);
        break;
      case AST::CFGAnnotation::IndGoto:
        std::visit(
            Co::visitors{
                [&](const Pa::AssignedGotoStmt *stmt) {
                  auto *sym = std::get<Pa::Name>(stmt->t).symbol;
                  if (assignedGotoMap.find(sym) != assignedGotoMap.end())
                    for (auto *x : assignedGotoMap[sym]) {
                      addSourceToSink(&e, x);
                    }
                  for (auto &l : std::get<std::list<Pa::Label>>(stmt->t)) {
                    addSourceToSink(&e, l);
                  }
                },
                [](auto) { assert(false); },
            },
            e.u);
        break;
      case AST::CFGAnnotation::IoSwitch:
        std::visit(Co::visitors{
                       [](const Pa::BackspaceStmt *stmt) { TODO(); },
                       [](const Pa::CloseStmt *stmt) { TODO(); },
                       [](const Pa::EndfileStmt *stmt) { TODO(); },
                       [](const Pa::FlushStmt *stmt) { TODO(); },
                       [](const Pa::InquireStmt *stmt) { TODO(); },
                       [](const Pa::OpenStmt *stmt) { TODO(); },
                       [](const Pa::ReadStmt *stmt) { TODO(); },
                       [](const Pa::RewindStmt *stmt) { TODO(); },
                       [](const Pa::WaitStmt *stmt) { TODO(); },
                       [](const Pa::WriteStmt *stmt) { TODO(); },
                       [](auto) { assert(false); },
                   },
            e.u);
        break;
      case AST::CFGAnnotation::Switch:
        std::visit(Co::visitors{
                       [](const Pa::CallStmt *stmt) { TODO(); },
                       [](const Pa::ArithmeticIfStmt *stmt) { TODO(); },
                       [](const Pa::ComputedGotoStmt *stmt) { TODO(); },
                       [](const Pa::SelectCaseStmt *stmt) { TODO(); },
                       [](const Pa::SelectRankStmt *stmt) { TODO(); },
                       [](const Pa::SelectTypeStmt *stmt) { TODO(); },
                       [](auto) { assert(false); },
                   },
            e.u);
        break;
      case AST::CFGAnnotation::Iterative:
        std::visit(Co::visitors{
                       [](const Pa::NonLabelDoStmt *stmt) { TODO(); },
                       [](const Pa::WhereStmt *stmt) { TODO(); },
                       [](const Pa::ForallStmt *stmt) { TODO(); },
                       [](const Pa::WhereConstruct *stmt) { TODO(); },
                       [](const Pa::ForallConstructStmt *stmt) { TODO(); },
                       [](auto) { assert(false); },
                   },
            e.u);
        break;
      case AST::CFGAnnotation::FirStructuredOp: continue;
      case AST::CFGAnnotation::Return:
        // do nothing - exits the function
        break;
      case AST::CFGAnnotation::Terminate:
        // do nothing - exits the function
        break;
      }
      if (e.subs) {
        reachabilityAnalysis(*e.subs);
      }
    }
  }

  void setActualTargets(std::list<AST::Evaluation> &evals) {
    for (auto &lst1 : cfgEdgeSetPool)
      for (auto *e : lst1) {
        e->isTarget = true;
      }
  }

  CFGMapType &cfgMap;
  std::list<CFGSinkListType> &cfgEdgeSetPool;

  L::DenseMap<Pa::Label, AST::Evaluation *> labels;
  std::map<Se::Symbol *, std::list<AST::Evaluation *>> assignedGotoMap;

public:
  CfgBuilder(CFGMapType &m, std::list<CFGSinkListType> &p)
    : cfgMap{m}, cfgEdgeSetPool{p} {}

  void run(AST::FunctionLikeUnit &func) {
    resetPotentialTargets(func.evals);
    cacheAssigns(func.evals);
    wrapIterationSpaces(func.evals);
    reachabilityAnalysis(func.evals);
    setActualTargets(func.evals);
  }
};

/// Converter from AST to FIR
///
/// After building the AST and decorating it, the FirConverter processes that
/// representation and lowers it to the FIR executable representation.
class FirConverter {
  using LabelMapType = std::map<AST::Evaluation *, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  //
  // Helper function members
  //

  M::Value *createFIRAddr(M::Location loc, const Se::SomeExpr *expr) {
    return createSomeAddress(
        loc, *builder, *expr, symbolMap, defaults, intrinsics);
  }

  M::Value *createFIRExpr(M::Location loc, const Se::SomeExpr *expr) {
    return createSomeExpression(
        loc, *builder, *expr, symbolMap, defaults, intrinsics);
  }
  M::Value *createLogicalExprAsI1(M::Location loc, const Se::SomeExpr *expr) {
    return createI1LogicalExpression(
        loc, *builder, *expr, symbolMap, defaults, intrinsics);
  }

  M::FuncOp genRuntimeFunction(RuntimeEntryCode rec, int kind) {
    return genFunctionFIR(
        getRuntimeEntryName(rec), getRuntimeEntryType(rec, mlirContext, kind));
  }

  M::FuncOp genFunctionFIR(L::StringRef callee, M::FunctionType funcTy) {
    if (auto func{getNamedFunction(module, callee)}) {
      return func;
    }
    return createFunction(module, callee, funcTy);
  }

  bool inMainProgram(AST::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](AST::FunctionLikeUnit *c) { return c->isMainProgram(); },
            [&](AST::Evaluation *c) { return inMainProgram(c); },
            [](auto *) { return false; },
        },
        cstr->parent);
  }
  const Pa::SubroutineStmt *inSubroutine(AST::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](AST::FunctionLikeUnit *c) { return c->isSubroutine(); },
            [&](AST::Evaluation *c) { return inSubroutine(c); },
            [](auto *) -> const Pa::SubroutineStmt * { return nullptr; },
        },
        cstr->parent);
  }
  const Pa::FunctionStmt *inFunction(AST::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](AST::FunctionLikeUnit *c) { return c->isFunction(); },
            [&](AST::Evaluation *c) { return inFunction(c); },
            [](auto *) -> const Pa::FunctionStmt * { return nullptr; },
        },
        cstr->parent);
  }
  const Pa::MpSubprogramStmt *inMPSubp(AST::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](AST::FunctionLikeUnit *c) { return c->isMPSubp(); },
            [&](AST::Evaluation *c) { return inMPSubp(c); },
            [](auto *) -> const Pa::MpSubprogramStmt * { return nullptr; },
        },
        cstr->parent);
  }

  template<typename A>
  const Se::SomeExpr *getScalarExprOfTuple(const A &tuple) {
    return Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple));
  }
  template<typename A> const Se::SomeExpr *getExprOfTuple(const A &tuple) {
    return Se::GetExpr(std::get<Pa::LogicalExpr>(tuple));
  }
  /// Get the condition expression for a CondGoto evaluation
  const Se::SomeExpr *getEvaluationCondition(AST::Evaluation &eval) {
    return std::visit(Co::visitors{
                          [&](const Pa::IfStmt *stmt) {
                            return getScalarExprOfTuple(stmt->t);
                          },
                          [&](const Pa::IfThenStmt *stmt) {
                            return getScalarExprOfTuple(stmt->t);
                          },
                          [&](const Pa::ElseIfStmt *stmt) {
                            return getScalarExprOfTuple(stmt->t);
                          },
                          [&](const Pa::WhereConstructStmt *stmt) {
                            return getExprOfTuple(stmt->t);
                          },
                          [&](const Pa::MaskedElsewhereStmt *stmt) {
                            return getExprOfTuple(stmt->t);
                          },
                          [](auto) -> const Se::SomeExpr * {
                            assert(
                                false && "unexpected conditional branch case");
                            return nullptr;
                          },
                      },
        eval.u);
  }

  //
  // Function-like AST entry and exit statements
  //

  void genFIR(const Pa::Statement<Pa::ProgramStmt> &stmt, std::string &name,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    name = mangler.getProgramEntry();
  }
  void genFIR(const Pa::Statement<Pa::EndProgramStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::FunctionStmt> &stmt, std::string &name,
      Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndFunctionStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::SubroutineStmt> &stmt, std::string &name,
      Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndSubroutineStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::MpSubprogramStmt> &stmt,
      std::string &name, Se::Symbol const *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{stmt.statement.v};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndMpSubprogramStmt> &stmt, std::string &,
      Se::Symbol const *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  //
  // Termination of symbolically referenced execution units
  //

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genFIRProgramExit() { builder->create<M::ReturnOp>(toLocation()); }
  void genFIR(const Pa::EndProgramStmt &) { genFIRProgramExit(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genFIRFunctionReturn(const Pa::FunctionStmt *stmt) {
    auto &name = std::get<Pa::Name>(stmt->t);
    assert(name.symbol);
    const auto &details{name.symbol->get<Se::SubprogramDetails>()};
    M::Value *resultRef{symbolMap.lookupSymbol(details.result())};
    // FIXME: what happens if result was never referenced before and hence no
    // temp was created?
    assert(resultRef);
    M::Value *r{builder->create<fir::LoadOp>(toLocation(), resultRef)};
    builder->create<M::ReturnOp>(toLocation(), r);
  }
  void genFIR(const Pa::EndFunctionStmt &stmt) {
    genFIRFunctionReturn(inFunction(currentEvaluation));
  }
  template<typename A> void genFIRProcedureExit(const A *) {
    // FIXME: alt-returns
    builder->create<M::ReturnOp>(toLocation());
  }
  void genFIR(const Pa::EndSubroutineStmt &stmt) {
    genFIRProcedureExit(inSubroutine(currentEvaluation));
  }
  void genFIR(const Pa::EndMpSubprogramStmt &stmt) {
    genFIRProcedureExit(inMPSubp(currentEvaluation));
  }

  //
  // Statements that have control-flow semantics
  //

  // Conditional goto control-flow semantics
  void genFIREvalCondGoto(AST::Evaluation &eval) {
    genFIR(eval);
    auto targets{findTargetsOf(eval)};
    auto *expr{getEvaluationCondition(eval)};
    assert(expr && "condition expression missing");
    auto *cond{createLogicalExprAsI1(toLocation(), expr)};
    genFIRCondBranch(cond, targets[0], targets[1]);
  }

  void genFIRCondBranch(
      M::Value *cond, AST::Evaluation *trueDest, AST::Evaluation *falseDest) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, M::Value *cnd,
            AST::Evaluation *trueDest, AST::Evaluation *falseDest,
            M::Location location, const LabelMapType &map) {
          L::SmallVector<M::Value *, 2> blk;
          builder->setInsertionPointToEnd(block);
          auto tdp{map.find(trueDest)};
          auto fdp{map.find(falseDest)};
          assert(tdp != map.end() && fdp != map.end());
          builder->create<M::CondBranchOp>(
              location, cnd, tdp->second, blk, fdp->second, blk);
        },
        builder, builder->getInsertionBlock(), cond, trueDest, falseDest,
        toLocation(), _1));
  }

  // Goto control-flow semantics
  //
  // These are unconditional jumps. There is nothing to evaluate.
  void genFIREvalGoto(AST::Evaluation &eval) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, AST::Evaluation *dest,
            M::Location location, const LabelMapType &map) {
          builder->setInsertionPointToEnd(block);
          assert(map.find(dest) != map.end() && "no destination");
          builder->create<M::BranchOp>(location, map.find(dest)->second);
        },
        builder, builder->getInsertionBlock(), findSinkOf(eval), toLocation(),
        _1));
  }

  // Indirect goto control-flow semantics
  //
  // For assigned gotos, which is an obsolescent feature. Lower to a switch.
  void genFIREvalIndGoto(AST::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // IO statements that have control-flow semantics
  //
  // First lower the IO statement and then do the multiway switch op
  void genFIREvalIoSwitch(AST::Evaluation &eval) {
    genFIR(eval);
    genFIRIOSwitch(eval);
  }
  void genFIRIOSwitch(AST::Evaluation &) { TODO(); }

  // Iterative loop control-flow semantics
  void genFIREvalIterative(AST::Evaluation &eval) {
    // FIXME
  }

  void switchInsertionPointToWhere(fir::WhereOp &where) {
    // FIXME
  }
  void switchInsertionPointToOtherwise(fir::WhereOp &where) {
    // FIXME
  }
  template<typename A>
  void handleCondition(fir::WhereOp &where, const A *stmt) {
    auto *cond{createLogicalExprAsI1(
        toLocation(), Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(stmt->t)))};
    where = builder->create<fir::WhereOp>(toLocation(), cond, true);
    switchInsertionPointToWhere(where);
  }

  // Structured control op (fir.loop, fir.where)
  void genFIREvalStructuredOp(AST::Evaluation &eval) {
    // process the list of Evaluations
    assert(eval.subs);
    auto *insPt = builder->getInsertionBlock();

    if (std::holds_alternative<const Pa::DoConstruct *>(eval.u)) {
      // Construct fir.loop
      fir::LoopOp doLoop;
      for (auto &e : *eval.subs) {
        if (auto **s = std::get_if<const Pa::NonLabelDoStmt *>(&e.u)) {
          // do bounds, fir.loop op
          std::visit(
              Co::visitors{
                  [&](const Pa::LoopControl::Bounds &x) {
                    auto *lo =
                        createFIRExpr(toLocation(), Se::GetExpr(x.lower));
                    auto *hi =
                        createFIRExpr(toLocation(), Se::GetExpr(x.upper));
                    L::SmallVector<M::Value *, 1> step;
                    if (x.step.has_value()) {
                      step.emplace_back(
                          createFIRExpr(toLocation(), Se::GetExpr(*x.step)));
                    }
                    doLoop = builder->create<fir::LoopOp>(
                        toLocation(), lo, hi, step);
                  },
                  [](const Pa::ScalarLogicalExpr &) {
                    assert(false && "loop lacks iteration space");
                  },
                  [&](const Pa::LoopControl::Concurrent &x) {
                    // FIXME: can project a multi-dimensional space
                    doLoop = builder->create<fir::LoopOp>(toLocation(),
                        (M::Value *)nullptr, (M::Value *)nullptr,
                        L::ArrayRef<M::Value *>{});
                  },
              },
              std::get<std::optional<Pa::LoopControl>>((*s)->t)->u);
        } else if (std::holds_alternative<const Pa::EndDoStmt *>(e.u)) {
          // close fir.loop op
          builder->clearInsertionPoint();
        } else {
          genFIR(e);
        }
      }
    } else if (std::holds_alternative<const Pa::IfConstruct *>(eval.u)) {
      // Construct fir.where
      fir::WhereOp where;
      bool hasElse = false;
      for (auto &e : *eval.subs) {
        if (auto **s = std::get_if<const Pa::IfThenStmt *>(&e.u)) {
          // fir.where op
          handleCondition(where, *s);
        } else if (auto **s = std::get_if<const Pa::ElseIfStmt *>(&e.u)) {
          // otherwise block, then nested fir.where
          switchInsertionPointToOtherwise(where);
          handleCondition(where, *s);
        } else if (std::holds_alternative<const Pa::ElseStmt *>(e.u)) {
          // otherwise block
          switchInsertionPointToOtherwise(where);
          hasElse = true;
        } else if (std::holds_alternative<const Pa::EndIfStmt *>(e.u)) {
          // close all open fir.where ops
          if (!hasElse) {
            switchInsertionPointToOtherwise(where);
          }
          builder->clearInsertionPoint();
        } else {
          genFIR(e);
        }
      }
    } else {
      assert(false && "not yet implemented");
    }
    builder->setInsertionPointToEnd(insPt);
  }

  // Return from subprogram control-flow semantics
  void genFIREvalReturn(AST::Evaluation &eval) {
    // Handled case-by-case
    // FIXME: think about moving the case code here
  }

  // Multiway switch control-flow semantics
  void genFIREvalSwitch(AST::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // Terminate process control-flow semantics
  //
  // Call a runtime routine that does not return
  void genFIREvalTerminate(AST::Evaluation &eval) {
    genFIR(eval);
    builder->create<fir::UnreachableOp>(toLocation());
  }

  // No control-flow
  void genFIREvalNone(AST::Evaluation &eval) { genFIR(eval); }

  void genFIR(const Pa::CallStmt &stmt) {
    // FIXME handle alternate return
    auto loc{toLocation()};
    (void)loc;
    TODO();
  }
  void genFIR(const Pa::IfStmt &) { TODO(); }
  void genFIR(const Pa::WaitStmt &) { TODO(); }
  void genFIR(const Pa::WhereStmt &) { TODO(); }
  void genFIR(const Pa::ComputedGotoStmt &stmt) {
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto *e1{createFIRExpr(toLocation(), exp)};
    (void)e1;
    TODO();
  }
  void genFIR(const Pa::ForallStmt &) { TODO(); }
  void genFIR(const Pa::ArithmeticIfStmt &stmt) {
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *e1{createFIRExpr(toLocation(), exp)};
    (void)e1;
    TODO();
  }
  void genFIR(const Pa::AssignedGotoStmt &) { TODO(); }

  void genFIR(const Pa::AssociateConstruct &) { TODO(); }
  void genFIR(const Pa::BlockConstruct &) { TODO(); }
  void genFIR(const SelectCaseConstruct &) { TODO(); }
  void genFIR(const Pa::ChangeTeamConstruct &) { TODO(); }
  void genFIR(const Pa::CriticalConstruct &) { TODO(); }
  void genFIR(const Pa::DoConstruct &d) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(d.t)};
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value()) {
      // std::visit([&](const auto &x) { genLoopEnterFIR(x, &ss, stmt.source);
      // }, ctrl->u);
    } else {
      // loop forever (See 11.1.7.4.1, para. 2)
      // pushDoContext(&ss);
    }
    TODO();
  }
  void genFIR(const Pa::IfConstruct &cst) { TODO(); }
  void genFIR(const SelectRankConstruct &) { TODO(); }
  void genFIR(const SelectTypeConstruct &) { TODO(); }
  void genFIR(const Pa::WhereConstruct &) { TODO(); }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(const Pa::ForallConstruct &forall) {
    auto &stmt{std::get<Pa::Statement<Pa::ForallConstructStmt>>(forall.t)};
    setCurrentPosition(stmt.source);
    auto &fas{stmt.statement};
    auto &ctrl{std::get<Co::Indirection<Pa::ConcurrentHeader>>(fas.t).value()};
    (void)ctrl;
    for (auto &s : std::get<std::list<Pa::ForallBodyConstruct>>(forall.t)) {
      std::visit(Co::visitors{
                     [&](const Pa::Statement<Pa::ForallAssignmentStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                     [&](const Pa::Statement<Pa::WhereStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                     [&](const Pa::WhereConstruct &b) { genFIR(b); },
                     [&](const Co::Indirection<Pa::ForallConstruct> &b) {
                       genFIR(b.value());
                     },
                     [&](const Pa::Statement<Pa::ForallStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                 },
          s.u);
    }
    TODO();
  }
  void genFIR(const Pa::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(b); }, s.u);
  }

  void genFIR(const Pa::CompilerDirective &) { TODO(); }
  void genFIR(const Pa::OpenMPConstruct &) { TODO(); }
  void genFIR(const Pa::OmpEndLoopDirective &) { TODO(); }

  void genFIR(const parser::AssociateStmt &) { TODO(); }
  void genFIR(const parser::EndAssociateStmt &) { TODO(); }
  void genFIR(const parser::BlockStmt &) { TODO(); }
  void genFIR(const parser::EndBlockStmt &) { TODO(); }
  void genFIR(const parser::SelectCaseStmt &) { TODO(); }
  void genFIR(const parser::CaseStmt &) { TODO(); }
  void genFIR(const parser::EndSelectStmt &) { TODO(); }
  void genFIR(const parser::ChangeTeamStmt &) { TODO(); }
  void genFIR(const parser::EndChangeTeamStmt &) { TODO(); }
  void genFIR(const parser::CriticalStmt &) { TODO(); }
  void genFIR(const parser::EndCriticalStmt &) { TODO(); }

  // Do loop is handled by EvalIterative(), EvalStructuredOp()
  void genFIR(const parser::NonLabelDoStmt &) {}  // do nothing
  void genFIR(const parser::EndDoStmt &) {}  // do nothing

  // If-Then-Else is handled by EvalCondGoto(), EvalStructuredOp()
  void genFIR(const parser::IfThenStmt &) {}  // do nothing
  void genFIR(const parser::ElseIfStmt &) {}  // do nothing
  void genFIR(const parser::ElseStmt &) {}  // do nothing
  void genFIR(const parser::EndIfStmt &) {}  // do nothing

  void genFIR(const parser::SelectRankStmt &) { TODO(); }
  void genFIR(const parser::SelectRankCaseStmt &) { TODO(); }
  void genFIR(const parser::SelectTypeStmt &) { TODO(); }
  void genFIR(const parser::TypeGuardStmt &) { TODO(); }

  void genFIR(const parser::WhereConstructStmt &) { TODO(); }
  void genFIR(const parser::MaskedElsewhereStmt &) { TODO(); }
  void genFIR(const parser::ElsewhereStmt &) { TODO(); }
  void genFIR(const parser::EndWhereStmt &) { TODO(); }
  void genFIR(const parser::ForallConstructStmt &) { TODO(); }
  void genFIR(const parser::EndForallStmt &) { TODO(); }

  //
  // Statements that do not have control-flow semantics
  //

  void genFIR(const Pa::AllocateStmt &) { TODO(); }
  void genFIR(const Pa::AssignmentStmt &stmt) {
    auto *rhs{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *lhs{Se::GetExpr(std::get<Pa::Variable>(stmt.t))};
    auto loc{toLocation()};
    builder->create<fir::StoreOp>(
        loc, createFIRExpr(loc, rhs), createFIRAddr(loc, lhs));
  }

  void genFIR(const Pa::BackspaceStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::CloseStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::ContinueStmt &) {}  // do nothing
  void genFIR(const Pa::DeallocateStmt &) { TODO(); }
  void genFIR(const Pa::EndfileStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::EventPostStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::EventWaitStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::FlushStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::FormTeamStmt &) { TODO(); }
  void genFIR(const Pa::InquireStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::LockStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::NullifyStmt &) { TODO(); }
  void genFIR(const Pa::OpenStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::PointerAssignmentStmt &) { TODO(); }

  void genFIR(const Pa::PrintStmt &stmt) {
    L::SmallVector<mlir::Value *, 4> args;
    for (auto &item : std::get<std::list<Pa::OutputItem>>(stmt.t)) {
      if (auto *parserExpr{std::get_if<Pa::Expr>(&item.u)}) {
        auto loc{toLocation(parserExpr->source)};
        args.push_back(createFIRExpr(loc, Se::GetExpr(*parserExpr)));
      } else {
        TODO();  // implied do
      }
    }
    genPrintStatement(*builder, toLocation(), args);
  }

  void genFIR(const Pa::ReadStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::RewindStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::SyncAllStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncImagesStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncMemoryStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncTeamStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::UnlockStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(const Pa::WriteStmt &) {
    // call some IO runtime routine(s)
    TODO();
  }
  void genFIR(const Pa::AssignStmt &) { TODO(); }
  void genFIR(const Pa::FormatStmt &) { TODO(); }
  void genFIR(const Pa::EntryStmt &) { TODO(); }
  void genFIR(const Pa::PauseStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::DataStmt &) { TODO(); }
  void genFIR(const Pa::NamelistStmt &) { TODO(); }

  // call FAIL IMAGE in runtime
  void genFIR(const Pa::FailImageStmt &stmt) {
    auto callee{genRuntimeFunction(FIRT_FAIL_IMAGE, 0)};
    L::SmallVector<M::Value *, 1> operands;  // FAIL IMAGE has no args
    builder->create<M::CallOp>(toLocation(), callee, operands);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Pa::StopStmt &stm) {
    auto callee{
        genRuntimeFunction(isStopStmt(stm) ? FIRT_STOP : FIRT_ERROR_STOP,
            defaults.GetDefaultKind(IntegerCat))};
    // 2 args: stop-code-opt, quiet-opt
    L::SmallVector<M::Value *, 8> operands;
    builder->create<M::CallOp>(toLocation(), callee, operands);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(const Pa::ReturnStmt &stmt) {
    if (inMainProgram(currentEvaluation)) {
      builder->create<M::ReturnOp>(toLocation());
    } else if (auto *stmt = inSubroutine(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else if (auto *stmt = inFunction(currentEvaluation)) {
      genFIRFunctionReturn(stmt);
    } else if (auto *stmt = inMPSubp(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else {
      assert(false && "unknown subprogram type");
    }
  }

  // stubs for generic goto statements; see genFIREvalGoto()
  void genFIR(const Pa::CycleStmt &) {}  // do nothing
  void genFIR(const Pa::ExitStmt &) {}  // do nothing
  void genFIR(const Pa::GotoStmt &) {}  // do nothing

  void genFIR(AST::Evaluation &eval) {
    currentEvaluation = &eval;
    std::visit(Co::visitors{
                   [&](const auto *p) { genFIR(*p); },
                   [](const AST::CGJump &) { /* do nothing */ },
               },
        eval.u);
  }

  /// Lower an Evaluation
  ///
  /// If the Evaluation is annotated, we can attempt to lower it by the class of
  /// annotation. Otherwise, attempt to lower the Evaluation on a case-by-case
  /// basis.
  void lowerEval(AST::Evaluation &eval) {
    setCurrentPosition(eval.pos);
    if (eval.isControlTarget()) {
      // start a new block
    }
    switch (eval.cfg) {
    case AST::CFGAnnotation::None: genFIREvalNone(eval); break;
    case AST::CFGAnnotation::Goto: genFIREvalGoto(eval); break;
    case AST::CFGAnnotation::CondGoto: genFIREvalCondGoto(eval); break;
    case AST::CFGAnnotation::IndGoto: genFIREvalIndGoto(eval); break;
    case AST::CFGAnnotation::IoSwitch: genFIREvalIoSwitch(eval); break;
    case AST::CFGAnnotation::Switch: genFIREvalSwitch(eval); break;
    case AST::CFGAnnotation::Iterative: genFIREvalIterative(eval); break;
    case AST::CFGAnnotation::FirStructuredOp:
      genFIREvalStructuredOp(eval);
      break;
    case AST::CFGAnnotation::Return: genFIREvalReturn(eval); break;
    case AST::CFGAnnotation::Terminate: genFIREvalTerminate(eval); break;
    }
  }

  M::FuncOp createNewFunction(L::StringRef name, const Se::Symbol *symbol) {
    // get arguments and return type if any, otherwise just use empty vectors
    L::SmallVector<M::Type, 8> args;
    L::SmallVector<M::Type, 2> results;
    if (symbol) {
      auto *details{symbol->detailsIf<Se::SubprogramDetails>()};
      assert(details && "details for semantics::Symbol must be subprogram");
      for (auto *a : details->dummyArgs()) {
        if (a) {  // nullptr indicates alternate return argument
          auto type{translateSymbolToFIRType(&mlirContext, defaults, *a)};
          args.push_back(fir::ReferenceType::get(type));
        }
      }
      if (details->isFunction()) {
        // FIXME: handle subroutines that return magic values
        auto result{details->result()};
        results.push_back(
            translateSymbolToFIRType(&mlirContext, defaults, result));
      }
    }
    auto funcTy{M::FunctionType::get(args, results, &mlirContext)};
    return createFunction(module, name, funcTy);
  }

  /// Prepare to translate a new function
  void startNewFunction(AST::FunctionLikeUnit &funit, L::StringRef name,
      const Se::Symbol *symbol) {
    M::FuncOp func{getNamedFunction(module, name)};
    if (!func) {
      func = createNewFunction(name, symbol);
    }
    func.addEntryBlock();
    assert(!builder && "expected nullptr");
    builder = new M::OpBuilder(func);
    assert(builder && "OpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    // plumb function's arguments
    if (symbol) {
      auto *entryBlock{&func.front()};
      auto *details{symbol->detailsIf<Se::SubprogramDetails>()};
      assert(details && "details for semantics::Symbol must be subprogram");
      for (const auto &v :
          L::zip(details->dummyArgs(), entryBlock->getArguments())) {
        if (std::get<0>(v)) {
          localSymbols.addSymbol(*std::get<0>(v), std::get<1>(v));
        } else {
          TODO();  // handle alternate return
        }
      }
    }
  }

  void finalizeQueuedEdges() {
    for (auto &edgeFunc : localEdgeQ) {
      edgeFunc(localBlockMap);
    }
    localEdgeQ.clear();
    localBlockMap.clear();
  }

  /// Cleanup after the function has been translated
  void endNewFunction() {
    finalizeQueuedEdges();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure-like construct
  void lowerFunc(AST::FunctionLikeUnit &func, L::ArrayRef<L::StringRef> modules,
      L::Optional<L::StringRef> host = {}) {
    std::string name;
    const Se::Symbol *symbol{nullptr};
    auto size{func.funStmts.size()};

    assert((size == 1 || size == 2) && "ill-formed subprogram");
    if (size == 2) {
      std::visit(
          [&](auto *p) { genFIR(*p, name, symbol); }, func.funStmts.front());
    } else {
      name = mangler.getProgramEntry();
    }

    startNewFunction(func, name, symbol);

    // lower this procedure
    for (auto &e : func.evals) {
      lowerEval(e);
    }
    std::visit(
        [&](auto *p) { genFIR(*p, name, symbol); }, func.funStmts.back());

    endNewFunction();

    // recursively lower internal procedures
    L::Optional<L::StringRef> optName{name};
    for (auto &f : func.funcs) {
      lowerFunc(f, modules, optName);
    }
  }

  void lowerMod(AST::ModuleLikeUnit &mod) {
    // FIXME: build the vector of module names
    std::vector<L::StringRef> moduleName;

    // FIXME: do we need to visit the module statements?
    for (auto &f : mod.funcs) {
      lowerFunc(f, moduleName);
    }
  }

  //
  // Finalization of the CFG structure
  //

  /// Lookup the set of sinks for this source. There must be at least one.
  L::ArrayRef<AST::Evaluation *> findTargetsOf(AST::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert(iter != cfgMap.end());
    return *iter->second;
  }

  /// Lookup the sink for this source. There must be exactly one.
  AST::Evaluation *findSinkOf(AST::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert((iter != cfgMap.end()) && (iter->second->size() == 1));
    return iter->second->front();
  }

  /// prune the CFG for `f`
  void pruneFunc(AST::FunctionLikeUnit &func) {
    // find and cache arcs, etc.
    if (!func.evals.empty()) {
      CfgBuilder{cfgMap, cfgEdgeSetPool}.run(func);
    }

    // do any internal procedures
    for (auto &f : func.funcs) {
      pruneFunc(f);
    }
  }

  void pruneMod(AST::ModuleLikeUnit &mod) {
    for (auto &f : mod.funcs) {
      pruneFunc(f);
    }
  }

  void setCurrentPosition(const Pa::CharBlock &pos) {
    if (pos != Pa::CharBlock{}) {
      currentPosition = pos;
    }
  }

  //
  // Utility methods
  //

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) {
    return parserPosToLoc(mlirContext, cooked, cb);
  }

  M::Location toLocation() { return toLocation(currentPosition); }

  // TODO: should these be moved to convert-expr?
  template<M::CmpIPredicate ICMPOPC>
  M::Value *genCompare(M::Value *lhs, M::Value *rhs) {
    auto lty{lhs->getType()};
    assert(lty == rhs->getType());
    if (lty.isIntOrIndex()) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::LogicalType::kindof(lty.getKind())) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::CharacterType::kindof(lty.getKind())) {
      // FIXME
      // return builder->create<M::CallOp>(lhs->getLoc(), );
    }
    assert(false && "cannot generate operation on this type");
    return {};
  }
  M::Value *genGE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::sge>(lhs, rhs);
  }
  M::Value *genLE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::sle>(lhs, rhs);
  }
  M::Value *genEQ(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::eq>(lhs, rhs);
  }
  M::Value *genAND(M::Value *lhs, M::Value *rhs) {
    return builder->create<M::AndOp>(lhs->getLoc(), lhs, rhs);
  }

private:
  M::MLIRContext &mlirContext;
  const Pa::CookedSource *cooked;
  M::ModuleOp &module;
  Co::IntrinsicTypeDefaultKinds const &defaults;
  IntrinsicLibrary intrinsics;
  M::OpBuilder *builder{nullptr};
  fir::NameMangler &mangler;
  SymMap localSymbols;
  std::list<Closure> localEdgeQ;
  LabelMapType localBlockMap;
  Pa::CharBlock currentPosition;
  CFGMapType cfgMap;
  std::list<CFGSinkListType> cfgEdgeSetPool;
  SymMap symbolMap;
  AST::Evaluation *currentEvaluation;  // FIXME: this is a hack

public:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  explicit FirConverter(BurnsideBridge &bridge, fir::NameMangler &mangler)
    : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
      module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
      intrinsics{IntrinsicLibrary::create(
          IntrinsicLibrary::Version::LLVM, bridge.getMLIRContext())},
      mangler{mangler} {}

  /// Convert the AST to FIR
  void run(AST::Program &ast) {
    // build pruned control
    for (auto &u : ast.getUnits()) {
      std::visit(common::visitors{
                     [&](AST::FunctionLikeUnit &f) { pruneFunc(f); },
                     [&](AST::ModuleLikeUnit &m) { pruneMod(m); },
                     [](AST::BlockDataUnit &) { /* do nothing */ },
                 },
          u);
    }

    // do translation
    for (auto &u : ast.getUnits()) {
      std::visit(common::visitors{
                     [&](AST::FunctionLikeUnit &f) { lowerFunc(f, {}); },
                     [&](AST::ModuleLikeUnit &m) { lowerMod(m); },
                     [](AST::BlockDataUnit &) { SOFT_TODO(); },
                 },
          u);
    }
  }
};

}  // namespace

void Br::BurnsideBridge::lower(
    const Pa::Program &prg, fir::NameMangler &mangler) {
  AST::Program *ast{Br::createAST(prg)};
  Br::annotateControl(*ast);
  FirConverter converter{*this, mangler};
  converter.run(*ast);
  delete ast;
}

void Br::BurnsideBridge::parseSourceFile(L::SourceMgr &srcMgr) {
  auto owningRef = M::parseSourceFile(srcMgr, context.get());
  module.reset(new M::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Br::BurnsideBridge::BurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds,
    const Pa::CookedSource *cooked)
  : defaultKinds{defaultKinds}, cooked{cooked} {
  context = std::make_unique<M::MLIRContext>();
  module = std::make_unique<M::ModuleOp>(
      M::ModuleOp::create(M::UnknownLoc::get(context.get())));
}
