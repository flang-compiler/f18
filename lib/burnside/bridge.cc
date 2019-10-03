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
#include "builder.h"
#include "convert-expr.h"
#include "fe-helper.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "flattened.h"
#include "intrinsics.h"
#include "runtime.h"
#include "../parser/parse-tree-visitor.h"
#include "../semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"

namespace Br = Fortran::burnside;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace Fl = Fortran::burnside::flat;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

namespace {

using SomeExpr = Ev::Expr<Ev::SomeType>;

std::unique_ptr<BurnsideBridge> bridgeInstance;

constexpr bool isStopStmt(Pa::StopStmt::Kind kind) {
  return kind == Pa::StopStmt::Kind::Stop;
}

constexpr bool firLoopOp{false};

/// Converter from Fortran to FIR
class FIRConverter {
  using LabelMapType = std::map<Fl::LabelMention, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  struct DoBoundsInfo {
    M::Value *doVar;

    M::Value *counter;
    M::Value *stepExpr;
    M::Operation *condition;
  };

  M::MLIRContext &mlirContext;
  const Pa::CookedSource *cooked;
  M::ModuleOp &module;
  std::unique_ptr<M::OpBuilder> builder;
  LabelMapType blockMap;  // map from flattened labels to MLIR blocks
  std::list<Closure> edgeQ;
  std::map<const Pa::NonLabelDoStmt *, DoBoundsInfo> doMap;
  SymMap symbolMap;
  IntrinsicLibrary intrinsics;
  Pa::CharBlock lastKnownPos;
  bool noInsPt{false};

  inline M::OpBuilder &build() { return *builder.get(); }
  inline M::ModuleOp &getMod() { return module; }
  inline LabelMapType &blkMap() { return blockMap; }
  void setCurrentPos(const Pa::CharBlock &pos) { lastKnownPos = pos; }

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) {
    return parserPosToLoc(mlirContext, cooked, cb);
  }
  M::Location toLocation() { return toLocation(lastKnownPos); }

  /// Construct the type of an Expr<A> expression
  M::Type exprType(const SomeExpr *expr) {
    return translateSomeExprToFIRType(&mlirContext, expr);
  }
  M::Type refExprType(const SomeExpr *expr) {
    auto type{translateSomeExprToFIRType(&mlirContext, expr)};
    return fir::ReferenceType::get(type);
  }

  int getDefaultIntegerKind() {
    return getDefaultKinds().GetDefaultKind(Co::TypeCategory::Integer);
  }
  M::Type getDefaultIntegerType() {
    return M::IntegerType::get(8 * getDefaultIntegerKind(), &mlirContext);
  }
  int getDefaultLogicalKind() {
    return getDefaultKinds().GetDefaultKind(Co::TypeCategory::Logical);
  }
  M::Type getDefaultLogicalType() {
    return fir::LogicalType::get(&mlirContext, getDefaultLogicalKind());
  }

  M::Value *createFIRAddr(M::Location loc, const SomeExpr *expr) {
    return createSomeAddress(loc, build(), *expr, symbolMap, intrinsics);
  }
  M::Value *createFIRExpr(M::Location loc, const SomeExpr *expr) {
    return createSomeExpression(loc, build(), *expr, symbolMap, intrinsics);
  }
  M::Value *createTemp(M::Type type, Se::Symbol *symbol = nullptr) {
    return createTemporary(toLocation(), build(), symbolMap, type, symbol);
  }

  M::FuncOp genFunctionFIR(llvm::StringRef callee, M::FunctionType funcTy) {
    if (auto func{getNamedFunction(callee)}) {
      return func;
    }
    return createFunction(getMod(), callee, funcTy);
  }

  M::FuncOp genRuntimeFunction(RuntimeEntryCode rec, int kind) {
    return genFunctionFIR(
        getRuntimeEntryName(rec), getRuntimeEntryType(rec, mlirContext, kind));
  }

  template<typename T> DoBoundsInfo *getBoundsInfo(const T &linearOp) {
    auto &st{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(linearOp.v->t)};
    setCurrentPos(st.source);
    auto *s{&st.statement};
    auto iter{doMap.find(s)};
    if (iter != doMap.end()) {
      return &iter->second;
    }
    assert(false && "DO context not present");
    return nullptr;
  }

  // Simple scalar expression builders
  // TODO: handle REAL and COMPLEX (iff needed)
  template<M::CmpIPredicate ICMPOPC>
  M::Value *genCompare(M::Value *lhs, M::Value *rhs) {
    auto lty{lhs->getType()};
    assert(lty == rhs->getType());
    if (lty.isIntOrIndex()) {
      return build().create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::LogicalType::kindof(lty.getKind())) {
      return build().create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (fir::CharacterType::kindof(lty.getKind())) {
      // return build().create<M::CallOp>(lhs->getLoc(), );
      return {};
    }
    assert(false && "cannot generate operation on this type");
    return {};
  }
  M::Value *genGE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::SGE>(lhs, rhs);
  }
  M::Value *genLE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::SLE>(lhs, rhs);
  }
  M::Value *genEQ(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::EQ>(lhs, rhs);
  }
  M::Value *genAND(M::Value *lhs, M::Value *rhs) {
    return build().create<M::AndOp>(lhs->getLoc(), lhs, rhs);
  }

  void genFIR(AnalysisData &ad, std::list<Fl::Op> &operations);

  // Control flow destination
  void genFIR(bool lastWasLabel, const Fl::LabelOp &op) {
    if (lastWasLabel) {
      blkMap().insert({op.get(), build().getInsertionBlock()});
    } else {
      auto *currBlock{build().getInsertionBlock()};
      auto *newBlock{createBlock(&build())};
      blkMap().insert({op.get(), newBlock});
      if (!noInsPt) {
        build().setInsertionPointToEnd(currBlock);
        build().create<M::BranchOp>(toLocation(), newBlock);
      }
      build().setInsertionPointToStart(newBlock);
    }
  }

  // Goto statements
  void genFIR(const Fl::GotoOp &op) {
    auto iter{blkMap().find(op.target)};
    if (iter != blkMap().end()) {
      build().create<M::BranchOp>(toLocation(), iter->second);
    } else {
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](M::OpBuilder *builder, M::Block *block, Fl::LabelMention dest,
              M::Location location, const LabelMapType &map) {
            builder->setInsertionPointToEnd(block);
            assert(map.find(dest) != map.end() && "no destination");
            builder->create<M::BranchOp>(location, map.find(dest)->second);
          },
          &build(), build().getInsertionBlock(), op.target, toLocation(), _1));
    }
    noInsPt = true;
  }
  void genFIR(const Fl::ReturnOp &op) {
    std::visit([&](const auto *stmt) { genFIR(*stmt); }, op.u);
    noInsPt = true;
  }
  void genFIR(const Fl::ConditionalGotoOp &op) {
    std::visit(
        [&](const auto *stmt) { genFIR(*stmt, op.trueLabel, op.falseLabel); },
        op.u);
    noInsPt = true;
  }

  void genFIR(const Fl::SwitchIOOp &op);

  // CALL with alt-return value returned
  void genFIR(const Fl::SwitchOp &op, const Pa::CallStmt &stmt) {
    auto loc{toLocation(op.source)};
    // FIXME
    (void)loc;
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::ComputedGotoStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::ArithmeticIfStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  M::Value *fromCaseValue(const M::Location &locs, const Pa::CaseValue &val) {
    return createFIRExpr(locs, Se::GetExpr(val));
  }
  void genFIR(const Fl::SwitchOp &op, const Pa::CaseConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op, const Pa::SelectRankConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op, const Pa::SelectTypeConstruct &stmt);
  void genFIR(const Fl::SwitchOp &op) {
    std::visit([&](auto *construct) { genFIR(op, *construct); }, op.u);
    noInsPt = true;
  }

  void genFIR(AnalysisData &ad, const Fl::ActionOp &op);

  void pushDoContext(const Pa::NonLabelDoStmt *doStmt,
      M::Value *doVar = nullptr, M::Value *counter = nullptr,
      M::Value *stepExpr = nullptr) {
    doMap.emplace(doStmt, DoBoundsInfo{doVar, counter, stepExpr});
  }

  void genLoopEnterFIR(const Pa::LoopControl::Bounds &bounds,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    auto loc{toLocation(source)};
    // evaluate e1, e2 [, e3] ...
    auto *e1{createFIRExpr(loc, Se::GetExpr(bounds.lower))};
    auto *e2{createFIRExpr(loc, Se::GetExpr(bounds.upper))};
    if (firLoopOp) {
      std::vector<M::Value *> step;
      if (bounds.step.has_value())
        step.push_back(createFIRExpr(loc, Se::GetExpr(bounds.step)));
      auto loopOp{build().create<fir::LoopOp>(loc, e1, e2, step)};
      auto *block = createBlock(&build(), &loopOp.getOperation()->getRegion(0));
      block->addArgument(M::IndexType::get(build().getContext()));
      return;
    }
    auto *nameExpr{bounds.name.thing.symbol};
    auto *name{createTemp(getDefaultIntegerType(), nameExpr)};
    M::Value *e3;
    if (bounds.step.has_value()) {
      auto *stepExpr{Se::GetExpr(bounds.step)};
      e3 = createFIRExpr(loc, stepExpr);
    } else {
      auto attr{build().getIntegerAttr(e2->getType(), 1)};
      e3 = build().create<M::ConstantOp>(loc, attr);
    }
    // name <- e1
    build().create<fir::StoreOp>(loc, e1, name);
    auto tripCounter{createTemp(getDefaultIntegerType())};
    // See 11.1.7.4.1, para. 1, item (3)
    // totalTrips ::= iteration count = a
    //   where a = (e2 - e1 + e3) / e3 if a > 0 and 0 otherwise
    auto c1{build().create<M::SubIOp>(loc, e2, e1)};
    auto c2{build().create<M::AddIOp>(loc, c1.getResult(), e3)};
    auto c3{build().create<M::DivISOp>(loc, c2.getResult(), e3)};
    auto *totalTrips{c3.getResult()};
    build().create<fir::StoreOp>(loc, totalTrips, tripCounter);
    pushDoContext(stmt, name, tripCounter, e3);
  }

  void genLoopEnterFIR(const Pa::ScalarLogicalExpr &logicalExpr,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.1, para. 2
    // See BuildLoopLatchExpression()
    pushDoContext(stmt);
  }

  void genLoopEnterFIR(const Pa::LoopControl::Concurrent &concurrent,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.2
    // FIXME
  }

  void genEnterFIR(const Pa::DoConstruct &construct) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    setCurrentPos(stmt.source);
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value()) {
      std::visit([&](const auto &x) { genLoopEnterFIR(x, &ss, stmt.source); },
          ctrl->u);
    } else {
      // loop forever (See 11.1.7.4.1, para. 2)
      pushDoContext(&ss);
    }
  }
  template<typename A> void genEnterFIR(const A &construct) {
    // FIXME: add other genEnterFIR() members
  }
  void genFIR(const Fl::BeginOp &op) {
    std::visit([&](auto *construct) { genEnterFIR(*construct); }, op.u);
  }

  void genExitFIR(const Pa::DoConstruct &construct) {
    if (firLoopOp) {
      build().setInsertionPointAfter(
          build().getBlock()->getParent()->getParentOp());
      return;
    }
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    setCurrentPos(stmt.source);
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(ss.t)};
    if (ctrl.has_value() &&
        std::holds_alternative<parser::LoopControl::Bounds>(ctrl->u)) {
      doMap.erase(&ss);
    }
    noInsPt = true;  // backedge already processed
  }

  void genFIR(const Fl::EndOp &op) {
    if (auto *construct{std::get_if<const Pa::DoConstruct *>(&op.u)})
      genExitFIR(**construct);
  }

  void genFIR(AnalysisData &ad, const Fl::IndirectGotoOp &op);

  void genFIR(const Fl::DoIncrementOp &op) {
    if (firLoopOp) {
      return;
    }
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: do_var = do_var + e3
      auto load{
          build().create<fir::LoadOp>(info->doVar->getLoc(), info->doVar)};
      auto incremented{build().create<M::AddIOp>(
          load.getLoc(), load.getResult(), info->stepExpr)};
      build().create<fir::StoreOp>(load.getLoc(), incremented, info->doVar);
      // add: counter--
      auto loadCtr{
          build().create<fir::LoadOp>(info->counter->getLoc(), info->counter)};
      auto one{build().create<M::ConstantOp>(
          loadCtr.getLoc(), build().getIntegerAttr(loadCtr.getType(), 1))};
      auto decremented{build().create<M::SubIOp>(
          loadCtr.getLoc(), loadCtr.getResult(), one)};
      build().create<fir::StoreOp>(
          loadCtr.getLoc(), decremented, info->counter);
    }
  }

  void genFIR(const Fl::DoCompareOp &op) {
    if (firLoopOp) {
      return;
    }
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: cond = counter > 0 (signed)
      auto load{
          build().create<fir::LoadOp>(info->counter->getLoc(), info->counter)};
      auto zero{build().create<M::ConstantOp>(
          load.getLoc(), build().getIntegerAttr(load.getType(), 0))};
      auto cond{build().create<M::CmpIOp>(
          load.getLoc(), M::CmpIPredicate::SGT, load, zero)};
      info->condition = cond;
    }
  }

  void genFIR(const Pa::FailImageStmt &stmt) {
    auto callee{genRuntimeFunction(FIRT_FAIL_IMAGE, 0)};
    llvm::SmallVector<M::Value *, 1> operands;  // FAIL IMAGE has no args
    build().create<M::CallOp>(toLocation(), callee, operands);
    build().create<fir::UnreachableOp>(toLocation());
  }
  void genFIR(const Pa::ReturnStmt &stmt) {
    build().create<M::ReturnOp>(toLocation());  // FIXME: argument(s)?
  }
  void genFIR(const Pa::StopStmt &stmt) {
    auto callee{genRuntimeFunction(
        isStopStmt(std::get<Pa::StopStmt::Kind>(stmt.t)) ? FIRT_STOP
                                                         : FIRT_ERROR_STOP,
        getDefaultIntegerKind())};
    // 2 args: stop-code-opt, quiet-opt
    llvm::SmallVector<M::Value *, 8> operands;
    build().create<M::CallOp>(toLocation(), callee, operands);
    build().create<fir::UnreachableOp>(toLocation());
  }

  // Conditional branch-like statements
  template<typename A>
  void genFIR(
      const A &tuple, Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    auto *exprRef{Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple))};
    assert(exprRef && "condition expression missing");
    auto *cond{createFIRExpr(toLocation(), exprRef)};
    genCondBranch(cond, trueLabel, falseLabel);
  }
  void genFIR(const Pa::Statement<Pa::IfThenStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    genFIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genFIR(const Pa::Statement<Pa::ElseIfStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    genFIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genFIR(const Pa::IfStmt &stmt, Fl::LabelMention trueLabel,
      Fl::LabelMention falseLabel) {
    genFIR(stmt.t, trueLabel, falseLabel);
  }

  M::Value *getTrueConstant() {
    auto attr{build().getBoolAttr(true)};
    return build().create<M::ConstantOp>(toLocation(), attr).getResult();
  }

  // Conditional branch to enter loop body or exit
  void genFIR(const Pa::Statement<Pa::NonLabelDoStmt> &stmt,
      Fl::LabelMention trueLabel, Fl::LabelMention falseLabel) {
    setCurrentPos(stmt.source);
    auto &loopCtrl{std::get<std::optional<Pa::LoopControl>>(stmt.statement.t)};
    M::Value *condition{nullptr};
    bool exitNow{false};
    if (loopCtrl.has_value()) {
      exitNow = std::visit(
          Co::visitors{
              [&](const parser::LoopControl::Bounds &) {
                if (firLoopOp) {
                  return true;
                }
                auto iter{doMap.find(&stmt.statement)};
                assert(iter != doMap.end());
                condition = iter->second.condition->getResult(0);
                return false;
              },
              [&](const parser::ScalarLogicalExpr &logical) {
                auto loc{toLocation(stmt.source)};
                auto *exp{Se::GetExpr(logical)};
                condition = createFIRExpr(loc, exp);
                return false;
              },
              [&](const parser::LoopControl::Concurrent &concurrent) {
                // FIXME: incorrectly lowering DO CONCURRENT
                condition = getTrueConstant();
                return false;
              },
          },
          loopCtrl->u);
      if (firLoopOp && exitNow) {
        return;
      }
    } else {
      condition = getTrueConstant();
    }
    assert(condition && "condition must be a Value");
    genCondBranch(condition, trueLabel, falseLabel);
  }

  // Action statements
  void genFIR(const Pa::AllocateStmt &stmt);
  void genFIR(const Pa::AssignmentStmt &stmt) {
    auto *rhs{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *lhs{Se::GetExpr(std::get<Pa::Variable>(stmt.t))};
    auto loc{toLocation()};
    build().create<fir::StoreOp>(
        loc, createFIRExpr(loc, rhs), createFIRAddr(loc, lhs));
  }
  void genFIR(const Pa::BackspaceStmt &stmt);
  void genFIR(const Pa::CallStmt &stmt);
  void genFIR(const Pa::CloseStmt &stmt);
  void genFIR(const Pa::DeallocateStmt &stmt);
  void genFIR(const Pa::EndfileStmt &stmt);
  void genFIR(const Pa::EventPostStmt &stmt);
  void genFIR(const Pa::EventWaitStmt &stmt);
  void genFIR(const Pa::FlushStmt &stmt);
  void genFIR(const Pa::FormTeamStmt &stmt);
  void genFIR(const Pa::InquireStmt &stmt);
  void genFIR(const Pa::LockStmt &stmt);
  void genFIR(const Pa::NullifyStmt &stmt);
  void genFIR(const Pa::OpenStmt &stmt);
  void genFIR(const Pa::PointerAssignmentStmt &stmt);
  void genFIR(const Pa::PrintStmt &stmt);
  void genFIR(const Pa::ReadStmt &stmt);
  void genFIR(const Pa::RewindStmt &stmt);
  void genFIR(const Pa::SyncAllStmt &stmt);
  void genFIR(const Pa::SyncImagesStmt &stmt);
  void genFIR(const Pa::SyncMemoryStmt &stmt);
  void genFIR(const Pa::SyncTeamStmt &stmt);
  void genFIR(const Pa::UnlockStmt &stmt);
  void genFIR(const Pa::WaitStmt &stmt);
  void genFIR(const Pa::WhereStmt &stmt);
  void genFIR(const Pa::WriteStmt &stmt);
  void genFIR(const Pa::ForallStmt &stmt);
  void genFIR(AnalysisData &ad, const Pa::AssignStmt &stmt);
  void genFIR(const Pa::PauseStmt &stmt);

  template<typename A>
  void translateRoutine(
      const A &routine, llvm::StringRef name, const Se::Symbol *funcSym);

  void genCondBranch(
      M::Value *cond, Fl::LabelMention trueBlock, Fl::LabelMention falseBlock) {
    auto trueIter{blkMap().find(trueBlock)};
    auto falseIter{blkMap().find(falseBlock)};
    if (trueIter != blkMap().end() && falseIter != blkMap().end()) {
      llvm::SmallVector<M::Value *, 2> blanks;
      build().create<M::CondBranchOp>(toLocation(), cond, trueIter->second,
          blanks, falseIter->second, blanks);
    } else {
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](M::OpBuilder *builder, M::Block *block, M::Value *cnd,
              Fl::LabelMention trueDest, Fl::LabelMention falseDest,
              M::Location location, const LabelMapType &map) {
            llvm::SmallVector<M::Value *, 2> blk;
            builder->setInsertionPointToEnd(block);
            auto tdp{map.find(trueDest)};
            auto fdp{map.find(falseDest)};
            assert(tdp != map.end() && fdp != map.end());
            builder->create<M::CondBranchOp>(
                location, cnd, tdp->second, blk, fdp->second, blk);
          },
          &build(), build().getInsertionBlock(), cond, trueBlock, falseBlock,
          toLocation(), _1));
    }
  }

  template<typename A>
  void genSwitchBranch(const M::Location &loc, M::Value *selector,
      std::list<typename A::Conditions> &&conditions,
      const std::vector<Fl::LabelMention> &labels) {
    assert(conditions.size() == labels.size());
    bool haveAllLabels{true};
    std::size_t u{0};
    // do we already have all the targets?
    for (auto last{labels.size()}; u != last; ++u) {
      haveAllLabels = blkMap().find(labels[u]) != blkMap().end();
      if (!haveAllLabels) break;
    }
    if (haveAllLabels) {
      // yes, so generate the FIR operation now
      u = 0;
      std::vector<M::Value *> conds;
      std::vector<M::Block *> blocks;
      std::vector<llvm::ArrayRef<M::Value *>> blockArgs;
      llvm::SmallVector<M::Value *, 2> blanks;
      for (auto cond : conditions) {
        conds.emplace_back(cond);
        blocks.emplace_back(blkMap().find(labels[u++])->second);
        blockArgs.emplace_back(blanks);
      }
      build().create<A>(loc, selector, conds, blocks, blockArgs);
    } else {
      // no, so queue the FIR operation for later
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](M::OpBuilder *builder, M::Block *block, M::Value *sel,
              const std::list<typename A::Conditions> &conditions,
              const std::vector<Fl::LabelMention> &labels, M::Location location,
              const LabelMapType &map) {
            std::size_t u{0};
            std::vector<M::Value *> conds;
            std::vector<M::Block *> blocks;
            std::vector<llvm::ArrayRef<M::Value *>> blockArgs;
            llvm::SmallVector<M::Value *, 2> blanks;
            for (auto &cond : conditions) {
              auto iter{map.find(labels[u++])};
              assert(iter != map.end());
              conds.emplace_back(cond);
              blocks.emplace_back(iter->second);
              blockArgs.emplace_back(blanks);
            }
            builder->setInsertionPointToEnd(block);
            builder->create<A>(location, sel, conds, blocks, blockArgs);
          },
          &build(), build().getInsertionBlock(), selector, conditions, labels,
          loc, _1));
    }
  }

  void finalizeQueued() {
    for (auto &edgeFunc : edgeQ) {
      edgeFunc(blkMap());
    }
  }

public:
  FIRConverter(BurnsideBridge &bridge)
    : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
      module{bridge.getModule()}, intrinsics{IntrinsicLibrary::create(
                                      IntrinsicLibrary::Version::LLVM,
                                      mlirContext)} {}
  FIRConverter() = delete;

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  /// Translate the various routines from the parse tree
  void Post(const Pa::MainProgram &mainp) {
    std::string mainName{"_MAIN"s};
    if (auto &ps{
            std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(mainp.t)}) {
      mainName = ps->statement.v.ToString();
      setCurrentPos(ps->source);
    }
    translateRoutine(mainp, mainName, nullptr);
  }
  void Post(const Pa::FunctionSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::FunctionStmt>>(subp.t)};
    setCurrentPos(stmt.source);
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
  void Post(const Pa::SubroutineSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::SubroutineStmt>>(subp.t)};
    setCurrentPos(stmt.source);
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
};

/// SELECT CASE
/// Build a switch-like structure for a SELECT CASE
void FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::CaseConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &cstm{std::get<Pa::Statement<Pa::SelectCaseStmt>>(stmt.t)};
  auto *exp{Se::GetExpr(std::get<Pa::Scalar<Pa::Expr>>(cstm.statement.t))};
  auto *e1{createFIRExpr(loc, exp)};
  auto &cases{std::get<std::list<Pa::CaseConstruct::Case>>(stmt.t)};
  std::list<fir::SelectCaseOp::Conditions> conds;
  // Per C1145, we know each `case-expr` must have type INTEGER, CHARACTER, or
  // LOGICAL
  for (auto &sel : cases) {
    auto &cs{std::get<Pa::Statement<Pa::CaseStmt>>(sel.t)};
    auto locs{toLocation(cs.source)};
    auto &csel{std::get<Pa::CaseSelector>(cs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const std::list<Pa::CaseValueRange> &ranges) {
              for (auto &r : ranges) {
                std::visit(Co::visitors{
                               [&](const Pa::CaseValue &val) {
                                 auto *term{fromCaseValue(locs, val)};
                                 conds.emplace_back(genEQ(e1, term));
                               },
                               [&](const Pa::CaseValueRange::Range &rng) {
                                 fir::SelectCaseOp::Conditions rangeComparison =
                                     nullptr;
                                 if (rng.lower.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.lower)};
                                   // rc = e1 >= lower.term
                                   rangeComparison = genGE(e1, term);
                                 }
                                 if (rng.upper.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.upper)};
                                   // c = e1 <= upper.term
                                   auto *comparison{genLE(e1, term)};
                                   // rc = if rc then (rc && c) else c
                                   if (rangeComparison) {
                                     rangeComparison =
                                         genAND(rangeComparison, comparison);
                                   } else {
                                     rangeComparison = comparison;
                                   }
                                 }
                                 conds.emplace_back(rangeComparison);
                               },
                           },
                    r.u);
              }
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        csel.u);
  }
  genSwitchBranch<fir::SelectCaseOp>(loc, e1, std::move(conds), op.refs);
}

/// SELECT RANK
/// Build a switch-like structure for a SELECT RANK
void FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::SelectRankConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &rstm{std::get<Pa::Statement<Pa::SelectRankStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(rstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &ranks{std::get<std::list<Pa::SelectRankConstruct::RankCase>>(stmt.t)};
  std::list<fir::SelectRankOp::Conditions> conds;
  for (auto &r : ranks) {
    auto &rs{std::get<Pa::Statement<Pa::SelectRankCaseStmt>>(r.t)};
    auto &rank{std::get<Pa::SelectRankCaseStmt::Rank>(rs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::ScalarIntConstantExpr &ex) {
              auto *ie{createFIRExpr(loc, Se::GetExpr(ex))};
              conds.emplace_back(ie);
            },
            [&](const Pa::Star &) {
              // FIXME: using a bogon for now.  Special value per
              // whatever the runtime returns.
              auto attr{build().getIntegerAttr(e1->getType(), -1)};
              conds.emplace_back(build().create<M::ConstantOp>(loc, attr));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        rank.u);
  }
  // FIXME: fix the type of the function
  auto callee{genRuntimeFunction(FIRT_GET_RANK, 0)};
  llvm::SmallVector<M::Value *, 1> operands{e1};
  auto e3{build().create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<fir::SelectRankOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

/// SELECT TYPE
/// Build a switch-like structure for a SELECT TYPE
void FIRConverter::genFIR(
    const Fl::SwitchOp &op, const Pa::SelectTypeConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &tstm{std::get<Pa::Statement<Pa::SelectTypeStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(tstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &types{std::get<std::list<Pa::SelectTypeConstruct::TypeCase>>(stmt.t)};
  std::list<fir::SelectTypeOp::Conditions> conds;
  for (auto &t : types) {
    auto &ts{std::get<Pa::Statement<Pa::TypeGuardStmt>>(t.t)};
    auto &ty{std::get<Pa::TypeGuardStmt::Guard>(ts.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::TypeSpec &) {
              // FIXME: add arguments
              auto func{genRuntimeFunction(FIRT_ISA_TYPE, 0)};
              llvm::SmallVector<M::Value *, 2> operands;
              auto call{build().create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::DerivedTypeSpec &) {
              // FIXME: add arguments
              auto func{genRuntimeFunction(FIRT_ISA_SUBTYPE, 0)};
              llvm::SmallVector<M::Value *, 2> operands;
              auto call{build().create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        ty.u);
  }
  auto callee{genRuntimeFunction(FIRT_GET_ELETYPE, 0)};
  llvm::SmallVector<M::Value *, 1> operands{e1};
  auto e3{build().create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<fir::SelectTypeOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

void FIRConverter::genFIR(const Fl::SwitchIOOp &op) {}

void FIRConverter::genFIR(const Pa::AllocateStmt &stmt) {}
void FIRConverter::genFIR(const Pa::BackspaceStmt &stmt) {
  // builder->create<IOCallOp>(stmt.v);
}
void FIRConverter::genFIR(const Pa::CallStmt &stmt) {}
void FIRConverter::genFIR(const Pa::CloseStmt &stmt) {}
void FIRConverter::genFIR(const Pa::DeallocateStmt &stmt) {}
void FIRConverter::genFIR(const Pa::EndfileStmt &stmt) {}
void FIRConverter::genFIR(const Pa::EventPostStmt &stmt) {}
void FIRConverter::genFIR(const Pa::EventWaitStmt &stmt) {}
void FIRConverter::genFIR(const Pa::FlushStmt &stmt) {}
void FIRConverter::genFIR(const Pa::FormTeamStmt &stmt) {}
void FIRConverter::genFIR(const Pa::InquireStmt &stmt) {}
void FIRConverter::genFIR(const Pa::LockStmt &stmt) {}
void FIRConverter::genFIR(const Pa::NullifyStmt &stmt) {}
void FIRConverter::genFIR(const Pa::OpenStmt &stmt) {}
void FIRConverter::genFIR(const Pa::PointerAssignmentStmt &stmt) {}
void FIRConverter::genFIR(const Pa::PrintStmt &stmt) {}
void FIRConverter::genFIR(const Pa::ReadStmt &stmt) {}
void FIRConverter::genFIR(const Pa::RewindStmt &stmt) {}
void FIRConverter::genFIR(const Pa::SyncAllStmt &stmt) {}
void FIRConverter::genFIR(const Pa::SyncImagesStmt &stmt) {}
void FIRConverter::genFIR(const Pa::SyncMemoryStmt &stmt) {}
void FIRConverter::genFIR(const Pa::SyncTeamStmt &stmt) {}
void FIRConverter::genFIR(const Pa::UnlockStmt &stmt) {}
void FIRConverter::genFIR(const Pa::WaitStmt &stmt) {}
void FIRConverter::genFIR(const Pa::WhereStmt &stmt) {}
void FIRConverter::genFIR(const Pa::WriteStmt &stmt) {}
void FIRConverter::genFIR(const Pa::ForallStmt &stmt) {}
void FIRConverter::genFIR(AnalysisData &ad, const Pa::AssignStmt &stmt) {}
void FIRConverter::genFIR(const Pa::PauseStmt &stmt) {}

/// translate action statements
void FIRConverter::genFIR(AnalysisData &ad, const Fl::ActionOp &op) {
  setCurrentPos(op.v->source);
  std::visit(
      Co::visitors{
          [](const Pa::ContinueStmt &) { assert(false); },
          [](const Pa::FailImageStmt &) { assert(false); },
          [](const Co::Indirection<Pa::ArithmeticIfStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::AssignedGotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ComputedGotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::CycleStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ExitStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::GotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::IfStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ReturnStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::StopStmt> &) { assert(false); },
          [&](const Co::Indirection<Pa::AssignStmt> &assign) {
            genFIR(ad, assign.value());
          },
          [&](const auto &stmt) { genFIR(stmt.value()); },
      },
      op.v->statement.u);
}

void FIRConverter::genFIR(AnalysisData &ad, const Fl::IndirectGotoOp &op) {
  // add or queue an igoto
}

void FIRConverter::genFIR(AnalysisData &ad, std::list<Fl::Op> &operations) {
  bool lastWasLabel{false};
  for (auto &op : operations) {
    std::visit(Co::visitors{
                   [&](const Fl::IndirectGotoOp &oper) {
                     genFIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::ActionOp &oper) {
                     noInsPt = false;
                     genFIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::LabelOp &oper) {
                     genFIR(lastWasLabel, oper);
                     lastWasLabel = true;
                   },
                   [&](const Fl::BeginOp &oper) {
                     noInsPt = false;
                     genFIR(oper);
                     lastWasLabel = true;
                   },
                   [&](const auto &oper) {
                     noInsPt = false;
                     genFIR(oper);
                     lastWasLabel = false;
                   },
               },
        op.u);
  }
  if (build().getInsertionBlock()) {
    // FIXME: assuming type of '() -> ()'
    build().create<M::ReturnOp>(toLocation());
  }
}

/// Translate the routine to MLIR
template<typename A>
void FIRConverter::translateRoutine(
    const A &routine, llvm::StringRef name, const Se::Symbol *funcSym) {
  M::FuncOp func{getNamedFunction(name)};
  if (!func) {
    // get arguments and return type if any, otherwise just use empty vectors
    llvm::SmallVector<M::Type, 8> args;
    llvm::SmallVector<M::Type, 2> results;
    if (funcSym) {
      if (auto *details{funcSym->detailsIf<Se::SubprogramDetails>()}) {
        for (auto a : details->dummyArgs()) {
          auto type{translateSymbolToFIRType(&mlirContext, a)};
          args.push_back(fir::ReferenceType::get(type));
        }
        if (details->isFunction()) {
          // FIXME: handle subroutines that return magic values
          auto *result{&details->result()};
          results.push_back(translateSymbolToFIRType(&mlirContext, result));
        }
      } else {
        llvm::errs() << "Symbol: " << funcSym->name().ToString() << " @ "
                     << funcSym->details().index() << '\n';
        assert(false && "symbol misidentified by front-end");
      }
    }
    auto funcTy{M::FunctionType::get(args, results, &mlirContext)};
    func = createFunction(getMod(), name, funcTy);
  }
  func.addEntryBlock();
  builder = std::make_unique<M::OpBuilder>(func);
  build().setInsertionPointToStart(&func.front());
  if (funcSym) {
    auto *entryBlock{&func.front()};
    if (auto *details{funcSym->detailsIf<Se::SubprogramDetails>()}) {
      for (const auto &v :
          llvm::zip(details->dummyArgs(), entryBlock->getArguments())) {
        symbolMap.addSymbol(std::get<0>(v), std::get<1>(v));
      }
    } else {
      llvm::errs() << "Symbol: " << funcSym->name().ToString() << " @ "
                   << funcSym->details().index() << '\n';
      assert(false && "symbol misidentified by front-end");
    }
  }
  AnalysisData ad;
  std::list<Fl::Op> operations;
  CreateFlatIR(routine, operations, ad);
  genFIR(ad, operations);
  finalizeQueued();
}

M::DialectRegistration<fir::FIROpsDialect> FIROps;

}  // namespace

void Br::crossBurnsideBridge(BurnsideBridge &bridge, const Pa::Program &prg) {
  FIRConverter converter{bridge};
  Walk(prg, converter);
}

std::unique_ptr<llvm::Module> Br::LLVMBridge(M::ModuleOp &module) {
  return M::translateModuleToLLVMIR(module);
}

void Br::BurnsideBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  auto owningRef = M::parseSourceFile(srcMgr, context.get());
  module.reset(new M::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
  if (validModule()) {
    // symbols are added by ModuleManager ctor
    manager.reset(new M::ModuleManager(getModule()));
  }
}

Br::BurnsideBridge::BurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds,
    const Pa::CookedSource *cooked)
  : defaultKinds{defaultKinds}, cooked{cooked} {
  context = std::make_unique<M::MLIRContext>();
  module = std::make_unique<M::ModuleOp>(
      M::ModuleOp::create(M::UnknownLoc::get(context.get())));
  manager = std::make_unique<M::ModuleManager>(getModule());
}

void Br::instantiateBurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds,
    const Pa::CookedSource *cooked) {
  auto p{BurnsideBridge::create(defaultKinds, cooked)};
  bridgeInstance.swap(p);
}

BurnsideBridge &Br::getBridge() { return *bridgeInstance.get(); }

const common::IntrinsicTypeDefaultKinds &Br::getDefaultKinds() {
  return getBridge().getDefaultKinds();
}
