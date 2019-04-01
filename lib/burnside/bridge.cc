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
#include "canonicalize.h"
#include "fe-helper.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "flattened.h"
#include "runtime.h"
#include "../evaluate/expression.h"
#include "../parser/parse-tree-visitor.h"
#include "../semantics/tools.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Module.h"
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

/// Converter from Fortran to FIR
class MLIRConverter {
  using LabelMapType = std::map<Fl::LabelRef, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  struct DoBoundsInfo {
    M::Value *doVar;

    M::Value *counter;
    M::Value *stepExpr;
    M::Operation *condition;
  };

  M::MLIRContext &mlirContext;
  M::OwningModuleRef module_;
  std::unique_ptr<M::OpBuilder> builder_;
  LabelMapType blockMap_;  // map from flattened labels to MLIR blocks
  std::list<Closure> edgeQ;
  std::map<const Pa::NonLabelDoStmt *, DoBoundsInfo> doMap;
  SymMap symbolMap;
  Pa::CharBlock lastKnownPos_;
  bool noInsPt{false};

  inline M::OpBuilder &build() { return *builder_.get(); }
  inline M::ModuleOp getMod() { return module_.get(); }
  inline LabelMapType &blkMap() { return blockMap_; }

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) {
    return parserPosToLoc(mlirContext, cb);
  }
  M::Location toLocation() { return toLocation(lastKnownPos_); }

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
    return createSomeAddress(loc, build(), *expr, symbolMap);
  }
  M::Value *createFIRExpr(M::Location loc, const SomeExpr *expr) {
    return createSomeExpression(loc, build(), *expr, symbolMap);
  }
  M::Value *createTemp(M::Type type, Se::Symbol *symbol = nullptr) {
    return createTemporary(toLocation(), build(), symbolMap, type, symbol);
  }

  M::FuncOp genFunctionMLIR(llvm::StringRef callee, M::FunctionType funcTy) {
    if (auto func{getNamedFunction(callee)}) {
      return func;
    }
    return createFunction(getMod(), callee, funcTy);
  }

  M::FuncOp genRuntimeFunction(RuntimeEntryCode rec, int kind) {
    return genFunctionMLIR(
        getRuntimeEntryName(rec), getRuntimeEntryType(rec, mlirContext, kind));
  }

  template<typename T> DoBoundsInfo *getBoundsInfo(const T &linearOp) {
    auto &st{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(linearOp.v->t)};
    lastKnownPos_ = st.source;
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

  void genMLIR(AnalysisData &ad, std::list<Fl::Op> &operations);

  // Control flow destination
  void genMLIR(bool lastWasLabel, const Fl::LabelOp &op) {
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
  void genMLIR(const Fl::GotoOp &op) {
    auto iter{blkMap().find(op.target)};
    if (iter != blkMap().end()) {
      build().create<M::BranchOp>(toLocation(), iter->second);
    } else {
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](M::OpBuilder *builder, M::Block *block, Fl::LabelRef dest,
              M::Location location, const LabelMapType &map) {
            builder->setInsertionPointToEnd(block);
            assert(map.find(dest) != map.end() && "no destination");
            builder->create<M::BranchOp>(location, map.find(dest)->second);
          },
          &build(), build().getInsertionBlock(), op.target, toLocation(), _1));
    }
    noInsPt = true;
  }
  void genMLIR(const Fl::ReturnOp &op) {
    std::visit([&](const auto *stmt) { genMLIR(*stmt); }, op.u);
    noInsPt = true;
  }
  void genMLIR(const Fl::ConditionalGotoOp &op) {
    std::visit(
        [&](const auto *stmt) { genMLIR(*stmt, op.trueLabel, op.falseLabel); },
        op.u);
    noInsPt = true;
  }

  void genMLIR(const Fl::SwitchIOOp &op);

  // CALL with alt-return value returned
  void genMLIR(const Fl::SwitchOp &op, const Pa::CallStmt &stmt) {
    auto loc{toLocation(op.source)};
    // FIXME
    (void)loc;
  }
  void genMLIR(const Fl::SwitchOp &op, const Pa::ComputedGotoStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  void genMLIR(const Fl::SwitchOp &op, const Pa::ArithmeticIfStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  M::Value *fromCaseValue(const M::Location &locs, const Pa::CaseValue &val) {
    return createFIRExpr(locs, Se::GetExpr(val));
  }
  void genMLIR(const Fl::SwitchOp &op, const Pa::CaseConstruct &stmt);
  void genMLIR(const Fl::SwitchOp &op, const Pa::SelectRankConstruct &stmt);
  void genMLIR(const Fl::SwitchOp &op, const Pa::SelectTypeConstruct &stmt);
  void genMLIR(const Fl::SwitchOp &op) {
    std::visit([&](auto *construct) { genMLIR(op, *construct); }, op.u);
    noInsPt = true;
  }

  void genMLIR(AnalysisData &ad, const Fl::ActionOp &op);

  void pushDoContext(const Pa::NonLabelDoStmt *doStmt,
      M::Value *doVar = nullptr, M::Value *counter = nullptr,
      M::Value *stepExpr = nullptr) {
    doMap.emplace(doStmt, DoBoundsInfo{doVar, counter, stepExpr});
  }

  void genLoopEnterMLIR(const Pa::LoopControl::Bounds &bounds,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    auto loc{toLocation(source)};
    auto *nameExpr{bounds.name.thing.symbol};
    auto *name{createTemp(getDefaultIntegerType(), nameExpr)};
    // evaluate e1, e2 [, e3] ...
    auto *lowerExpr{Se::GetExpr(bounds.lower)};
    auto *e1{createFIRExpr(loc, lowerExpr)};
    auto *upperExpr{Se::GetExpr(bounds.upper)};
    auto *e2{createFIRExpr(loc, upperExpr)};
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

  void genLoopEnterMLIR(const Pa::ScalarLogicalExpr &logicalExpr,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.1, para. 2
    // See BuildLoopLatchExpression()
    pushDoContext(stmt);
  }
  void genLoopEnterMLIR(const Pa::LoopControl::Concurrent &concurrent,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.2
    // FIXME
  }
  void genEnterMLIR(const Pa::DoConstruct &construct) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    lastKnownPos_ = stmt.source;
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value()) {
      std::visit([&](const auto &x) { genLoopEnterMLIR(x, &ss, stmt.source); },
          ctrl->u);
    } else {
      // loop forever (See 11.1.7.4.1, para. 2)
      pushDoContext(&ss);
    }
  }
  template<typename A> void genEnterMLIR(const A &construct) {
    // FIXME: add other genEnterMLIR() members
  }
  void genMLIR(const Fl::BeginOp &op) {
    std::visit([&](auto *construct) { genEnterMLIR(*construct); }, op.u);
  }

  void genExitMLIR(const Pa::DoConstruct &construct) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    lastKnownPos_ = stmt.source;
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(ss.t)};
    if (ctrl.has_value() &&
        std::holds_alternative<parser::LoopControl::Bounds>(ctrl->u)) {
      doMap.erase(&ss);
    }
    noInsPt = true;  // backedge already processed
  }
  void genMLIR(const Fl::EndOp &op) {
    if (auto *construct{std::get_if<const Pa::DoConstruct *>(&op.u)})
      genExitMLIR(**construct);
  }

  void genMLIR(AnalysisData &ad, const Fl::IndirectGotoOp &op);
  void genMLIR(const Fl::DoIncrementOp &op) {
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: do_var = do_var + e3
      auto load{
          build().create<fir::LoadOp>(info->doVar->getLoc(), info->doVar)};
      auto incremented{build().create<M::AddIOp>(
          load.getLoc(), load.getResult(), info->stepExpr)};
      build().create<fir::StoreOp>(load.getLoc(), incremented, info->doVar);
      // add: counter--
      auto loadCtr{build().create<fir::LoadOp>(
          info->counter->getLoc(), info->counter)};
      auto one{build().create<M::ConstantOp>(
          loadCtr.getLoc(), build().getIntegerAttr(loadCtr.getType(), 1))};
      auto decremented{build().create<M::SubIOp>(
          loadCtr.getLoc(), loadCtr.getResult(), one)};
      build().create<fir::StoreOp>(
          loadCtr.getLoc(), decremented, info->counter);
    }
  }
  void genMLIR(const Fl::DoCompareOp &op) {
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: cond = counter > 0 (signed)
      auto load{build().create<fir::LoadOp>(
          info->counter->getLoc(), info->counter)};
      auto zero{build().create<M::ConstantOp>(
          load.getLoc(), build().getIntegerAttr(load.getType(), 0))};
      auto cond{build().create<M::CmpIOp>(
          load.getLoc(), M::CmpIPredicate::SGT, load, zero)};
      info->condition = cond;
    }
  }
  void genMLIR(const Pa::FailImageStmt &stmt) {
    auto callee{genRuntimeFunction(FIRT_FAIL_IMAGE, 0)};
    llvm::SmallVector<M::Value *, 1> operands;  // FAIL IMAGE has no args
    build().create<M::CallOp>(toLocation(), callee, operands);
    build().create<fir::UnreachableOp>(toLocation());
  }
  void genMLIR(const Pa::ReturnStmt &stmt) {
    build().create<M::ReturnOp>(toLocation());  // FIXME: argument(s)?
  }
  void genMLIR(const Pa::StopStmt &stmt) {
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
  void genMLIR(
      const A &tuple, Fl::LabelRef trueLabel, Fl::LabelRef falseLabel) {
    auto *exprRef{Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple))};
    assert(exprRef && "condition expression missing");
    auto *cond{createFIRExpr(toLocation(), exprRef)};
    genCondBranch(cond, trueLabel, falseLabel);
  }
  void genMLIR(const Pa::Statement<Pa::IfThenStmt> &stmt,
      Fl::LabelRef trueLabel, Fl::LabelRef falseLabel) {
    lastKnownPos_ = stmt.source;
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(const Pa::Statement<Pa::ElseIfStmt> &stmt,
      Fl::LabelRef trueLabel, Fl::LabelRef falseLabel) {
    lastKnownPos_ = stmt.source;
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(
      const Pa::IfStmt &stmt, Fl::LabelRef trueLabel, Fl::LabelRef falseLabel) {
    genMLIR(stmt.t, trueLabel, falseLabel);
  }

  M::Value *getTrueConstant() {
    auto attr{build().getBoolAttr(true)};
    return build().create<M::ConstantOp>(toLocation(), attr).getResult();
  }

  // Conditional branch to enter loop body or exit
  void genMLIR(const Pa::Statement<Pa::NonLabelDoStmt> &stmt,
      Fl::LabelRef trueLabel, Fl::LabelRef falseLabel) {
    lastKnownPos_ = stmt.source;
    auto &loopCtrl{std::get<std::optional<Pa::LoopControl>>(stmt.statement.t)};
    M::Value *condition{nullptr};
    if (loopCtrl.has_value()) {
      std::visit(Co::visitors{
                     [&](const parser::LoopControl::Bounds &) {
                       auto iter{doMap.find(&stmt.statement)};
                       assert(iter != doMap.end());
                       condition = iter->second.condition->getResult(0);
                     },
                     [&](const parser::ScalarLogicalExpr &logical) {
                       auto loc{toLocation(stmt.source)};
                       auto *exp{Se::GetExpr(logical)};
                       condition = createFIRExpr(loc, exp);
                     },
                     [&](const parser::LoopControl::Concurrent &concurrent) {
                       // FIXME: incorrectly lowering DO CONCURRENT
                       condition = getTrueConstant();
                     },
                 },
          loopCtrl->u);
    } else {
      condition = getTrueConstant();
    }
    assert(condition && "condition must be a Value");
    genCondBranch(condition, trueLabel, falseLabel);
  }

  // Action statements
  void genMLIR(const Pa::AllocateStmt &stmt);
  void genMLIR(const Pa::AssignmentStmt &stmt) {
    auto *rhs{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *lhs{Se::GetExpr(std::get<Pa::Variable>(stmt.t))};
    auto loc{toLocation()};
    build().create<fir::StoreOp>(
        loc, createFIRExpr(loc, rhs), createFIRAddr(loc, lhs));
  }
  void genMLIR(const Pa::BackspaceStmt &stmt);
  void genMLIR(const Pa::CallStmt &stmt);
  void genMLIR(const Pa::CloseStmt &stmt);
  void genMLIR(const Pa::DeallocateStmt &stmt);
  void genMLIR(const Pa::EndfileStmt &stmt);
  void genMLIR(const Pa::EventPostStmt &stmt);
  void genMLIR(const Pa::EventWaitStmt &stmt);
  void genMLIR(const Pa::FlushStmt &stmt);
  void genMLIR(const Pa::FormTeamStmt &stmt);
  void genMLIR(const Pa::InquireStmt &stmt);
  void genMLIR(const Pa::LockStmt &stmt);
  void genMLIR(const Pa::NullifyStmt &stmt);
  void genMLIR(const Pa::OpenStmt &stmt);
  void genMLIR(const Pa::PointerAssignmentStmt &stmt);
  void genMLIR(const Pa::PrintStmt &stmt);
  void genMLIR(const Pa::ReadStmt &stmt);
  void genMLIR(const Pa::RewindStmt &stmt);
  void genMLIR(const Pa::SyncAllStmt &stmt);
  void genMLIR(const Pa::SyncImagesStmt &stmt);
  void genMLIR(const Pa::SyncMemoryStmt &stmt);
  void genMLIR(const Pa::SyncTeamStmt &stmt);
  void genMLIR(const Pa::UnlockStmt &stmt);
  void genMLIR(const Pa::WaitStmt &stmt);
  void genMLIR(const Pa::WhereStmt &stmt);
  void genMLIR(const Pa::WriteStmt &stmt);
  void genMLIR(const Pa::ForallStmt &stmt);
  void genMLIR(AnalysisData &ad, const Pa::AssignStmt &stmt);
  void genMLIR(const Pa::PauseStmt &stmt);

  template<typename A>
  void translateRoutine(
      const A &routine, const std::string &name, const Se::Symbol *funcSym);

  void genCondBranch(
      M::Value *cond, Fl::LabelRef trueBlock, Fl::LabelRef falseBlock) {
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
              Fl::LabelRef trueDest, Fl::LabelRef falseDest,
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
      const std::vector<Fl::LabelRef> &labels) {
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
              const std::vector<Fl::LabelRef> &labels, M::Location location,
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
  MLIRConverter(BurnsideBridge &bridge)
    : mlirContext{bridge.getMLIRContext()}, module_{bridge.getModule()} {}
  MLIRConverter() = delete;

  M::ModuleOp getModule() { return getMod(); }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  /// Translate the various routines from the parse tree
  void Post(const Pa::MainProgram &mainp) {
    std::string mainName{"_MAIN"s};
    if (auto &ps{
            std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(mainp.t)}) {
      mainName = ps->statement.v.ToString();
      lastKnownPos_ = ps->source;
    }
    translateRoutine(mainp, mainName, nullptr);
  }
  void Post(const Pa::FunctionSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::FunctionStmt>>(subp.t)};
    lastKnownPos_ = stmt.source;
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
  void Post(const Pa::SubroutineSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::SubroutineStmt>>(subp.t)};
    lastKnownPos_ = stmt.source;
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
};

/// SELECT CASE
/// Build a switch-like structure for a SELECT CASE
void MLIRConverter::genMLIR(
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
void MLIRConverter::genMLIR(
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
void MLIRConverter::genMLIR(
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

void MLIRConverter::genMLIR(const Fl::SwitchIOOp &op) {}

void MLIRConverter::genMLIR(const Pa::AllocateStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::BackspaceStmt &stmt) {
  // builder->create<IOCallOp>(stmt.v);
}
void MLIRConverter::genMLIR(const Pa::CallStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::CloseStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::DeallocateStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EndfileStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EventPostStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EventWaitStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::FlushStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::FormTeamStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::InquireStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::LockStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::NullifyStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::OpenStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::PointerAssignmentStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::PrintStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::ReadStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::RewindStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncAllStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncImagesStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncMemoryStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncTeamStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::UnlockStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WaitStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WhereStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WriteStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::ForallStmt &stmt) {}
void MLIRConverter::genMLIR(AnalysisData &ad, const Pa::AssignStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::PauseStmt &stmt) {}

/// translate action statements
void MLIRConverter::genMLIR(AnalysisData &ad, const Fl::ActionOp &op) {
  lastKnownPos_ = op.v->source;
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
            genMLIR(ad, assign.value());
          },
          [&](const auto &stmt) { genMLIR(stmt.value()); },
      },
      op.v->statement.u);
}

void MLIRConverter::genMLIR(AnalysisData &ad, const Fl::IndirectGotoOp &op) {
  // add or queue an igoto
}

void MLIRConverter::genMLIR(AnalysisData &ad, std::list<Fl::Op> &operations) {
  bool lastWasLabel{false};
  for (auto &op : operations) {
    std::visit(Co::visitors{
                   [&](const Fl::IndirectGotoOp &oper) {
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::ActionOp &oper) {
                     noInsPt = false;
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Fl::LabelOp &oper) {
                     genMLIR(lastWasLabel, oper);
                     lastWasLabel = true;
                   },
                   [&](const Fl::BeginOp &oper) {
                     noInsPt = false;
                     genMLIR(oper);
                     lastWasLabel = true;
                   },
                   [&](const auto &oper) {
                     noInsPt = false;
                     genMLIR(oper);
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
void MLIRConverter::translateRoutine(
    const A &routine, const std::string &name, const Se::Symbol *funcSym) {
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
  builder_ = std::make_unique<M::OpBuilder>(func);
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
  genMLIR(ad, operations);
  finalizeQueued();
}

M::DialectRegistration<fir::FIROpsDialect> FIROps;

}  // namespace

void Br::crossBurnsideBridge(BurnsideBridge &bridge, const Pa::Program &prg) {
  MLIRConverter converter{bridge};
  Walk(prg, converter);
}

std::unique_ptr<llvm::Module> Br::LLVMBridge(M::ModuleOp &module) {
  return M::translateModuleToLLVMIR(module);
}

void Br::BurnsideBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  module_ = M::parseSourceFile(srcMgr, context_.get());
  if (validModule()) {
    // symbols are added by ModuleManager ctor
    manager_.reset(new M::ModuleManager(getModule()));
  }
}

Br::BurnsideBridge::BurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds)
  : defaultKinds_{defaultKinds} {
  context_ = std::make_unique<M::MLIRContext>();
  module_ = M::OwningModuleRef{
      M::ModuleOp::create(M::UnknownLoc::get(context_.get()))};
  manager_ = std::make_unique<M::ModuleManager>(getModule());
}

void Br::instantiateBurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds) {
  auto p{BurnsideBridge::create(defaultKinds)};
  bridgeInstance.swap(p);
}

BurnsideBridge &Br::getBridge() { return *bridgeInstance.get(); }

const common::IntrinsicTypeDefaultKinds &Br::getDefaultKinds() {
  return getBridge().getDefaultKinds();
}
