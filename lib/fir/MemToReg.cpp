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

#include "fir/Transforms/MemToReg.h"
#include "fir/Analysis/IteratedDominanceFrontier.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>
#include <vector>

namespace M = mlir;

using namespace fir;

using DominatorTree = M::DominanceInfo;

namespace {

bool isAllocaPromotable(fir::AllocaOp &ae) {
  for (auto &use : ae.getResult()->getUses()) {
    auto *op = use.getOwner();
    if (auto load = M::dyn_cast<fir::LoadOp>(op)) {
      // do nothing
    } else if (auto stor = M::dyn_cast<fir::StoreOp>(op)) {
      if (stor.getOperand(0)->getDefiningOp() == op) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

struct AllocaInfo {
  llvm::SmallVector<M::Block *, 32> definingBlocks;
  llvm::SmallVector<M::Block *, 32> usingBlocks;

  M::Operation *onlyStore;
  M::Block *onlyBlock;
  bool onlyUsedInOneBlock;

  void clear() {
    definingBlocks.clear();
    usingBlocks.clear();
    onlyStore = nullptr;
    onlyBlock = nullptr;
    onlyUsedInOneBlock = true;
  }

  /// Scan the uses of the specified alloca, filling in the AllocaInfo used
  /// by the rest of the pass to reason about the uses of this alloca.
  void analyzeAlloca(fir::AllocaOp &AI) {
    clear();

    // As we scan the uses of the alloca instruction, keep track of stores,
    // and decide whether all of the loads and stores to the alloca are within
    // the same basic block.
    for (auto UI = AI.getResult()->use_begin(), E = AI.getResult()->use_end();
         UI != E;) {
      auto *User = UI->getOwner();
      UI++;

      if (auto SI = M::dyn_cast<fir::StoreOp>(User)) {
        // Remember the basic blocks which define new values for the alloca
        definingBlocks.push_back(SI.getOperation()->getBlock());
        onlyStore = SI.getOperation();
      } else {
        auto LI = M::cast<fir::LoadOp>(User);
        // Otherwise it must be a load instruction, keep track of variable
        // reads.
        usingBlocks.push_back(LI.getOperation()->getBlock());
      }

      if (onlyUsedInOneBlock) {
        if (!onlyBlock)
          onlyBlock = User->getBlock();
        else if (onlyBlock != User->getBlock())
          onlyUsedInOneBlock = false;
      }
    }
  }
};

struct RenamePassData {
  using ValVector = std::vector<M::Value *>;

  RenamePassData(M::Block *b, M::Block *p, const ValVector &v)
      : BB(b), Pred(p), Values(v) {}
  RenamePassData(const RenamePassData &) = delete;
  RenamePassData &operator=(const RenamePassData &) = delete;
  RenamePassData(RenamePassData &&) = default;
  ~RenamePassData() = default;

  M::Block *BB;
  M::Block *Pred;
  ValVector Values;
};

struct LargeBlockInfo {
  using INMap = llvm::DenseMap<M::Operation *, unsigned>;
  INMap instNumbers;

  static bool isInterestingInstruction(M::Operation &I) {
    if (M::isa<fir::LoadOp>(I)) {
      if (auto *op = I.getOperand(0)->getDefiningOp())
        return M::isa<fir::AllocaOp>(op);
    } else if (M::isa<fir::StoreOp>(I)) {
      if (auto *op = I.getOperand(1)->getDefiningOp())
        return M::isa<fir::AllocaOp>(op);
    }
    return false;
  }

  template <typename A>
  unsigned getInstructionIndex(A &op) {
    auto *oper = op.getOperation();

    // has it already been numbered?
    INMap::iterator it = instNumbers.find(oper);
    if (it != instNumbers.end()) {
      return it->second;
    }

    // No. search for the oper
    auto *block = oper->getBlock();
    unsigned num = 0u;
    for (auto &o : block->getOperations())
      if (isInterestingInstruction(o)) {
        instNumbers[&o] = num++;
      }
    it = instNumbers.find(oper);
    assert(it != instNumbers.end() && "operation not in block?");
    return it->second;
  }

  template <typename A>
  void deleteValue(A &op) {
    auto *oper = op.getOperation();
    instNumbers.erase(oper);
  }

  void clear() { instNumbers.clear(); }
};

struct MemToReg : public M::FunctionPass<MemToReg> {
  explicit MemToReg() {}

  std::vector<fir::AllocaOp> allocas;
  DominatorTree *domTree = nullptr;
  M::OpBuilder *builder = nullptr;

  /// Contains a stable numbering of basic blocks to avoid non-determinstic
  /// behavior.
  llvm::DenseMap<M::Block *, unsigned> BBNumbers;

  /// Reverse mapping of Allocas.
  llvm::DenseMap<M::Operation *, unsigned> allocaLookup;

  /// The set of basic blocks the renamer has already visited.
  llvm::SmallPtrSet<M::Block *, 16> Visited;

  llvm::DenseMap<std::pair<M::Block *, M::Operation *>, unsigned> BlockArgs;
  llvm::DenseMap<std::pair<M::Block *, unsigned>, unsigned> argToAllocaMap;

  bool rewriteSingleStoreAlloca(fir::AllocaOp &AI, AllocaInfo &Info,
                                LargeBlockInfo &LBI) {
    fir::StoreOp onlyStore(M::cast<fir::StoreOp>(Info.onlyStore));
    M::Block *StoreBB = Info.onlyStore->getBlock();
    int StoreIndex = -1;

    // Clear out usingBlocks.  We will reconstruct it here if needed.
    Info.usingBlocks.clear();

    for (auto UI = AI.getResult()->use_begin(), E = AI.getResult()->use_end();
         UI != E;) {
      auto *UserInst = UI->getOwner();
      UI++;

      if (M::dyn_cast<fir::StoreOp>(UserInst)) {
        continue;
      }
      auto LI = M::cast<fir::LoadOp>(UserInst);

      // Okay, if we have a load from the alloca, we want to replace it with the
      // only value stored to the alloca.  We can do this if the value is
      // dominated by the store.  If not, we use the rest of the MemToReg
      // machinery to insert the phi nodes as needed.
      if (LI.getOperation()->getBlock() == StoreBB) {
        // If we have a use that is in the same block as the store, compare
        // the indices of the two instructions to see which one came first. If
        // the load came before the store, we can't handle it.
        if (StoreIndex == -1) {
          StoreIndex = LBI.getInstructionIndex(onlyStore);
        }

        if (unsigned(StoreIndex) > LBI.getInstructionIndex(LI)) {
          // Can't handle this load, bail out.
          Info.usingBlocks.push_back(StoreBB);
          continue;
        }
      } else if (!domTree->dominates(StoreBB, LI.getOperation()->getBlock())) {
        // If the load and store are in different blocks, use BB dominance to
        // check their relationships.  If the store doesn't dom the use, bail
        // out.
        Info.usingBlocks.push_back(LI.getOperation()->getBlock());
        continue;
      }

      // Otherwise, we *can* safely rewrite this load.
      M::Value *ReplVal = onlyStore.getOperand(0);
      // If the replacement value is the load, this must occur in unreachable
      // code.
      if (ReplVal == LI.getResult()) {
        ReplVal = builder->create<UndefOp>(LI.getLoc(), LI.getType());
      }

      LI.replaceAllUsesWith(ReplVal);
      LI.erase();
      LBI.deleteValue(LI);
    }

    // Finally, after the scan, check to see if the store is all that is left.
    if (!Info.usingBlocks.empty())
      return false; // If not, we'll have to fall back for the remainder.

    // Remove the (now dead) store and alloca.
    Info.onlyStore->erase();

    AI.erase();
    return true;
  }

  bool promoteSingleBlockAlloca(fir::AllocaOp &AI, AllocaInfo &Info,
                                LargeBlockInfo &LBI) {
    // Walk the use-def list of the alloca, getting the locations of all stores.
    using StoresByIndexTy =
        llvm::SmallVector<std::pair<unsigned, fir::StoreOp *>, 64>;
    StoresByIndexTy StoresByIndex;

    for (auto U = AI.getResult()->use_begin(), E = AI.getResult()->use_end();
         U != E; U++)
      if (fir::StoreOp SI = M::dyn_cast<fir::StoreOp>(U->getOwner())) {
        StoresByIndex.emplace_back(LBI.getInstructionIndex(SI), &SI);
      }

    // Sort the stores by their index, making it efficient to do a lookup with a
    // binary search.
    llvm::sort(StoresByIndex, llvm::less_first());

    // Walk all of the loads from this alloca, replacing them with the nearest
    // store above them, if any.
    for (auto UI = AI.getResult()->use_begin(), E = AI.getResult()->use_end();
         UI != E;) {
      auto LI = M::dyn_cast<fir::LoadOp>(UI->getOwner());
      UI++;
      if (!LI) {
        continue;
      }

      unsigned LoadIdx = LBI.getInstructionIndex(LI);

      // Find the nearest store that has a lower index than this load.
      typename StoresByIndexTy::iterator I = llvm::lower_bound(
          StoresByIndex,
          std::make_pair(LoadIdx, static_cast<fir::StoreOp *>(nullptr)),
          llvm::less_first());
      if (I == StoresByIndex.begin()) {
        if (StoresByIndex.empty()) {
          // If there are no stores, the load takes the undef value.
          auto undef = builder->create<UndefOp>(LI.getLoc(), LI.getType());
          LI.replaceAllUsesWith(undef.getResult());
        } else {
          // There is no store before this load, bail out (load may be affected
          // by the following stores - see main comment).
          return false;
        }
      } else {
        // Otherwise, there was a store before this load, the load takes its
        // value. Note, if the load was marked as nonnull we don't want to lose
        // that information when we erase it. So we preserve it with an assume.
        M::Value *ReplVal = std::prev(I)->second->getOperand(0);

        // If the replacement value is the load, this must occur in unreachable
        // code.
        if (ReplVal == LI) {
          ReplVal = builder->create<UndefOp>(LI.getLoc(), LI.getType());
        }

        LI.replaceAllUsesWith(ReplVal);
      }

      LI.erase();
      LBI.deleteValue(LI);
    }

    // Remove the (now dead) stores and alloca.
    while (!AI.use_empty()) {
      auto *ae = AI.getResult();
      for (auto ai = ae->use_begin(), E = ae->use_end(); ai != E; ai++) {
        if (fir::StoreOp SI =
                M::dyn_cast<fir::StoreOp>(ai->get()->getDefiningOp())) {
          SI.erase();
          LBI.deleteValue(SI);
        }
      }
    }

    AI.erase();
    return true;
  }

  void computeLiveInBlocks(fir::AllocaOp &ae, AllocaInfo &Info,
                           const llvm::SmallPtrSetImpl<M::Block *> &DefBlocks,
                           llvm::SmallPtrSetImpl<M::Block *> &liveInBlks) {
    auto *AI = ae.getOperation();
    // To determine liveness, we must iterate through the predecessors of blocks
    // where the def is live.  Blocks are added to the worklist if we need to
    // check their predecessors.  Start with all the using blocks.
    llvm::SmallVector<M::Block *, 64> LiveInBlockWorklist(
        Info.usingBlocks.begin(), Info.usingBlocks.end());

    // If any of the using blocks is also a definition block, check to see if
    // the definition occurs before or after the use.  If it happens before the
    // use, the value isn't really live-in.
    for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
      M::Block *BB = LiveInBlockWorklist[i];
      if (!DefBlocks.count(BB)) {
        continue;
      }

      // Okay, this is a block that both uses and defines the value.  If the
      // first reference to the alloca is a def (store), then we know it isn't
      // live-in.
      for (M::Block::iterator I = BB->begin();; ++I) {
        if (auto SI = M::dyn_cast<fir::StoreOp>(I)) {
          if (SI.getOperand(1)->getDefiningOp() != AI) {
            continue;
          }

          // We found a store to the alloca before a load.  The alloca is not
          // actually live-in here.
          LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
          LiveInBlockWorklist.pop_back();
          --i;
          --e;
          break;
        }

        if (auto LI = M::dyn_cast<fir::LoadOp>(I))
          // Okay, we found a load before a store to the alloca.  It is actually
          // live into this block.
          if (LI.getOperand()->getDefiningOp() == AI) {
            break;
          }
      }
    }

    // Now that we have a set of blocks where the phi is live-in, recursively
    // add their predecessors until we find the full region the value is live.
    while (!LiveInBlockWorklist.empty()) {
      M::Block *BB = LiveInBlockWorklist.pop_back_val();

      // The block really is live in here, insert it into the set.  If already
      // in the set, then it has already been processed.
      if (!liveInBlks.insert(BB).second) {
        continue;
      }

      // Since the value is live into BB, it is either defined in a predecessor
      // or live into it to.  Add the preds to the worklist unless they are a
      // defining block.
      for (M::Block *P : BB->getPredecessors()) {
        // The value is not live into a predecessor if it defines the value.
        if (DefBlocks.count(P)) {
          continue;
        }

        // Otherwise it is, add to the worklist.
        LiveInBlockWorklist.push_back(P);
      }
    }
  }

  bool addBlockArgument(M::Block *BB, fir::AllocaOp &Alloca,
                        unsigned allocaNum) {
    auto *ae = Alloca.getOperation();
    auto key = std::make_pair(BB, ae);
    auto argNoIter = BlockArgs.find(key);
    if (argNoIter != BlockArgs.end()) {
      return false;
    }
    auto argNo = BB->getNumArguments();
    BB->addArgument(Alloca.getAllocatedType());
    BlockArgs[key] = argNo;
    argToAllocaMap[std::make_pair(BB, argNo)] = allocaNum;
    return true;
  }

  template <typename A>
  void initOperands(std::vector<M::Value *> &opers, M::Location &&loc,
                    M::Block *dest, unsigned size, unsigned ai, M::Value *val,
                    A &&oldOpers) {
    unsigned i = 0;
    for (auto v : oldOpers) {
      opers[i++] = v;
    }
    // we must fill additional args with temporary undef values
    for (; i < size; ++i) {
      if (i == ai)
        continue;
      auto opTy = dest->getArgument(i)->getType();
      auto typedUndef = builder->create<UndefOp>(loc, opTy);
      opers[i] = typedUndef;
    }
    opers[ai] = val;
  }

  static void eraseIfNoUse(M::Value *val) {
    if (val->use_begin() == val->use_end()) {
      val->getDefiningOp()->erase();
    }
  }

  /// Set the incoming value on the branch side for the `ai`th block argument
  void setParam(M::Block *blk, unsigned ai, M::Value *val, M::Block *target,
                unsigned size) {
    auto *term = blk->getTerminator();
    if (auto br = M::dyn_cast<M::BranchOp>(term)) {
      if (br.getNumOperands() <= ai) {
        // construct a new BranchOp to replace term
        std::vector<M::Value *> opers(size);
        auto *dest = br.getDest();
        builder->setInsertionPoint(term);
        initOperands(opers, br.getLoc(), dest, size, ai, val, br.getOperands());
        builder->create<M::BranchOp>(br.getLoc(), dest, opers);
        br.erase();
      } else {
        auto *oldParam = br.getOperand(ai);
        br.setOperand(ai, val);
        eraseIfNoUse(oldParam);
      }
    } else if (auto cond = M::dyn_cast<M::CondBranchOp>(term)) {
      if (target == cond.getTrueDest()) {
        if (cond.getNumTrueOperands() <= ai) {
          // construct a new CondBranchOp to replace term
          std::vector<M::Value *> opers(size);
          auto *dest = cond.getTrueDest();
          builder->setInsertionPoint(term);
          initOperands(opers, cond.getLoc(), dest, size, ai, val,
                       cond.getTrueOperands());
          auto *c = cond.getCondition();
          auto *othDest = cond.getFalseDest();
          std::vector<M::Value *> othOpers(cond.false_operand_begin(),
                                           cond.false_operand_end());
          builder->create<M::CondBranchOp>(cond.getLoc(), c, dest, opers,
                                           othDest, othOpers);
          cond.erase();
        } else {
          auto *oldParam = cond.getTrueOperand(ai);
          cond.setTrueOperand(ai, val);
          eraseIfNoUse(oldParam);
        }
      } else {
        if (cond.getNumFalseOperands() <= ai) {
          // construct a new CondBranchOp to replace term
          std::vector<M::Value *> opers(size);
          auto *dest = cond.getFalseDest();
          builder->setInsertionPoint(term);
          initOperands(opers, cond.getLoc(), dest, size, ai, val,
                       cond.getFalseOperands());
          auto *c = cond.getCondition();
          auto *othDest = cond.getTrueDest();
          std::vector<M::Value *> othOpers(cond.true_operand_begin(),
                                           cond.true_operand_end());
          builder->create<M::CondBranchOp>(cond.getLoc(), c, othDest, othOpers,
                                           dest, opers);
          cond.erase();
        } else {
          auto *oldParam = cond.getFalseOperand(ai);
          cond.setFalseOperand(ai, val);
          eraseIfNoUse(oldParam);
        }
      }
    } else {
      assert(false && "unhandled terminator");
    }
  }

  inline static void addValue(RenamePassData::ValVector &vector,
                              RenamePassData::ValVector::size_type size,
                              M::Value *value) {
    if (vector.size() < size + 1) {
      vector.resize(size + 1);
    }
    vector[size] = value;
  }

  /// Recursively traverse the CFG of the function, renaming loads and
  /// stores to the allocas which we are promoting.
  ///
  /// IncomingVals indicates what value each Alloca contains on exit from the
  /// predecessor block Pred.
  void renamePass(M::Block *BB, M::Block *Pred,
                  RenamePassData::ValVector &IncomingVals,
                  std::vector<RenamePassData> &Worklist) {
  NextIteration:
    // Does this block take arguments?
    if ((!BB->hasNoPredecessors()) && (BB->getNumArguments() > 0)) {
      // add the values from block `Pred` to the argument list in the proper
      // positions
      for (unsigned ai = 0, AI = BB->getNumArguments(); ai != AI; ++ai) {
        auto allocaNo = argToAllocaMap[std::make_pair(BB, ai)];
        setParam(Pred, ai, IncomingVals[allocaNo], BB, AI);
        // use the block argument, not the live def in the pred block
        addValue(IncomingVals, allocaNo, BB->getArgument(ai));
      }
    }

    // Don't revisit blocks.
    if (!Visited.insert(BB).second) {
      return;
    }

    M::Block::iterator II = BB->begin();
    while (true) {
      if (II == BB->end())
        break;
      M::Operation &opn = *II;
      II++;

      if (auto LI = M::dyn_cast<fir::LoadOp>(opn)) {
        auto *srcOpn = LI.getOperand()->getDefiningOp();
        if (!srcOpn) {
          continue;
        }

        auto Src = M::dyn_cast<fir::AllocaOp>(srcOpn);
        if (!Src) {
          continue;
        }

        llvm::DenseMap<M::Operation *, unsigned>::iterator AI =
            allocaLookup.find(srcOpn);
        if (AI == allocaLookup.end()) {
          continue;
        }

        M::Value *V = IncomingVals[AI->second];

        // Anything using the load now uses the current value.
        LI.replaceAllUsesWith(V);
        LI.erase();
      } else if (auto SI = M::dyn_cast<fir::StoreOp>(opn)) {
        auto *dstOpn = SI.getOperand(1)->getDefiningOp();
        if (!dstOpn) {
          continue;
        }

        // Delete this instruction and mark the name as the current holder of
        // the value
        auto Dest = M::dyn_cast<fir::AllocaOp>(dstOpn);
        if (!Dest) {
          continue;
        }

        llvm::DenseMap<M::Operation *, unsigned>::iterator ai =
            allocaLookup.find(dstOpn);
        if (ai == allocaLookup.end()) {
          continue;
        }

        // what value were we writing?
        unsigned AllocaNo = ai->second;
        addValue(IncomingVals, AllocaNo, SI.getOperand(0));
        SI.erase();
      }
    }

    // 'Recurse' to our successors.
    auto I = BB->succ_begin();
    auto E = BB->succ_end();
    if (I == E) {
      return;
    }

    // Keep track of the successors so we don't visit the same successor twice
    llvm::SmallPtrSet<M::Block *, 8> VisitedSuccs;

    // Handle the first successor without using the worklist.
    VisitedSuccs.insert(*I);
    Pred = BB;
    BB = *I;
    ++I;

    for (; I != E; ++I)
      if (VisitedSuccs.insert(*I).second)
        Worklist.emplace_back(*I, Pred, IncomingVals);
    goto NextIteration;
  }

  void doPromotion() {
    auto F = getFunction();
    std::vector<fir::AllocaOp> aes;
    AllocaInfo info;
    LargeBlockInfo lbi;
    ForwardIDFCalculator IDF(*domTree);

    assert(!allocas.empty());

    for (unsigned allocaNum = 0, End = allocas.size(); allocaNum != End;
         ++allocaNum) {
      auto ae = allocas[allocaNum];
      assert(ae.getParentOfType<M::FuncOp>() == F);
      if (ae.use_empty()) {
        ae.erase();
        continue;
      }
      info.analyzeAlloca(ae);
      builder->setInsertionPointToStart(&F.front());
      if (info.definingBlocks.size() == 1)
        if (rewriteSingleStoreAlloca(ae, info, lbi)) {
          continue;
        }
      if (info.onlyUsedInOneBlock)
        if (promoteSingleBlockAlloca(ae, info, lbi)) {
          continue;
        }

      // If we haven't computed a numbering for the BB's in the function, do
      // so now.
      if (BBNumbers.empty()) {
        unsigned ID = 0;
        for (auto &BB : F)
          BBNumbers[&BB] = ID++;
      }

      // Keep the reverse mapping of the 'Allocas' array for the rename pass.
      allocaLookup[allocas[allocaNum].getOperation()] = allocaNum;

      // At this point, we're committed to promoting the alloca using IDF's,
      // and the standard SSA construction algorithm.  Determine which blocks
      // need PHI nodes and see if we can optimize out some work by avoiding
      // insertion of dead phi nodes.

      // Unique the set of defining blocks for efficient lookup.
      llvm::SmallPtrSet<M::Block *, 32> DefBlocks(info.definingBlocks.begin(),
                                                  info.definingBlocks.end());

      // Determine which blocks the value is live in.  These are blocks which
      // lead to uses.
      llvm::SmallPtrSet<M::Block *, 32> liveInBlks;
      computeLiveInBlocks(ae, info, DefBlocks, liveInBlks);

      // At this point, we're committed to promoting the alloca using IDF's,
      // and the standard SSA construction algorithm.  Determine which blocks
      // need phi nodes and see if we can optimize out some work by avoiding
      // insertion of dead phi nodes.
      IDF.setLiveInBlocks(liveInBlks);
      IDF.setDefiningBlocks(DefBlocks);
      llvm::SmallVector<M::Block *, 32> PHIBlocks;
      IDF.calculate(PHIBlocks);
      llvm::sort(PHIBlocks, [this](M::Block *A, M::Block *B) {
        return BBNumbers.find(A)->second < BBNumbers.find(B)->second;
      });

      for (M::Block *BB : PHIBlocks)
        addBlockArgument(BB, ae, allocaNum);

      aes.push_back(ae);
    }
    std::swap(allocas, aes);
    if (allocas.empty()) {
      return;
    }

    lbi.clear();

    // Set the incoming values for the basic block to be null values for all
    // of the alloca's.  We do this in case there is a load of a value that
    // has not been stored yet.  In this case, it will get this null value.
    RenamePassData::ValVector Values(allocas.size());
    for (unsigned i = 0, e = allocas.size(); i != e; ++i) {
      Values[i] = builder->create<UndefOp>(allocas[i].getLoc(),
                                           allocas[i].getAllocatedType());
    }

    // Walks all basic blocks in the function performing the SSA rename
    // algorithm and inserting the phi nodes we marked as necessary
    std::vector<RenamePassData> renameWorklist;
    renameWorklist.emplace_back(&F.front(), nullptr, Values);
    do {
      RenamePassData RPD(std::move(renameWorklist.back()));
      renameWorklist.pop_back();
      // renamePass may add new worklist entries.
      renamePass(RPD.BB, RPD.Pred, RPD.Values, renameWorklist);
    } while (!renameWorklist.empty());

    // The renamer uses the Visited set to avoid infinite loops.  Clear it
    // now.
    Visited.clear();

    // Remove the allocas themselves from the function.
    for (auto aa : allocas) {
      M::Operation *A = aa.getOperation();
      // If there are any uses of the alloca instructions left, they must be
      // in unreachable basic blocks that were not processed by walking the
      // dominator tree. Just delete the users now.
      if (!A->use_empty()) {
        auto undef = builder->create<UndefOp>(aa.getLoc(), aa.getType());
        aa.replaceAllUsesWith(undef.getResult());
      }
      aa.erase();
    }
  }

  /// run the MemToReg pass on the FIR dialect
  void runOnFunction() override {
    auto f = getFunction();
    auto &entry = f.front();
    auto bldr = M::OpBuilder(f.getBody());

    domTree = &getAnalysis<DominatorTree>();
    builder = &bldr;

    while (true) {
      allocas.clear();

      for (auto &op : entry)
        if (auto ae = M::dyn_cast<fir::AllocaOp>(op)) {
          if (isAllocaPromotable(ae)) {
            allocas.push_back(ae);
          }
        }

      if (allocas.empty()) {
        break;
      }
      doPromotion();
    }

    domTree = nullptr;
    builder = nullptr;
  }
};

} // namespace

std::unique_ptr<M::FunctionPassBase> fir::createMemToRegPass() {
  return std::make_unique<MemToReg>();
}
