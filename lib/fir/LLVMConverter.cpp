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

#include "fir/Tilikum/LLVMConverter.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/Type.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// This module performs the conversion of FIR operations to MLIR standard and/or
// LLVM-IR dialects.

namespace L = llvm;
namespace M = mlir;

using namespace fir;

namespace {

using SmallVecResult = L::SmallVector<M::Value *, 4>;
using OperandTy = L::ArrayRef<M::Value *>;
using AttributeTy = L::ArrayRef<M::NamedAttribute>;

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class FIRToLLVMTypeConverter : public M::LLVMTypeConverter {
public:
  using LLVMTypeConverter::LLVMTypeConverter;

  // convert a front-end kind value to either a std or LLVM IR dialect type
  static M::Type kindToRealType(M::MLIRContext *ctx, L::LLVMContext &llvmCtx,
                                KindTy kind) {
    switch (kind) {
    case 0:
      return M::FloatType::getF32(ctx); // FIXME: use defaulted kind
    case 2:
      return M::FloatType::getF16(ctx);
    case 3:
      return M::FloatType::getBF16(ctx);
    case 4:
      return M::FloatType::getF32(ctx);
    case 8:
      return M::FloatType::getF64(ctx);
    case 10:
      return M::LLVM::LLVMType::get(ctx, L::Type::getX86_FP80Ty(llvmCtx));
    case 16:
      return M::LLVM::LLVMType::get(ctx, L::Type::getFP128Ty(llvmCtx));
    }
    assert(!kind && "unhandled kind");
    return {};
  }

  // lower the type descriptor
  M::Type convertTypeDescType(M::MLIRContext *ctx) {
    auto &llvmCtx = getLLVMContext();
    auto i64 = L::Type::getIntNTy(llvmCtx, 64);
    return M::LLVM::LLVMType::get(ctx, i64->getPointerTo());
  }

  template <typename A>
  M::Type convertPointerLike(A &ty) {
    M::Type ele = ty.getEleTy();
    auto eleTy = convertType(ele);
    if (ele.dyn_cast<SequenceType>()) {
      return eleTy;
    }
    auto *ptrTy = eleTy.cast<M::LLVM::LLVMType>().getUnderlyingType();
    return M::LLVM::LLVMType::get(ty.getContext(), ptrTy->getPointerTo());
  }

  /// Convert FIR types to LLVM IR dialect types
  M::Type convertType(M::Type t) override {
    auto &llvmCtx = getLLVMContext();
    if (auto box = t.dyn_cast<BoxType>()) {
      // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
      L::SmallVector<M::Type, 6> parts;
      // buffer*
      M::Type ele = box.getEleTy();
      auto *ctx = box.getContext();
      M::Type eleTy = convertType(ele);
      L::Type *i64 = L::Type::getIntNTy(llvmCtx, 64);
      if (ele.dyn_cast<SequenceType>()) {
        parts.push_back(eleTy);
      } else {
        auto *ptrTy = eleTy.cast<M::LLVM::LLVMType>().getUnderlyingType();
        parts.push_back(M::LLVM::LLVMType::get(ctx, ptrTy->getPointerTo()));
      }
      // ele-size
      parts.push_back(M::LLVM::LLVMType::get(ctx, i64));
      // rank
      parts.push_back(convertTypeDescType(ctx));
      // attribute
      parts.push_back(M::LLVM::LLVMType::get(ctx, i64));
      // [(int,int,int)]
      parts.push_back(M::LLVM::LLVMType::get(
          ctx, L::ArrayType::get(i64, 3)->getPointerTo()));
      // ...
      return LLVMTypeConverter::convertType(M::TupleType::get(parts, ctx));
    }
    if (auto boxchar = t.dyn_cast<BoxCharType>()) {
      // (buffer*, buffer-size)
      L::SmallVector<M::Type, 2> parts;
      // buffer*
      parts.push_back(M::LLVM::LLVMType::get(boxchar.getContext(),
                                             L::Type::getIntNTy(llvmCtx, 64)));
      // ...
      assert(false);
      return t; // fixme
    }
    if (auto boxproc = t.dyn_cast<BoxProcType>()) {
      // (function*, host-context*)
      assert(false);
      return t; // fixme
    }
    if (auto chr = t.dyn_cast<CharacterType>()) {
      L::Type *llTy = L::Type::getIntNTy(llvmCtx, chr.getSizeInBits());
      return M::LLVM::LLVMType::get(chr.getContext(), llTy);
    }
    if (auto cplx = t.dyn_cast<CplxType>()) {
      M::Type realTy =
          kindToRealType(cplx.getContext(), llvmCtx, cplx.getFKind());
      return LLVMTypeConverter::convertType(M::ComplexType::get(realTy));
    }
    if (auto derived = t.dyn_cast<RecordType>()) {
      assert(false);
      return t; // fixme
    }
    if (auto dims = t.dyn_cast<DimsType>()) {
      // [rank x <lower, extent, stride:index>]
      L::Type *i64 = L::Type::getIntNTy(llvmCtx, 64);
      if (auto rank = dims.getRank()) {
        return M::LLVM::LLVMType::get(
            dims.getContext(),
            L::ArrayType::get(L::VectorType::get(i64, 3), rank));
      }
      return M::LLVM::LLVMType::get(dims.getContext(),
                                    L::VectorType::get(i64, 3)->getPointerTo());
    }
    if (auto field = t.dyn_cast<FieldType>()) {
      return M::LLVM::LLVMType::get(field.getContext(),
                                    L::Type::getIntNTy(llvmCtx, 64));
    }
    if (auto heap = t.dyn_cast<HeapType>()) {
      return convertPointerLike(heap);
    }
    if (auto integer = t.dyn_cast<IntType>()) {
      L::Type *llTy = L::Type::getIntNTy(llvmCtx, integer.getSizeInBits());
      return M::LLVM::LLVMType::get(integer.getContext(), llTy);
    }
    if (auto log = t.dyn_cast<LogicalType>()) {
      L::Type *llTy = L::Type::getIntNTy(llvmCtx, log.getSizeInBits());
      return M::LLVM::LLVMType::get(log.getContext(), llTy);
    }
    if (auto pointer = t.dyn_cast<PointerType>()) {
      return convertPointerLike(pointer);
    }
    if (auto real = t.dyn_cast<RealType>()) {
      M::Type realTy =
          kindToRealType(real.getContext(), llvmCtx, real.getFKind());
      return LLVMTypeConverter::convertType(realTy);
    }
    if (auto ref = t.dyn_cast<ReferenceType>()) {
      return convertPointerLike(ref);
    }
    if (auto seq = t.dyn_cast<SequenceType>()) {
      M::Type eleTy = convertType(seq.getEleTy());
      auto shape = seq.getShape();
      L::SmallVector<int64_t, 4> memshape;
      if (shape.hasValue()) {
        for (auto bi : *shape) {
          if (bi.hasValue()) {
            memshape.push_back(*bi);
          } else {
            memshape.push_back(-1); // unknown shape
          }
        }
      }
      std::reverse(memshape.begin(), memshape.end());
      return LLVMTypeConverter::convertType(
          M::MemRefType::get(memshape, eleTy));
    }
    if (auto tdesc = t.dyn_cast<TypeDescType>()) {
      return convertTypeDescType(tdesc.getContext());
    }
    return LLVMTypeConverter::convertType(t);
  }
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public M::ConversionPattern {
public:
  explicit FIROpConversion(M::MLIRContext *ctx,
                           FIRToLLVMTypeConverter &lowering)
      : ConversionPattern(FromOp::getOperationName(), 1, ctx),
        lowering(lowering) {}

protected:
  L::LLVMContext &getLLVMContext() const { return lowering.getLLVMContext(); }
  M::LLVM::LLVMDialect *getDialect() const { return lowering.getDialect(); }

  FIRToLLVMTypeConverter &lowering;
};

struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto addr = M::cast<fir::AddrOfOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto alloc = M::cast<fir::AllocaOp>(op);
    auto ty = lowering.convertType(alloc.getType());
    rewriter.replaceOpWithNewOp<M::LLVM::AllocaOp>(alloc, ty, operands,
                                                   alloc.getAttrs());
    return matchSuccess();
  }
};

// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto heap = M::cast<AllocMemOp>(op);
    auto ty = lowering.convertType(heap.getType());
    // FIXME: should be a call to malloc
    rewriter.replaceOpWithNewOp<M::LLVM::AllocaOp>(heap, ty, operands,
                                                   heap.getAttrs());
    return matchSuccess();
  }
};

struct BoxAddrOpConversion : public FIROpConversion<BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxaddr = M::cast<BoxAddrOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxCharLenOpConversion : public FIROpConversion<BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxchar = M::cast<BoxCharLenOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxDimsOpConversion : public FIROpConversion<BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxdims = M::cast<BoxDimsOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxEleSizeOpConversion : public FIROpConversion<BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxelesz = M::cast<BoxEleSizeOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxIsAllocOpConversion : public FIROpConversion<BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisalloc = M::cast<BoxIsAllocOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxIsArrayOpConversion : public FIROpConversion<BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisarray = M::cast<BoxIsArrayOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxIsPtrOpConversion : public FIROpConversion<BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisptr = M::cast<BoxIsPtrOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxProcHostOpConversion : public FIROpConversion<BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxprochost = M::cast<BoxProcHostOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxRankOpConversion : public FIROpConversion<BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxrank = M::cast<BoxRankOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct BoxTypeDescOpConversion : public FIROpConversion<BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxtypedesc = M::cast<BoxTypeDescOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto call = M::cast<fir::CallOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<ConvertOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto convert = M::cast<ConvertOp>(op);
    M::Type fromTy = lowering.convertType(convert.value()->getType());
    M::Type toTy = lowering.convertType(convert.res()->getType());
    auto loc = op->getLoc();
    M::Value *op0 = operands[0];
    auto *fromLLVMTy = fromTy.cast<M::LLVM::LLVMType>().getUnderlyingType();
    auto *toLLVMTy = fromTy.cast<M::LLVM::LLVMType>().getUnderlyingType();
    M::Value *v = nullptr;
    if (fromLLVMTy == toLLVMTy) {
      rewriter.replaceOp(op, op0);
      return matchSuccess();
    }
    if (fromLLVMTy->isFloatingPointTy()) {
      if (toLLVMTy->isIntegerTy()) {
        v = rewriter.create<M::LLVM::FPToSIOp>(loc, toTy, op0);
      } else if (toLLVMTy->isFloatingPointTy()) {
        unsigned fromBits = fromLLVMTy->getIntegerBitWidth();
        unsigned toBits = toLLVMTy->getIntegerBitWidth();
        assert(fromBits != toBits);
        if (fromBits > toBits)
          v = rewriter.create<M::LLVM::FPTruncOp>(loc, toTy, op0);
        else
          v = rewriter.create<M::LLVM::FPExtOp>(loc, toTy, op0);
      }
    } else if (fromLLVMTy->isIntegerTy()) {
      if (toLLVMTy->isIntegerTy()) {
        unsigned fromBits = fromLLVMTy->getIntegerBitWidth();
        unsigned toBits = toLLVMTy->getIntegerBitWidth();
        assert(fromBits != toBits);
        if (fromBits > toBits)
          v = rewriter.create<M::LLVM::TruncOp>(loc, toTy, op0);
        else
          v = rewriter.create<M::LLVM::SExtOp>(loc, toTy, op0);
      } else if (toLLVMTy->isFloatingPointTy()) {
        v = rewriter.create<M::LLVM::SIToFPOp>(loc, toTy, op0);
      } else if (toLLVMTy->isPointerTy()) {
        v = rewriter.create<M::LLVM::IntToPtrOp>(loc, toTy, op0);
      }
    } else if (fromLLVMTy->isPointerTy()) {
      if (toLLVMTy->isIntegerTy())
        v = rewriter.create<M::LLVM::PtrToIntOp>(loc, toTy, op0);
    }
    if (v == nullptr) {
      v = rewriter.create<M::LLVM::BitcastOp>(loc, toTy, op0);
    }

    if (auto fromInt = fromTy.dyn_cast<M::IntegerType>()) {
      if (auto toInt = toTy.dyn_cast<M::IntegerType>()) {
        M::Value *v;
        if (fromInt.getIntOrFloatBitWidth() < toInt.getIntOrFloatBitWidth()) {
          v = rewriter.create<M::LLVM::SExtOp>(loc, toInt, op0);
        } else {
          v = rewriter.create<M::LLVM::TruncOp>(loc, toInt, op0);
        }
        rewriter.replaceOp(op, v);
        return matchSuccess();
      }
      // FIXME -- finish implementation
    }
    assert(false);
    return matchSuccess();
  }
};

// convert to reference to a reference to a subobject
struct CoordinateOpConversion : public FIROpConversion<CoordinateOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto coor = M::cast<CoordinateOp>(op);
    auto baseOp = coor.ref()->getDefiningOp();
    auto loc = coor.getLoc();
    if (auto box = M::dyn_cast<EmboxOp>(baseOp)) {
      // FIXME: for now assume this is always an array
      M::Value *v = rewriter.create<M::LLVM::GEPOp>(
          loc, lowering.convertType(coor.getType()), box.memref(), operands);
      rewriter.replaceOp(op, v);
      return matchSuccess();
    }
    assert(false); // FIXME
    return matchSuccess();
  }
};

// virtual call to a method in a dispatch table
struct DispatchOpConversion : public FIROpConversion<DispatchOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto dispatch = M::cast<DispatchOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// dispatch table for a Fortran derived type
struct DispatchTableOpConversion : public FIROpConversion<DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto disptable = M::cast<DispatchTableOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// entry in a dispatch table; binds a method-name to a function
struct DTEntryOpConversion : public FIROpConversion<DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto dtentry = M::cast<DTEntryOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// create a CHARACTER box
struct EmboxCharOpConversion : public FIROpConversion<EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto emboxchar = M::cast<EmboxCharOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// create a generic box on a memory reference
struct EmboxOpConversion : public FIROpConversion<EmboxOp> {
  using FIROpConversion::FIROpConversion;

  M::Value *insertValue(M::Location loc, int index, M::Value *aggr,
                        M::Value *op, M::Type partTy,
                        M::ConversionPatternRewriter &rewriter) const {
    L::SmallVector<M::Attribute, 1> attrs;
    attrs.push_back(rewriter.getI64IntegerAttr(index));
    return rewriter.create<M::LLVM::InsertValueOp>(
        loc, partTy, aggr, op, rewriter.getArrayAttr(attrs));
  }

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto embox = M::cast<EmboxOp>(op);
    auto *ctx = embox.getContext();
    M::Type convTy =
        lowering.convertType(embox.getType().cast<BoxType>().getEleTy());
    L::Type *eleTy =
        convTy.cast<M::LLVM::LLVMType>().getUnderlyingType()->getPointerTo();
    L::SmallVector<L::Type *, 2> members;
    members.push_back(eleTy);
    M::Type opTy = lowering.convertType(operands[1]->getType());
    L::Type *dimLLVMType = opTy.cast<M::LLVM::LLVMType>().getUnderlyingType();
    members.push_back(dimLLVMType);
    L::Type *boxLLVMType = L::StructType::get(getLLVMContext(), members);
    M::Type boxTy = M::LLVM::LLVMType::get(ctx, boxLLVMType);
    auto loc = embox.getLoc();
    M::Value *u = rewriter.create<M::LLVM::UndefOp>(loc, boxTy,
                                                    L::ArrayRef<M::Value *>{});
    auto *v = insertValue(loc, 0, u, operands[0], boxTy, rewriter);
    M::Type dimTy = M::LLVM::LLVMType::get(ctx, dimLLVMType);
    auto *w = insertValue(loc, 1, v, operands[1], dimTy, rewriter);
    rewriter.replaceOp(op, w);
    return matchSuccess();
  }
};

// create a procedure pointer box
struct EmboxProcOpConversion : public FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto emboxproc = M::cast<EmboxProcOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion : public FIROpConversion<ExtractValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto extractVal = M::cast<ExtractValueOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto fieldindex = M::cast<FieldIndexOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// call free function
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  M::LLVM::LLVMType getVoidPtrType() const {
    return M::LLVM::LLVMType::getInt8PtrTy(getDialect());
  }

  M::FuncOp genFreeFunc(M::Operation *op,
                        M::ConversionPatternRewriter &rewriter) const {
    M::FuncOp freeFunc =
        op->getParentOfType<M::ModuleOp>().lookupSymbol<M::FuncOp>("free");
    if (!freeFunc) {
      auto freeType = rewriter.getFunctionType(getVoidPtrType(), {});
      freeFunc = M::FuncOp::create(rewriter.getUnknownLoc(), "free", freeType);
      op->getParentOfType<M::ModuleOp>().push_back(freeFunc);
    }
    return freeFunc;
  }

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto freemem = M::cast<fir::FreeMemOp>(op);
    M::FuncOp freeFunc = genFreeFunc(op, rewriter);
    M::Value *casted = rewriter.create<M::LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(), operands[0]);
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(
        op, llvm::ArrayRef<M::Type>(), rewriter.getSymbolRefAttr(freeFunc),
        casted);
    return matchSuccess();
  }
};

struct GenDimsOpConversion : public FIROpConversion<GenDimsOp> {
  using FIROpConversion::FIROpConversion;

  // gendims(args:index, ...) ==> %v = ... : [size x <3 x index>]
  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto argsize = operands.size();
    assert(argsize % 3 == 0 && "must be a multiple of 3");
    auto loc = op->getLoc();
    auto *ctx = op->getContext();
    auto &llvmCtx = getLLVMContext();
    L::Type *i64 = L::Type::getIntNTy(llvmCtx, 64);
    L::Type *v3i64 = L::VectorType::get(i64, 3);
    M::Type vec3 = M::LLVM::LLVMType::get(ctx, v3i64);
    L::Type *av3i64 = L::ArrayType::get(v3i64, argsize / 3);
    M::Type arrvec3 = M::LLVM::LLVMType::get(ctx, av3i64);
    auto i32Type = M::LLVM::LLVMType::get(ctx, L::Type::getIntNTy(llvmCtx, 32));
    M::Value *v = rewriter.create<M::LLVM::UndefOp>(loc, arrvec3,
                                                    L::ArrayRef<M::Value *>{});
    // BUG? insertelement requires an i32 for 3rd argument
    M::Value *zero = rewriter.create<M::LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI64IntegerAttr(0));
    M::Value *one = rewriter.create<M::LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI64IntegerAttr(1));
    M::Value *two = rewriter.create<M::LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI64IntegerAttr(2));
    unsigned rank = 0;
    for (std::size_t i = 0; i < argsize;) {
      auto a1 = rewriter.create<M::LLVM::UndefOp>(loc, vec3);
      auto a2 = rewriter.create<M::LLVM::InsertElementOp>(loc, vec3, a1,
                                                          operands[i++], zero);
      auto a3 = rewriter.create<M::LLVM::InsertElementOp>(loc, vec3, a2,
                                                          operands[i++], one);
      auto a4 = rewriter.create<M::LLVM::InsertElementOp>(loc, vec3, a3,
                                                          operands[i++], two);
      L::SmallVector<M::Attribute, 1> attrs;
      attrs.push_back(rewriter.getI64IntegerAttr(rank++));
      v = rewriter.create<M::LLVM::InsertValueOp>(loc, arrvec3, v, a4,
                                                  rewriter.getArrayAttr(attrs));
    }
    rewriter.replaceOp(op, v);
    return matchSuccess();
  }
};

struct GenTypeDescOpConversion : public FIROpConversion<GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto gentypedesc = M::cast<GenTypeDescOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct GlobalEntryOpConversion : public FIROpConversion<GlobalEntryOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto globalentry = M::cast<GlobalEntryOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

class GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
public:
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto global = M::cast<fir::GlobalOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

// indirect call (via a pointer); see dispatch as well
struct ICallOpConversion : public FIROpConversion<fir::ICallOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto icall = M::cast<fir::ICallOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct InsertValueOpConversion : public FIROpConversion<InsertValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto insertVal = cast<InsertValueOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto lenparam = M::cast<LenParamIndexOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto load = M::cast<fir::LoadOp>(op);
    auto newLoad = rewriter.create<M::LLVM::LoadOp>(
        load.getLoc(), lowering.convertType(load.getType()), operands,
        load.getAttrs());
    // ???: the next line works around a bug [do we still need this?]
    load.replaceAllUsesWith(newLoad.getResult());
    rewriter.replaceOp(op, newLoad.getResult());
    return matchSuccess();
  }
};

// abstract loop construct
struct LoopOpConversion : public FIROpConversion<fir::LoopOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto loop = M::cast<fir::LoopOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

struct NoReassocOpConversion : public FIROpConversion<NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto noreassoc = M::cast<NoReassocOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

struct SelectCaseOpConversion : public FIROpConversion<SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selectcase = M::cast<SelectCaseOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

// conversion of fir::SelectOp
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto select = M::cast<fir::SelectOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

struct SelectRankOpConversion : public FIROpConversion<SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selectrank = M::cast<SelectRankOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selecttype = M::cast<SelectRankOp>(op);
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto store = M::cast<fir::StoreOp>(op);
    rewriter.replaceOpWithNewOp<M::LLVM::StoreOp>(store, operands[0],
                                                  operands[1]);
    return matchSuccess();
  }
};

// unbox a CHARACTER box value, yielding its components
struct UnboxCharOpConversion : public FIROpConversion<UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto unboxchar = M::cast<UnboxCharOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// unbox a generic box value, yielding its components
struct UnboxOpConversion : public FIROpConversion<UnboxOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto unbox = M::cast<UnboxOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// unbox a procedure box value, yielding its components
struct UnboxProcOpConversion : public FIROpConversion<UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto unboxproc = M::cast<UnboxProcOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public FIROpConversion<UndefOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto undef = M::cast<UndefOp>(op);
    M::Value *v{rewriter.create<M::LLVM::UndefOp>(
        undef.getLoc(), lowering.convertType(undef.getType()))};
    rewriter.replaceOp(op, v);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `unreachable`
struct UnreachableOpConversion : public FIROpConversion<UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    L::SmallVector<M::Block *, 1> destinations; // none
    L::SmallVector<OperandTy, 1> destOperands;  // none
    rewriter.create<M::LLVM::UnreachableOp>(
        op->getLoc(), operands, destinations, destOperands, op->getAttrs());
    return matchSuccess();
  }
};

// abstract conditional construct
struct WhereOpConversion : public FIROpConversion<fir::WhereOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto where = M::cast<fir::WhereOp>(op);
    // TODO
    assert(false);
    return matchSuccess();
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
inline void rewriteSelectConstruct(M::Operation *op, OperandTy operands,
                                   L::ArrayRef<M::Block *> dests,
                                   L::ArrayRef<OperandTy> destOperands,
                                   M::OpBuilder &rewriter) {
  L::SmallVector<M::Value *, 1> noargs;
  L::SmallVector<M::Block *, 8> blocks;
  auto loc{op->getLoc()};
  blocks.push_back(rewriter.getInsertionBlock());
  for (std::size_t i = 1; i < dests.size(); ++i)
    blocks.push_back(rewriter.createBlock(dests[0]));
  rewriter.setInsertionPointToEnd(blocks[0]);
  if (dests.size() == 1) {
    rewriter.create<M::BranchOp>(loc, dests[0], destOperands[0]);
    return;
  }
  rewriter.create<M::CondBranchOp>(loc, operands[1], dests[0], destOperands[0],
                                   blocks[1], noargs);
  for (std::size_t i = 1; i < dests.size() - 1; ++i) {
    rewriter.setInsertionPointToEnd(blocks[i]);
    rewriter.create<M::CondBranchOp>(loc, operands[i + 1], dests[i],
                                     destOperands[i], blocks[i + 1], noargs);
  }
  std::size_t last{dests.size() - 1};
  rewriter.setInsertionPointToEnd(blocks[last]);
  rewriter.create<M::BranchOp>(loc, dests[last], destOperands[last]);
}

/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect.  An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
class FIRToLLVMLoweringPass : public M::ModulePass<FIRToLLVMLoweringPass> {
public:
  void runOnModule() override {
    auto &context{getContext()};
    FIRToLLVMTypeConverter typeConverter{&context};
    M::OwningRewritePatternList patterns;
    patterns.insert<
        AddrOfOpConversion, AllocaOpConversion, AllocMemOpConversion,
        BoxAddrOpConversion, BoxCharLenOpConversion, BoxDimsOpConversion,
        BoxEleSizeOpConversion, BoxIsAllocOpConversion, BoxIsArrayOpConversion,
        BoxIsPtrOpConversion, BoxProcHostOpConversion, BoxRankOpConversion,
        BoxTypeDescOpConversion, CallOpConversion, ConvertOpConversion,
        CoordinateOpConversion, DispatchOpConversion, DispatchTableOpConversion,
        DTEntryOpConversion, EmboxCharOpConversion, EmboxOpConversion,
        EmboxProcOpConversion, ExtractValueOpConversion, FieldIndexOpConversion,
        FreeMemOpConversion, GenDimsOpConversion, GenTypeDescOpConversion,
        GlobalEntryOpConversion, GlobalOpConversion, ICallOpConversion,
        InsertValueOpConversion, LenParamIndexOpConversion, LoadOpConversion,
        LoopOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        StoreOpConversion, UnboxCharOpConversion, UnboxOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion,
        WhereOpConversion>(&context, typeConverter);
    M::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    M::populateFuncOpTypeConversionPattern(patterns, &context, typeConverter);
    M::ConversionTarget target{context};
    target.addLegalDialect<M::LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<M::FuncOp>([&](M::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    // required NOP stubs for applying a full conversion
    target.addDynamicallyLegalOp<M::ModuleOp>(
        [&](M::ModuleOp op) { return true; });
    target.addDynamicallyLegalOp<M::ModuleTerminatorOp>(
        [&](M::ModuleTerminatorOp op) { return true; });

    // apply the patterns
    if (M::failed(M::applyFullConversion(
            getModule(), target, std::move(patterns), &typeConverter))) {
      M::emitError(M::UnknownLoc::get(&context),
                   "error in converting to LLVM-IR dialect\n");
      signalPassFailure();
    }
  }
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass : public M::ModulePass<LLVMIRLoweringPass> {
  void runOnModule() override {
    if (auto llvmModule{M::translateModuleToLLVMIR(getModule())}) {
      std::error_code ec;
      auto stream{L::raw_fd_ostream("a.ll", ec, L::sys::fs::F_None)};
      stream << *llvmModule << '\n';
    } else {
      auto ctxt{getModule().getContext()};
      M::emitError(M::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<M::Pass> fir::createFIRToLLVMPass() {
  return std::make_unique<FIRToLLVMLoweringPass>();
}

// returns the predefined pass
std::unique_ptr<M::Pass> fir::createStdToLLVMPass() {
  return M::createLowerToLLVMPass();
}

std::unique_ptr<M::Pass> fir::createLLVMDialectToLLVMPass() {
  return std::make_unique<LLVMIRLoweringPass>();
}
