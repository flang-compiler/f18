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

#include "fir/Tilikum/Tilikum.h"
#include "fir/Dialect.h"
#include "fir/FIROps.h"
#include "fir/KindMapping.h"
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

/// The bridge that performs the conversion of FIR and standard dialect
/// operations to the LLVM-IR dialect.

#undef TODO
#define TODO(X)                                                                \
  (void)X;                                                                     \
  assert(false && "not yet implemented")

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
  KindMapping kindMapping;
  static L::StringMap<M::LLVM::LLVMType> identStructCache;

public:
  FIRToLLVMTypeConverter(M::MLIRContext *context)
      : LLVMTypeConverter(context), kindMapping(context) {}

  // fir.dims<r>  -->  llvm<"[r x [3 x i64]]">
  // FIXME
  M::LLVM::LLVMType dimsType() {
    auto i64Ty{M::LLVM::LLVMType::getInt64Ty(llvmDialect)};
    return M::LLVM::LLVMType::getArrayTy(i64Ty, 3);
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible. TODO: See if
  // we can't use i64 anyway; the restiction may not longer hold.
  M::LLVM::LLVMType indexType() {
    return M::LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  // FIXME: This code should be generated and follow SPOT
  M::LLVM::LLVMType convertBoxType(BoxType box) {
    // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
    L::SmallVector<M::LLVM::LLVMType, 6> parts;
    M::Type ele = box.getEleTy();
    // auto *ctx = box.getContext();
    auto eleTy = unwrap(convertType(ele));
    // buffer*
    parts.push_back(eleTy.getPointerTo());
    // ele-size
    parts.push_back(M::LLVM::LLVMType::getInt64Ty(llvmDialect));
    // version
    parts.push_back(M::LLVM::LLVMType::getInt32Ty(llvmDialect));
    // rank
    parts.push_back(M::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // type (code)
    parts.push_back(M::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // attribute
    parts.push_back(M::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // addendum
    parts.push_back(M::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // opt-dims: [0..15 x [int,int,int]]  (see fir.dims)
    // opt-type-ptr: i8* (see fir.tdesc)
    // opt-flags: i64
    // opt-len-params: [? x i64]
    return M::LLVM::LLVMType::getStructTy(llvmDialect, parts).getPointerTo();
  }

  // fir.boxchar<n>  -->  llvm<"{ ix*, i64 }">   where ix is kind mapping
  M::LLVM::LLVMType convertBoxCharType(BoxCharType boxchar) {
    auto ptrTy = convertCharType(boxchar.getEleTy()).getPointerTo();
    auto i64Ty = M::LLVM::LLVMType::getInt64Ty(llvmDialect);
    L::SmallVector<M::LLVM::LLVMType, 2> tuple = {ptrTy, i64Ty};
    return M::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  M::LLVM::LLVMType convertBoxProcType(BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto ptrTy = unwrap(funcTy).getPointerTo();
    auto i8Ty = M::LLVM::LLVMType::getInt8Ty(llvmDialect);
    L::SmallVector<M::LLVM::LLVMType, 2> tuple{ptrTy, i8Ty};
    return M::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  M::LLVM::LLVMType convertCharType(CharacterType charTy) {
    return M::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getCharacterBitsize(charTy.getFKind()));
  }

  // fir.complex<n>  -->  llvm<"{ anyfloat, anyfloat }">
  M::LLVM::LLVMType convertComplexType(KindTy kind) {
    auto realID = kindMapping.getComplexTypeID(kind);
    auto realTy = fromRealTypeID(realID, kind);
    L::SmallVector<M::LLVM::LLVMType, 2> tuple{realTy, realTy};
    return M::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  LLVM::LLVMType getDefaultInt() {
    // FIXME: this should be tied to the front-end default
    return M::LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  // fir.int<n>  -->  llvm.ix   where ix is a kind mapping
  M::LLVM::LLVMType convertIntegerType(IntType intTy) {
    return M::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getIntegerBitsize(intTy.getFKind()));
  }

  // fir.logical<n>  -->  llvm.ix  where ix is a kind mapping
  M::LLVM::LLVMType convertLogicalType(LogicalType boolTy) {
    return M::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getLogicalBitsize(boolTy.getFKind()));
  }

  template <typename A>
  M::LLVM::LLVMType convertPointerLike(A &ty) {
    return unwrap(convertType(ty.getEleTy())).getPointerTo();
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  M::LLVM::LLVMType convertRealType(KindTy kind) {
    return fromRealTypeID(kindMapping.getRealTypeID(kind), kind);
  }

  // The cache is needed to keep a unique mapping from name -> StructType
  M::LLVM::LLVMType convertRecordType(RecordType derived) {
    auto name{derived.getName()};
    auto iter{identStructCache.find(name)};
    if (iter != identStructCache.end())
      return iter->second;
    auto st{M::LLVM::LLVMType::createStructTy(llvmDialect, name)};
    identStructCache[name] = st;
    L::SmallVector<M::LLVM::LLVMType, 8> members;
    for (auto mem : derived.getTypeList())
      members.push_back(convertType(mem.second).cast<M::LLVM::LLVMType>());
    M::LLVM::LLVMType::setStructTyBody(st, members);
    return st;
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  M::LLVM::LLVMType convertSequenceType(SequenceType seq) {
    auto baseTy = unwrap(convertType(seq.getEleTy()));
    if (auto shape = seq.getShape()) {
      for (auto e : shape.getValue())
        if (e.hasValue())
          baseTy = M::LLVM::LLVMType::getArrayTy(baseTy, e.getValue());
        else
          return baseTy.getPointerTo();
      return baseTy;
    }
    return baseTy.getPointerTo();
  }

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  M::LLVM::LLVMType convertTypeDescType(M::MLIRContext *ctx) {
    return M::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  }

  /// Convert FIR types to LLVM IR dialect types
  M::Type convertType(M::Type t) override {
    if (auto box = t.dyn_cast<BoxType>())
      return convertBoxType(box);
    if (auto boxchar = t.dyn_cast<BoxCharType>())
      return convertBoxCharType(boxchar);
    if (auto boxproc = t.dyn_cast<BoxProcType>())
      return convertBoxProcType(boxproc);
    if (auto charTy = t.dyn_cast<CharacterType>())
      return convertCharType(charTy);
    if (auto cplx = t.dyn_cast<CplxType>())
      return convertComplexType(cplx.getFKind());
    if (auto derived = t.dyn_cast<RecordType>())
      return convertRecordType(derived);
    if (auto dims = t.dyn_cast<DimsType>())
      return M::LLVM::LLVMType::getArrayTy(dimsType(), dims.getRank());
    if (auto field = t.dyn_cast<FieldType>())
      return M::LLVM::LLVMType::getInt64Ty(llvmDialect);
    if (auto heap = t.dyn_cast<HeapType>())
      return convertPointerLike(heap);
    if (auto integer = t.dyn_cast<IntType>())
      return convertIntegerType(integer);
    if (auto logical = t.dyn_cast<LogicalType>())
      return convertLogicalType(logical);
    if (auto pointer = t.dyn_cast<PointerType>())
      return convertPointerLike(pointer);
    if (auto real = t.dyn_cast<RealType>())
      return convertRealType(real.getFKind());
    if (auto ref = t.dyn_cast<ReferenceType>())
      return convertPointerLike(ref);
    if (auto sequence = t.dyn_cast<SequenceType>())
      return convertSequenceType(sequence);
    if (auto tdesc = t.dyn_cast<TypeDescType>())
      return convertTypeDescType(tdesc.getContext());
    return LLVMTypeConverter::convertType(t);
  }

  /// Convert llvm::Type::TypeID to mlir::LLVM::LLVMType
  M::LLVM::LLVMType fromRealTypeID(L::Type::TypeID typeID, KindTy kind) {
    switch (typeID) {
    case L::Type::TypeID::HalfTyID:
      return M::LLVM::LLVMType::getHalfTy(llvmDialect);
    case L::Type::TypeID::FloatTyID:
      return M::LLVM::LLVMType::getFloatTy(llvmDialect);
    case L::Type::TypeID::DoubleTyID:
      return M::LLVM::LLVMType::getDoubleTy(llvmDialect);
    case L::Type::TypeID::X86_FP80TyID:
      return M::LLVM::LLVMType::getX86_FP80Ty(llvmDialect);
    case L::Type::TypeID::FP128TyID:
      return M::LLVM::LLVMType::getFP128Ty(llvmDialect);
    default:
      emitError(UnknownLoc::get(llvmDialect->getContext()))
          << "unsupported type: !fir.real<" << kind << ">";
      return {};
    }
  }

  /// HACK: cloned from LLVMTypeConverter since this is private there
  LLVM::LLVMType unwrap(Type type) {
    if (!type)
      return nullptr;
    auto *mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
    if (!wrappedLLVMType)
      emitError(UnknownLoc::get(mlirContext),
                "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
  }
};

// instantiate static data member
L::StringMap<M::LLVM::LLVMType> FIRToLLVMTypeConverter::identStructCache;

L::SmallVector<M::NamedAttribute, 4>
pruneNamedAttrDict(L::ArrayRef<M::NamedAttribute> attrs,
                   L::ArrayRef<L::StringRef> omitNames) {
  L::SmallVector<M::NamedAttribute, 4> result;
  for (auto x : attrs) {
    bool omit = false;
    for (auto o : omitNames)
      if (x.first.strref() == o) {
        omit = true;
        break;
      }
    if (!omit)
      result.push_back(x);
  }
  return result;
}

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
    auto ty = lowering.unwrap(lowering.convertType(addr.getType()));
    auto attrs = pruneNamedAttrDict(addr.getAttrs(), {"symbol"});
    rewriter.replaceOpWithNewOp<M::LLVM::AddressOfOp>(addr, ty, addr.symbol(),
                                                      attrs);
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
    auto loc = alloc.getLoc();
    auto ty = lowering.convertType(alloc.getType());
    auto ity = lowering.indexType();
    auto c1attr = rewriter.getI32IntegerAttr(1);
    auto c1 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c1attr);
    rewriter.replaceOpWithNewOp<M::LLVM::AllocaOp>(alloc, ty, c1.getResult(),
                                                   alloc.getAttrs());
    return matchSuccess();
  }
};

M::LLVM::LLVMFuncOp getMalloc(AllocMemOp op,
                              M::ConversionPatternRewriter &rewriter,
                              M::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<M::ModuleOp>();
  auto mallocFunc = module.lookupSymbol<M::LLVM::LLVMFuncOp>("malloc");
  if (!mallocFunc) {
    M::OpBuilder moduleBuilder(
        op.getParentOfType<M::ModuleOp>().getBodyRegion());
    auto voidPtrType = M::LLVM::LLVMType::getInt8PtrTy(dialect);
    auto indexType = M::LLVM::LLVMType::getInt64Ty(dialect);
    mallocFunc = moduleBuilder.create<M::LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), "malloc",
        M::LLVM::LLVMType::getFunctionTy(voidPtrType, indexType,
                                         /*isVarArg=*/false));
  }
  return mallocFunc;
}

// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto heap = M::cast<AllocMemOp>(op);
    auto ty = lowering.convertType(heap.getType());
    // FIXME: should be a call to malloc
    auto loc = heap.getLoc();
    auto ity = lowering.indexType();
    auto c1attr = rewriter.getI32IntegerAttr(1);
    auto c1 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c1attr);
    rewriter.replaceOpWithNewOp<M::LLVM::AllocaOp>(heap, ty, c1.getResult(),
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
    auto a = operands[0];
    auto loc = boxaddr.getLoc();
    auto ty = lowering.convertType(boxaddr.getType());
    auto c0attr = rewriter.getI32IntegerAttr(0);
    if (auto argty = boxaddr.val()->getType().dyn_cast<BoxType>()) {
      auto ity = lowering.indexType();
      auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
      L::SmallVector<M::Value *, 4> args({a, c0, c0});
      auto pty = lowering.unwrap(ty).getPointerTo();
      auto p = rewriter.create<M::LLVM::GEPOp>(loc, pty, args);
      rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(boxaddr, ty, p);
    } else {
      auto c0 = M::ArrayAttr::get(c0attr, boxaddr.getContext());
      rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(boxaddr, ty, a, c0);
    }
    return matchSuccess();
  }
};

struct BoxCharLenOpConversion : public FIROpConversion<BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxchar = M::cast<BoxCharLenOp>(op);
    auto a = operands[0];
    auto ty = lowering.convertType(boxchar.getType());
    auto ctx = boxchar.getContext();
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(boxchar, ty, a, c1);
    return matchSuccess();
  }
};

struct BoxDimsOpConversion : public FIROpConversion<BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxdims = M::cast<BoxDimsOp>(op);
    auto a = operands[0];
    auto dim = operands[1];
    auto loc = boxdims.getLoc();
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c7attr = rewriter.getI32IntegerAttr(7);
    auto c7 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c7attr);
    auto ty = lowering.convertType(boxdims.getResult(0)->getType());
    L::SmallVector<M::Value *, 4> args({a, c0, c7, dim});
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(boxdims, ty, p);
    return matchSuccess();
  }
};

struct BoxEleSizeOpConversion : public FIROpConversion<BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxelesz = M::cast<BoxEleSizeOp>(op);
    auto a = operands[0];
    auto loc = boxelesz.getLoc();
    auto ty = lowering.convertType(boxelesz.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c1attr = rewriter.getI32IntegerAttr(1);
    auto c1 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c1attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c1});
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(boxelesz, ty, p);
    return matchSuccess();
  }
};

struct BoxIsAllocOpConversion : public FIROpConversion<BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisalloc = M::cast<BoxIsAllocOp>(op);
    auto a = operands[0];
    auto loc = boxisalloc.getLoc();
    auto ty = lowering.convertType(boxisalloc.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c5attr = rewriter.getI32IntegerAttr(5);
    auto c5 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c5attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c5});
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto c2attr = rewriter.getI32IntegerAttr(2);
    auto ab = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c2attr);
    auto bit = rewriter.create<M::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<M::LLVM::ICmpOp>(
        boxisalloc, M::LLVM::ICmpPredicate::ne, bit, c0);
    return matchSuccess();
  }
};

struct BoxIsArrayOpConversion : public FIROpConversion<BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisarray = M::cast<BoxIsArrayOp>(op);
    auto a = operands[0];
    auto loc = boxisarray.getLoc();
    auto ty = lowering.convertType(boxisarray.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c3attr = rewriter.getI32IntegerAttr(3);
    auto c3 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c3attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c3});
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    rewriter.replaceOpWithNewOp<M::LLVM::ICmpOp>(
        boxisarray, M::LLVM::ICmpPredicate::ne, ld, c0);
    return matchSuccess();
  }
};

struct BoxIsPtrOpConversion : public FIROpConversion<BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxisptr = M::cast<BoxIsPtrOp>(op);
    auto a = operands[0];
    auto loc = boxisptr.getLoc();
    auto ty = lowering.convertType(boxisptr.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c5attr = rewriter.getI32IntegerAttr(5);
    auto c5 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c5attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c5});
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto c1attr = rewriter.getI32IntegerAttr(1);
    auto ab = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c1attr);
    auto bit = rewriter.create<M::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<M::LLVM::ICmpOp>(
        boxisptr, M::LLVM::ICmpPredicate::ne, bit, c0);
    return matchSuccess();
  }
};

struct BoxProcHostOpConversion : public FIROpConversion<BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxprochost = M::cast<BoxProcHostOp>(op);
    auto a = operands[0];
    auto ty = lowering.convertType(boxprochost.getType());
    auto ctx = boxprochost.getContext();
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(boxprochost, ty, a,
                                                         c1);
    return matchSuccess();
  }
};

struct BoxRankOpConversion : public FIROpConversion<BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxrank = M::cast<BoxRankOp>(op);
    auto a = operands[0];
    auto loc = boxrank.getLoc();
    auto ty = lowering.convertType(boxrank.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c3attr = rewriter.getI32IntegerAttr(3);
    auto c3 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c3attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c3});
    auto pty = lowering.unwrap(ty).getPointerTo();
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, pty, args);
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(boxrank, ty, p);
    return matchSuccess();
  }
};

struct BoxTypeDescOpConversion : public FIROpConversion<BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxtypedesc = M::cast<BoxTypeDescOp>(op);
    auto a = operands[0];
    auto loc = boxtypedesc.getLoc();
    auto ty = lowering.convertType(boxtypedesc.getType());
    auto ity = lowering.indexType();
    auto c0attr = rewriter.getI32IntegerAttr(0);
    auto c0 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c0attr);
    auto c4attr = rewriter.getI32IntegerAttr(4);
    auto c4 = rewriter.create<M::LLVM::ConstantOp>(loc, ity, c4attr);
    L::SmallVector<M::Value *, 4> args({a, c0, c4});
    auto pty = lowering.unwrap(ty).getPointerTo();
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, pty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto i8ptr = M::LLVM::LLVMType::getInt8PtrTy(getDialect());
    rewriter.replaceOpWithNewOp<M::LLVM::IntToPtrOp>(boxtypedesc, i8ptr, ld);
    return matchSuccess();
  }
};

// direct call LLVM function
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto call = M::cast<fir::CallOp>(op);
    L::SmallVector<M::Type, 4> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(lowering.convertType(r->getType()));
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(call, resultTys, operands,
                                                 call.getAttrs());
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
    auto fromTy_ = lowering.convertType(convert.value()->getType());
    auto fromTy = lowering.unwrap(fromTy_);
    auto toTy_ = lowering.convertType(convert.res()->getType());
    auto toTy = lowering.unwrap(toTy_);
    auto *fromLLVMTy = fromTy.getUnderlyingType();
    auto *toLLVMTy = toTy.getUnderlyingType();
    auto *op0 = operands[0];
    if (fromLLVMTy == toLLVMTy) {
      rewriter.replaceOp(convert, op0);
      return matchSuccess();
    }
    auto loc = convert.getLoc();
    M::Value *v = {};
    if (fromLLVMTy->isFloatingPointTy()) {
      if (toLLVMTy->isFloatingPointTy()) {
        unsigned fromBits = fromLLVMTy->getPrimitiveSizeInBits();
        unsigned toBits = toLLVMTy->getPrimitiveSizeInBits();
        // TODO: what if different reps (F16, BF16) are the same size?
        assert(fromBits != toBits);
        if (fromBits > toBits)
          v = rewriter.create<M::LLVM::FPTruncOp>(loc, toTy, op0);
        else
          v = rewriter.create<M::LLVM::FPExtOp>(loc, toTy, op0);
      } else if (toLLVMTy->isIntegerTy()) {
        v = rewriter.create<M::LLVM::FPToSIOp>(loc, toTy, op0);
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
      if (toLLVMTy->isIntegerTy()) {
        v = rewriter.create<M::LLVM::PtrToIntOp>(loc, toTy, op0);
      } else if (toLLVMTy->isPointerTy()) {
        v = rewriter.create<M::LLVM::BitcastOp>(loc, toTy, op0);
      }
    }
    if (v)
      rewriter.replaceOp(op, v);
    else
      emitError(loc) << "cannot convert " << fromTy_ << " to " << toTy_;
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
    TODO(coor);
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
    auto ty = lowering.convertType(dispatch.getFunctionType());
    // get the table, lookup the method, fetch the func-ptr
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(dispatch, ty, operands);
    TODO(dispatch);
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
    TODO(disptable);
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
    TODO(dtentry);
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
    auto a = operands[0];
    auto b = operands[1];
    auto loc = emboxchar.getLoc();
    auto ctx = emboxchar.getContext();
    auto ty = lowering.convertType(emboxchar.getType());
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(emboxchar, ty, r, b,
                                                        c1);
    return matchSuccess();
  }
};

// create a generic box on a memory reference
struct EmboxOpConversion : public FIROpConversion<EmboxOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto embox = M::cast<EmboxOp>(op);
    TODO(embox);
    return matchSuccess();
  }
};

// create a procedure pointer box
struct EmboxProcOpConversion : public FIROpConversion<EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto emboxproc = M::cast<EmboxProcOp>(op);
    auto a = operands[0];
    auto b = operands[1];
    auto loc = emboxproc.getLoc();
    auto ctx = emboxproc.getContext();
    auto ty = lowering.convertType(emboxproc.getType());
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(emboxproc, ty, r, b,
                                                        c1);
    return matchSuccess();
  }
};

// extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion : public FIROpConversion<fir::ExtractValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto extractVal = M::cast<ExtractValueOp>(op);
    TODO(extractVal);
    return matchSuccess();
  }
};

// Compute the offset of a field in a variable of derived type. A value of type
// field can only be used as an argument to a coordinate_of, extract_value, or
// insert_value operation. It derives it's meaning from the context of where it
// is used.
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto fieldindex = M::cast<FieldIndexOp>(op);
    rewriter.replaceOp(fieldindex, {});
    return matchSuccess();
  }
};

// Replace the fir-end op with a null
struct FirEndOpConversion : public FIROpConversion<FirEndOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {});
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
    auto freeFunc = genFreeFunc(freemem, rewriter);
    M::Value *casted = rewriter.create<M::LLVM::BitcastOp>(
        freemem.getLoc(), getVoidPtrType(), operands[0]);
    auto sym = rewriter.getSymbolRefAttr(freeFunc);
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(
        freemem, llvm::ArrayRef<M::Type>(), sym, casted);
    return matchSuccess();
  }
};

struct GenDimsOpConversion : public FIROpConversion<GenDimsOp> {
  using FIROpConversion::FIROpConversion;

  // gendims(args:index, ...) ==> %v = ... : [size x <3 x index>]
  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto gendims = M::cast<GenDimsOp>(op);
    TODO(gendims);
    return matchSuccess();
  }
};

struct GenTypeDescOpConversion : public FIROpConversion<GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto gentypedesc = M::cast<GenTypeDescOp>(op);
    TODO(gentypedesc);
    return matchSuccess();
  }
};

struct GlobalEntryOpConversion : public FIROpConversion<GlobalEntryOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto globalentry = M::cast<GlobalEntryOp>(op);
    TODO(globalentry);
    return matchSuccess();
  }
};

struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto global = M::cast<fir::GlobalOp>(op);
    auto tyAttr = M::TypeAttr::get(lowering.convertType(global.getType()));
    M::UnitAttr isConst;
    if (global.getAttrOfType<M::BoolAttr>("constant").getValue())
      isConst = M::UnitAttr::get(global.getContext());
    auto name = M::StringAttr::get(
        global.getAttrOfType<M::StringAttr>(M::SymbolTable::getSymbolAttrName())
            .getValue(),
        global.getContext());
    M::Attribute value;
    auto addrSpace = /* FIXME: hard-coded i32 here; is that ok? */
        rewriter.getI32IntegerAttr(0);
    rewriter.replaceOpWithNewOp<M::LLVM::GlobalOp>(global, tyAttr, isConst,
                                                   name, value, addrSpace);
    return matchSuccess();
  }
};

// InsertValue is the generalized instruction for the composition of new
// aggregate type values.
struct InsertValueOpConversion : public FIROpConversion<InsertValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto insertVal = cast<InsertValueOp>(op);
    TODO(insertVal);
    // rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(insertVal, ...);
    return matchSuccess();
  }
};

// Compute the index of the LEN param in the descriptor addendum.  A value of
// type field can only be used as an argument to a coordinate_of, extract_value,
// or insert_value operation. It derives it's meaning from the context of where
// it is used.  A LEN parameter cannot be an aggregate itself and thus a
// LenParamIndexOp can appear only once and must be last in the argument list.
struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto lenparam = M::cast<LenParamIndexOp>(op);
    rewriter.replaceOp(lenparam, {});
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
    auto ty = lowering.convertType(load.getType());
    auto at = load.getAttrs();
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(op, ty, operands, at);
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
    TODO(loop);
    return matchSuccess();
  }
};

// TODO: how do we want to enforce this in LLVM-IR?
struct NoReassocOpConversion : public FIROpConversion<NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto noreassoc = M::cast<NoReassocOp>(op);
    noreassoc.replaceAllUsesWith(operands[0]);
    rewriter.replaceOp(noreassoc, {});
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
    TODO(selectcase);
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
    TODO(select);
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
    TODO(selectrank);
    return matchSuccess();
  }
};

// SelectTypeOp should have already been lowered
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selecttype = M::cast<SelectRankOp>(op);
    TODO(selecttype);
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
    unboxchar.replaceAllUsesWith(operands);
    rewriter.replaceOp(unboxchar, {});
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
    unbox.replaceAllUsesWith(operands);
    rewriter.replaceOp(unbox, {});
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
    unboxproc.replaceAllUsesWith(operands);
    rewriter.replaceOp(unboxproc, {});
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
    rewriter.replaceOpWithNewOp<M::LLVM::UndefOp>(
        undef, lowering.convertType(undef.getType()));
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `unreachable`
struct UnreachableOpConversion : public FIROpConversion<UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto unreach = M::cast<UnreachableOp>(op);
    L::SmallVector<M::Block *, 1> destinations; // none
    L::SmallVector<OperandTy, 1> destOperands;  // none
    rewriter.replaceOpWithNewOp<M::LLVM::UnreachableOp>(
        unreach, operands, destinations, destOperands, unreach.getAttrs());
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
    TODO(where);
    return matchSuccess();
  }
};

// Generate code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
M::LLVM::InsertValueOp complexSum(OPTY sumop, OperandTy opnds,
                                  M::ConversionPatternRewriter &rewriter,
                                  FIRToLLVMTypeConverter &lowering) {
  auto a = opnds[0];
  auto b = opnds[1];
  auto loc = sumop.getLoc();
  auto ctx = sumop.getContext();
  auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
  auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
  auto ty = lowering.convertType(sumop.getType());
  auto x = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c0);
  auto x_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c0);
  auto rx = rewriter.create<LLVMOP>(loc, ty, x, x_);
  auto y = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c1);
  auto y_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c1);
  auto ry = rewriter.create<LLVMOP>(loc, ty, y, y_);
  auto r = rewriter.create<M::LLVM::UndefOp>(loc, ty);
  auto r_ = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, r, rx, c0);
  return rewriter.create<M::LLVM::InsertValueOp>(loc, ty, r_, ry, c1);
}

template <typename BINOP, typename LLVMOP>
void lowerRealBinaryOp(M::Operation *op, OperandTy operands,
                       M::ConversionPatternRewriter &rewriter,
                       FIRToLLVMTypeConverter &lowering) {
  auto binop = cast<BINOP>(op);
  auto ty = lowering.convertType(binop.getType());
  rewriter.replaceOpWithNewOp<LLVMOP>(binop, ty, operands);
}

struct AddfOpConversion : public FIROpConversion<fir::AddfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::AddfOp, M::LLVM::FAddOp>(op, operands, rewriter,
                                                    lowering);
    return matchSuccess();
  }
};
struct SubfOpConversion : public FIROpConversion<fir::SubfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::SubfOp, M::LLVM::FSubOp>(op, operands, rewriter,
                                                    lowering);
    return matchSuccess();
  }
};
struct MulfOpConversion : public FIROpConversion<fir::MulfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::MulfOp, M::LLVM::FMulOp>(op, operands, rewriter,
                                                    lowering);
    return matchSuccess();
  }
};
struct DivfOpConversion : public FIROpConversion<fir::DivfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::DivfOp, M::LLVM::FDivOp>(op, operands, rewriter,
                                                    lowering);
    return matchSuccess();
  }
};
struct ModfOpConversion : public FIROpConversion<fir::ModfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::ModfOp, M::LLVM::FRemOp>(op, operands, rewriter,
                                                    lowering);
    return matchSuccess();
  }
};

struct AddcOpConversion : public FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    // result: (x + x') + i(y + y')
    auto addc = cast<fir::AddcOp>(op);
    auto r = complexSum<M::LLVM::FAddOp>(addc, operands, rewriter, lowering);
    addc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(addc, r.getResult());
    return matchSuccess();
  }
};

struct SubcOpConversion : public FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    // result: (x - x') + i(y - y')
    auto subc = M::cast<fir::SubcOp>(op);
    auto r = complexSum<M::LLVM::FSubOp>(subc, operands, rewriter, lowering);
    subc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(subc, r.getResult());
    return matchSuccess();
  }
};

struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto mulc = M::cast<fir::MulcOp>(op);
    // TODO: should this just call __muldc3 ?
    // result: (xx'-yy')+i(xy'+yx')
    auto a = operands[0];
    auto b = operands[1];
    auto loc = mulc.getLoc();
    auto ctx = mulc.getContext();
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = lowering.convertType(mulc.getType());
    auto x = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, x, x_);
    auto y = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, x, y_);
    auto ri = rewriter.create<M::LLVM::FAddOp>(loc, ty, xy_, yx_);
    auto yy_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, y, y_);
    auto rr = rewriter.create<M::LLVM::FSubOp>(loc, ty, xx_, yy_);
    auto ra = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    mulc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(mulc, r.getResult());
    return matchSuccess();
  }
};

struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto divc = M::cast<fir::DivcOp>(op);
    // TODO: should this just call __divdc3 ?
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    auto a = operands[0];
    auto b = operands[1];
    auto loc = divc.getLoc();
    auto ctx = divc.getContext();
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = lowering.convertType(divc.getType());
    auto x = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, x, x_);
    auto x_x_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, x_, x_);
    auto y = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, x, y_);
    auto yy_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, y, y_);
    auto y_y_ = rewriter.create<M::LLVM::FMulOp>(loc, ty, y_, y_);
    auto d = rewriter.create<M::LLVM::FAddOp>(loc, ty, x_x_, y_y_);
    auto rrn = rewriter.create<M::LLVM::FAddOp>(loc, ty, xx_, yy_);
    auto rin = rewriter.create<M::LLVM::FSubOp>(loc, ty, yx_, xy_);
    auto rr = rewriter.create<M::LLVM::FDivOp>(loc, ty, rrn, d);
    auto ri = rewriter.create<M::LLVM::FDivOp>(loc, ty, rin, d);
    auto ra = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    divc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(divc, r.getResult());
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
struct FIRToLLVMLoweringPass : public M::ModulePass<FIRToLLVMLoweringPass> {
  void runOnModule() override {
    auto &context{getContext()};
    FIRToLLVMTypeConverter typeConverter{&context};
    M::OwningRewritePatternList patterns;
    patterns.insert<
        AddcOpConversion, AddfOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeDescOpConversion,
        CallOpConversion, ConvertOpConversion, CoordinateOpConversion,
        DispatchOpConversion, DispatchTableOpConversion, DivcOpConversion,
        DivfOpConversion, DTEntryOpConversion, EmboxCharOpConversion,
        EmboxOpConversion, EmboxProcOpConversion, FirEndOpConversion,
        ExtractValueOpConversion, FieldIndexOpConversion, FreeMemOpConversion,
        GenDimsOpConversion, GenTypeDescOpConversion, GlobalEntryOpConversion,
        GlobalOpConversion, InsertValueOpConversion, LenParamIndexOpConversion,
        LoadOpConversion, LoopOpConversion, ModfOpConversion, MulcOpConversion,
        MulfOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        StoreOpConversion, SubcOpConversion, SubfOpConversion,
        UnboxCharOpConversion, UnboxOpConversion, UnboxProcOpConversion,
        UndefOpConversion, UnreachableOpConversion, WhereOpConversion>(
        &context, typeConverter);
    M::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    M::ConversionTarget target{context};
    target.addLegalDialect<M::LLVM::LLVMDialect>();

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
      L::raw_fd_ostream stream("a.ll", ec, L::sys::fs::F_None);
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

std::unique_ptr<M::Pass> fir::createLLVMDialectToLLVMPass() {
  return std::make_unique<LLVMIRLoweringPass>();
}
