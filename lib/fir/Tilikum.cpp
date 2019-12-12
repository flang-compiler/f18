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
#include "fir/Attribute.h"
#include "fir/FIRDialect.h"
#include "fir/FIROps.h"
#include "fir/FIRType.h"
#include "fir/InternalNames.h"
#include "fir/KindMapping.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

/// The Tilikum bridge performs the conversion of operations from both the FIR
/// and standard dialects to the LLVM-IR dialect.
///
/// Some FIR operations may be lowered to other dialects, such as standard, but
/// some FIR operations will pass through to the Tilikum bridge.  This may be
/// necessary to preserve the semantics of the Fortran program.

#undef TODO
#define TODO(X)                                                                \
  (void)X;                                                                     \
  assert(false && "not yet implemented")

namespace L = llvm;
namespace M = mlir;

static L::cl::opt<bool>
    ClDisableFirToLLVMIR("disable-fir2llvmir",
                         L::cl::desc("disable FIR to LLVM-IR dialect pass"),
                         L::cl::init(false), L::cl::Hidden);

static L::cl::opt<bool> ClDisableLLVM("disable-llvm",
                                      L::cl::desc("disable LLVM pass"),
                                      L::cl::init(false), L::cl::Hidden);

using namespace fir;

namespace {

using SmallVecResult = L::SmallVector<M::Value *, 4>;
using OperandTy = L::ArrayRef<M::Value *>;
using AttributeTy = L::ArrayRef<M::NamedAttribute>;

const unsigned defaultAlign = 8;

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class FIRToLLVMTypeConverter : public M::LLVMTypeConverter {
  KindMapping kindMapping;
  static L::StringMap<M::LLVM::LLVMType> identStructCache;

public:
  FIRToLLVMTypeConverter(M::MLIRContext *context, NameUniquer &uniquer)
      : LLVMTypeConverter(context), kindMapping(context), uniquer(uniquer) {}

  // This returns the type of a single column. Rows are added by the caller.
  // fir.dims<r>  -->  llvm<"[r x [3 x i64]]">
  M::LLVM::LLVMType dimsType() {
    auto i64Ty{M::LLVM::LLVMType::getInt64Ty(llvmDialect)};
    return M::LLVM::LLVMType::getArrayTy(i64Ty, 3);
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  M::LLVM::LLVMType offsetType() {
    return M::LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  // i64 can be used to index into aggregates like arrays
  M::LLVM::LLVMType indexType() {
    return M::LLVM::LLVMType::getInt64Ty(llvmDialect);
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

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  M::LLVM::LLVMType convertRecordType(RecordType derived) {
    auto name{derived.getName()};
    // The cache is needed to keep a unique mapping from name -> StructType
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
    auto shape = seq.getShape();
    if (shape.size()) {
      for (auto e : shape) {
        if (e < 0)
          e = 0;
        baseTy = M::LLVM::LLVMType::getArrayTy(baseTy, e);
      }
      return baseTy;
    }
    return baseTy.getPointerTo();
  }

  // tuple<TS...>  -->  llvm<"{ ts... }">
  M::LLVM::LLVMType convertTupleType(M::TupleType tuple) {
    L::SmallVector<M::Type, 8> inMembers;
    tuple.getFlattenedTypes(inMembers);
    L::SmallVector<M::LLVM::LLVMType, 8> members;
    for (auto mem : inMembers)
      members.push_back(convertType(mem).cast<M::LLVM::LLVMType>());
    return M::LLVM::LLVMType::getStructTy(llvmDialect, members);
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
      return M::LLVM::LLVMType::getInt32Ty(llvmDialect);
    if (auto heap = t.dyn_cast<HeapType>())
      return convertPointerLike(heap);
    if (auto integer = t.dyn_cast<IntType>())
      return convertIntegerType(integer);
    if (auto field = t.dyn_cast<LenType>())
      return M::LLVM::LLVMType::getInt32Ty(llvmDialect);
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
    if (auto tuple = t.dyn_cast<M::TupleType>())
      return convertTupleType(tuple);
    if (auto none = t.dyn_cast<M::NoneType>())
      return M::LLVM::LLVMType::getStructTy(llvmDialect, {});
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

  NameUniquer &uniquer;
};

// instantiate static data member
L::StringMap<M::LLVM::LLVMType> FIRToLLVMTypeConverter::identStructCache;

/// remove `omitNames` (by name) from the attribute dictionary
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

inline M::LLVM::LLVMType getVoidPtrType(M::LLVM::LLVMDialect *dialect) {
  return M::LLVM::LLVMType::getInt8PtrTy(dialect);
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
  M::Type convertType(M::Type ty) const { return lowering.convertType(ty); }
  M::LLVM::LLVMType unwrap(M::Type ty) const { return lowering.unwrap(ty); }
  M::LLVM::LLVMType voidPtrTy() const { return getVoidPtrType(getDialect()); }

  M::LLVM::ConstantOp genConstantOffset(M::Location loc,
                                        M::ConversionPatternRewriter &rewriter,
                                        int offset) const {
    auto ity = lowering.offsetType();
    auto cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<M::LLVM::ConstantOp>(loc, ity, cattr);
  }

  FIRToLLVMTypeConverter &lowering;
};

struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto addr = M::cast<fir::AddrOfOp>(op);
    auto ty = unwrap(convertType(addr.getType()));
    auto attrs = pruneNamedAttrDict(addr.getAttrs(), {"symbol"});
    rewriter.replaceOpWithNewOp<M::LLVM::AddressOfOp>(
        addr, ty, addr.symbol().getRootReference(), attrs);
    return matchSuccess();
  }
};

M::LLVM::ConstantOp genConstantIndex(M::Location loc, M::LLVM::LLVMType ity,
                                     M::ConversionPatternRewriter &rewriter,
                                     int offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<M::LLVM::ConstantOp>(loc, ity, cattr);
}

/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto alloc = M::cast<fir::AllocaOp>(op);
    auto loc = alloc.getLoc();
    auto ity = lowering.indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto *size = c1.getResult();
    for (auto *opnd : operands)
      size = rewriter.create<M::LLVM::MulOp>(loc, ity, size, opnd);
    auto ty = convertType(alloc.getType());
    rewriter.replaceOpWithNewOp<M::LLVM::AllocaOp>(alloc, ty, size,
                                                   alloc.getAttrs());
    return matchSuccess();
  }
};

M::LLVM::LLVMFuncOp getMalloc(AllocMemOp op,
                              M::ConversionPatternRewriter &rewriter,
                              M::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<M::ModuleOp>();
  if (auto mallocFunc = module.lookupSymbol<M::LLVM::LLVMFuncOp>("malloc"))
    return mallocFunc;
  M::OpBuilder moduleBuilder(op.getParentOfType<M::ModuleOp>().getBodyRegion());
  auto indexType = M::LLVM::LLVMType::getInt64Ty(dialect);
  return moduleBuilder.create<M::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "malloc",
      M::LLVM::LLVMType::getFunctionTy(getVoidPtrType(dialect), indexType,
                                       /*isVarArg=*/false));
}

/// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto heap = M::cast<AllocMemOp>(op);
    auto ty = convertType(heap.getType());
    auto dialect = getDialect();
    auto mallocFunc = getMalloc(heap, rewriter, dialect);
    auto loc = heap.getLoc();
    auto ity = lowering.indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto *size = c1.getResult();
    for (auto *opnd : operands)
      size = rewriter.create<M::LLVM::MulOp>(loc, ity, size, opnd);
    heap.setAttr("callee", rewriter.getSymbolRefAttr(mallocFunc));
    L::SmallVector<M::Value *, 1> args{size};
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(heap, ty, args,
                                                 heap.getAttrs());
    return matchSuccess();
  }
};

/// obtain the free() function
M::LLVM::LLVMFuncOp getFree(FreeMemOp op,
                            M::ConversionPatternRewriter &rewriter,
                            M::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<M::ModuleOp>();
  if (auto freeFunc = module.lookupSymbol<M::LLVM::LLVMFuncOp>("free"))
    return freeFunc;
  M::OpBuilder moduleBuilder(op.getParentOfType<M::ModuleOp>().getBodyRegion());
  auto voidType = M::LLVM::LLVMType::getVoidTy(dialect);
  return moduleBuilder.create<M::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "free",
      M::LLVM::LLVMType::getFunctionTy(voidType, getVoidPtrType(dialect),
                                       /*isVarArg=*/false));
}

/// lower a freemem instruction into a call to free()
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto freemem = M::cast<fir::FreeMemOp>(op);
    auto dialect = getDialect();
    auto freeFunc = getFree(freemem, rewriter, dialect);
    auto bitcast = rewriter.create<M::LLVM::BitcastOp>(
        freemem.getLoc(), voidPtrTy(), operands[0]);
    freemem.setAttr("callee", rewriter.getSymbolRefAttr(freeFunc));
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(
        freemem, M::LLVM::LLVMType::getVoidTy(dialect),
        L::SmallVector<M::Value *, 1>{bitcast}, freemem.getAttrs());
    return matchSuccess();
  }
};

template <typename... ARGS>
M::LLVM::GEPOp genGEP(M::Location loc, M::LLVM::LLVMType ty,
                      M::ConversionPatternRewriter &rewriter, M::Value *base,
                      ARGS... args) {
  L::SmallVector<M::Value *, 8> cv{args...};
  return rewriter.create<M::LLVM::GEPOp>(loc, ty, base, cv);
}

/// convert to returning the first element of the box (any flavor)
struct BoxAddrOpConversion : public FIROpConversion<BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxaddr = M::cast<BoxAddrOp>(op);
    auto a = operands[0];
    auto loc = boxaddr.getLoc();
    auto ty = convertType(boxaddr.getType());
    if (auto argty = boxaddr.val()->getType().dyn_cast<BoxType>()) {
      auto c0 = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(ty).getPointerTo();
      auto p = genGEP(loc, unwrap(pty), rewriter, a, c0, c0);
      // load the pointer from the buffer
      rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(boxaddr, ty, p);
    } else {
      auto c0attr = rewriter.getI32IntegerAttr(0);
      auto c0 = M::ArrayAttr::get(c0attr, boxaddr.getContext());
      rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(boxaddr, ty, a, c0);
    }
    return matchSuccess();
  }
};

/// convert to an extractvalue for the 2nd part of the boxchar
struct BoxCharLenOpConversion : public FIROpConversion<BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxchar = M::cast<BoxCharLenOp>(op);
    auto a = operands[0];
    auto ty = convertType(boxchar.getType());
    auto ctx = boxchar.getContext();
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(boxchar, ty, a, c1);
    return matchSuccess();
  }
};

/// convert to a triple set of GEPs and loads
struct BoxDimsOpConversion : public FIROpConversion<BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto boxdims = M::cast<BoxDimsOp>(op);
    auto a = operands[0];
    auto dim = operands[1];
    auto loc = boxdims.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c7 = genConstantOffset(loc, rewriter, 7);
    auto l0 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 0, rewriter);
    auto l1 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 1, rewriter);
    auto l2 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 2, rewriter);
    rewriter.replaceOp(boxdims,
                       {l0.getResult(), l1.getResult(), l2.getResult()});
    return matchSuccess();
  }

  M::LLVM::LoadOp loadFromOffset(BoxDimsOp boxdims, M::Location loc,
                                 M::Value *a, M::LLVM::ConstantOp c0,
                                 M::LLVM::ConstantOp c7, M::Value *dim, int off,
                                 M::ConversionPatternRewriter &rewriter) const {
    auto ty = convertType(boxdims.getResult(off)->getType());
    auto pty = unwrap(ty).getPointerTo();
    auto c = genConstantOffset(loc, rewriter, off);
    auto p = genGEP(loc, pty, rewriter, a, c0, c7, dim, c);
    return rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
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
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c1 = genConstantOffset(loc, rewriter, 1);
    auto ty = convertType(boxelesz.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c1);
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
    auto ity = lowering.offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    auto ty = convertType(boxisalloc.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c5);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 2);
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
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    auto ty = convertType(boxisarray.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c3);
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
    auto ty = convertType(boxisptr.getType());
    auto ity = lowering.offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    L::SmallVector<M::Value *, 4> args{a, c0, c5};
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 1);
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
    auto ty = convertType(boxprochost.getType());
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
    auto ty = convertType(boxrank.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    L::SmallVector<M::Value *, 4> args{a, c0, c3};
    auto pty = unwrap(ty).getPointerTo();
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
    auto ty = convertType(boxtypedesc.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c4 = genConstantOffset(loc, rewriter, 4);
    L::SmallVector<M::Value *, 4> args{a, c0, c4};
    auto pty = unwrap(ty).getPointerTo();
    auto p = rewriter.create<M::LLVM::GEPOp>(loc, pty, args);
    auto ld = rewriter.create<M::LLVM::LoadOp>(loc, ty, p);
    auto i8ptr = M::LLVM::LLVMType::getInt8PtrTy(getDialect());
    rewriter.replaceOpWithNewOp<M::LLVM::IntToPtrOp>(boxtypedesc, i8ptr, ld);
    return matchSuccess();
  }
};

struct ConstantOpConversion : public FIROpConversion<fir::ConstantOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto constop = M::cast<fir::ConstantOp>(op);
    auto ty_ = constop.getType();
    auto ty = convertType(ty_);
    auto attr = constop.getValue();
    auto attr_ = attr.cast<M::StringAttr>();
    if (auto ft = ty_.dyn_cast<fir::RealType>()) {
      L::APFloat f{L::APFloat::IEEEdouble(), attr_.getValue()};
      attr = M::FloatAttr::get(M::FloatType::getF64(constop.getContext()), f);
    }
    rewriter.replaceOpWithNewOp<M::LLVM::ConstantOp>(constop, ty, attr);
    return matchSuccess();
  }
};

/// direct call LLVM function
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto call = M::cast<fir::CallOp>(op);
    L::SmallVector<M::Type, 4> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r->getType()));
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(call, resultTys, operands,
                                                 call.getAttrs());
    return matchSuccess();
  }
};

/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto cmp = M::cast<fir::CmpcOp>(op);
    auto ctxt = cmp.getContext();
    auto kind = cmp.lhs()->getType().cast<fir::CplxType>().getFKind();
    auto ty = convertType(fir::RealType::get(ctxt, kind));
    auto loc = cmp.getLoc();
    auto pos0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    L::SmallVector<M::Value *, 2> rp{
        rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, operands[0], pos0),
        rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, operands[1], pos0)};
    auto rcp = rewriter.create<M::LLVM::FCmpOp>(loc, ty, rp, cmp.getAttrs());
    auto pos1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    L::SmallVector<M::Value *, 2> ip{
        rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, operands[0], pos1),
        rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, operands[1], pos1)};
    auto icp = rewriter.create<M::LLVM::FCmpOp>(loc, ty, ip, cmp.getAttrs());
    L::SmallVector<M::Value *, 2> cp{rcp, icp};
    switch (cmp.getPredicate()) {
    case fir::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<M::LLVM::AndOp>(cmp, ty, cp);
      break;
    case fir::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<M::LLVM::OrOp>(cmp, ty, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return matchSuccess();
  }
};

struct CmpfOpConversion : public FIROpConversion<fir::CmpfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto cmp = M::cast<fir::CmpfOp>(op);
    auto type = convertType(cmp.getType());
    rewriter.replaceOpWithNewOp<M::LLVM::FCmpOp>(cmp, type, operands,
                                                 cmp.getAttrs());
    return matchSuccess();
  }
};

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<ConvertOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto convert = M::cast<ConvertOp>(op);
    auto fromTy_ = convertType(convert.value()->getType());
    auto fromTy = unwrap(fromTy_);
    auto toTy_ = convertType(convert.res()->getType());
    auto toTy = unwrap(toTy_);
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
        // FIXME: what if different reps (F16, BF16) are the same size?
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

/// virtual call to a method in a dispatch table
struct DispatchOpConversion : public FIROpConversion<DispatchOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto dispatch = M::cast<DispatchOp>(op);
    auto ty = convertType(dispatch.getFunctionType());
    // get the table, lookup the method, fetch the func-ptr
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(dispatch, ty, operands);
    TODO(dispatch);
    return matchSuccess();
  }
};

/// dispatch table for a Fortran derived type
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

/// entry in a dispatch table; binds a method-name to a function
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

/// create a CHARACTER box
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
    auto ty = convertType(emboxchar.getType());
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(emboxchar, ty, r, b,
                                                        c1);
    return matchSuccess();
  }
};

/// create a generic box on a memory reference
struct EmboxOpConversion : public FIROpConversion<EmboxOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto embox = M::cast<EmboxOp>(op);
    auto loc = embox.getLoc();
    auto dialect = getDialect();
    auto ty = unwrap(convertType(embox.getType()));
    auto alloca = genAllocaWithType(loc, ty, 24, defaultAlign, rewriter);
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto rty = unwrap(operands[0]->getType()).getPointerTo();
    auto f0p = genGEP(loc, rty, rewriter, alloca, c0, c0);
    auto f0p_ = rewriter.create<M::LLVM::BitcastOp>(loc, rty, f0p);
    rewriter.create<M::LLVM::StoreOp>(loc, operands[0], f0p_);
    auto i64Ty = M::LLVM::LLVMType::getInt64Ty(dialect);
    auto i64PtrTy = i64Ty.getPointerTo();
    auto f1p = genGEPToField(loc, i64PtrTy, rewriter, alloca, c0, 1);
    auto c0_ = rewriter.create<M::LLVM::SExtOp>(loc, i64Ty, c0);
    rewriter.create<M::LLVM::StoreOp>(loc, c0_, f1p);
    auto i32PtrTy = M::LLVM::LLVMType::getInt32Ty(dialect).getPointerTo();
    auto f2p = genGEPToField(loc, i32PtrTy, rewriter, alloca, c0, 2);
    rewriter.create<M::LLVM::StoreOp>(loc, c0, f2p);
    auto i8Ty = M::LLVM::LLVMType::getInt8Ty(dialect);
    auto i8PtrTy = M::LLVM::LLVMType::getInt8PtrTy(dialect);
    auto c0__ = rewriter.create<M::LLVM::TruncOp>(loc, i8Ty, c0);
    auto f3p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 3);
    rewriter.create<M::LLVM::StoreOp>(loc, c0__, f3p);
    auto f4p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 4);
    rewriter.create<M::LLVM::StoreOp>(loc, c0__, f4p);
    auto f5p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 5);
    rewriter.create<M::LLVM::StoreOp>(loc, c0__, f5p);
    auto f6p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 6);
    rewriter.create<M::LLVM::StoreOp>(loc, c0__, f6p);
    // FIXME: copy the dims info, etc.

    rewriter.replaceOp(embox, alloca.getResult());
    return matchSuccess();
  }

  /// Generate an alloca of size `size` and cast it to type `toTy`
  M::LLVM::BitcastOp
  genAllocaWithType(M::Location loc, M::LLVM::LLVMType toTy, unsigned size,
                    unsigned alignment,
                    M::ConversionPatternRewriter &rewriter) const {
    auto i8Ty = M::LLVM::LLVMType::getInt8PtrTy(getDialect());
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto func = M::cast<M::LLVM::LLVMFuncOp>(thisBlock->getParentOp());
    rewriter.setInsertionPointToStart(&func.front());
    auto size_ = genConstantOffset(loc, rewriter, size);
    auto al = rewriter.create<M::LLVM::AllocaOp>(loc, i8Ty, size_, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return rewriter.create<M::LLVM::BitcastOp>(loc, toTy, al);
  }

  M::LLVM::BitcastOp genGEPToField(M::Location loc, M::LLVM::LLVMType ty,
                                   M::ConversionPatternRewriter &rewriter,
                                   M::Value *base, M::Value *zero,
                                   int field) const {
    auto coff = genConstantOffset(loc, rewriter, field);
    auto gep = genGEP(loc, ty, rewriter, base, zero, coff);
    return rewriter.create<M::LLVM::BitcastOp>(loc, ty, gep);
  }
};

/// create a procedure pointer box
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
    auto ty = convertType(emboxproc.getType());
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<M::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(emboxproc, ty, r, b,
                                                        c1);
    return matchSuccess();
  }
};

/// return true if all `Value`s in `operands` are `ConstantOp`s
bool allConstants(OperandTy operands) {
  for (auto *opnd : operands) {
    if (auto defop = opnd->getDefiningOp())
      if (dyn_cast<M::LLVM::ConstantOp>(defop) ||
          dyn_cast<M::ConstantOp>(defop))
        continue;
    return false;
  }
  return true;
}

M::Attribute getValue(M::Value *value) {
  assert(value->getDefiningOp());
  if (auto v = dyn_cast<M::LLVM::ConstantOp>(value->getDefiningOp()))
    return v.value();
  if (auto v = dyn_cast<M::ConstantOp>(value->getDefiningOp()))
    return v.value();
  assert(false && "must be a constant op");
  return {};
}

template <typename A>
inline void appendTo(L::SmallVectorImpl<A> &dest, L::ArrayRef<A> from) {
  dest.append(from.begin(), from.end());
}

/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion : public FIROpConversion<fir::ExtractValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto extractVal = M::cast<ExtractValueOp>(op);
    auto ty = convertType(extractVal.getType());
    assert(allConstants(operands.drop_front(1)));
    // since all indices are constants use LLVM's extractvalue instruction
    L::SmallVector<M::Attribute, 8> attrs;
    for (int i = 1, end = operands.size(); i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    auto position = M::ArrayAttr::get(attrs, extractVal.getContext());
    rewriter.replaceOpWithNewOp<M::LLVM::ExtractValueOp>(extractVal, ty,
                                                         operands[0], position);
    return matchSuccess();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion : public FIROpConversion<InsertValueOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto insertVal = cast<InsertValueOp>(op);
    auto ty = convertType(insertVal.getType());
    assert(allConstants(operands.drop_front(2)));
    // since all indices must be constants use LLVM's insertvalue instruction
    L::SmallVector<M::Attribute, 8> attrs;
    for (int i = 2, end = operands.size(); i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    auto position = M::ArrayAttr::get(attrs, insertVal.getContext());
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(
        insertVal, ty, operands[0], operands[1], position);
    return matchSuccess();
  }
};

/// return true if all `Value`s in `operands` are not `FieldIndexOp`s
bool noFieldIndexOps(M::Operation::operand_range operands) {
  for (auto *opnd : operands) {
    if (auto defop = opnd->getDefiningOp())
      if (dyn_cast<FieldIndexOp>(defop))
        return false;
  }
  return true;
}

/// convert to reference to a reference to a subobject
struct CoordinateOpConversion : public FIROpConversion<CoordinateOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto coor = M::cast<CoordinateOp>(op);
    auto ty = convertType(coor.getType());
    auto loc = coor.getLoc();
    M::Value *base = operands[0];
    auto c0 = genConstantIndex(loc, lowering.indexType(), rewriter, 0);

    // The base can be a boxed reference or a raw reference
    if (auto boxTy = coor.ref()->getType().dyn_cast<BoxType>()) {
      if (coor.getNumOperands() == 2) {
        auto *coorPtr = *coor.coor().begin();
        auto *s = coorPtr->getDefiningOp();
        if (dyn_cast_or_null<LenParamIndexOp>(s)) {
          auto *lenParam = operands[1]; // byte offset
          auto bc = rewriter.create<M::LLVM::BitcastOp>(loc, voidPtrTy(), base);
          auto uty = unwrap(ty);
          auto gep = genGEP(loc, uty, rewriter, bc, lenParam);
          rewriter.replaceOpWithNewOp<M::LLVM::BitcastOp>(coor, uty, gep);
          return matchSuccess();
        }
      }
      auto c0_ = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(convertType(boxTy.getEleTy())).getPointerTo();
      // Extract the boxed reference
      auto p = genGEP(loc, pty, rewriter, base, c0, c0_);
      base = rewriter.create<M::LLVM::LoadOp>(loc, pty, p);
    }

    L::SmallVector<M::Value *, 8> offs{c0};
    auto indices = operands.drop_front(1);
    offs.append(indices.begin(), indices.end());
    if (noFieldIndexOps(coor.coor())) {
      // do not need to lower any field index ops, so use a GEP
      rewriter.replaceOpWithNewOp<M::LLVM::GEPOp>(coor, ty, base, offs);
      return matchSuccess();
    }

    // lower the field index ops by walking the indices
    auto bty = coor.ref()->getType().cast<BoxType>();
    M::Type baseTy = ReferenceType::get(bty.getEleTy());
    L::SmallVector<M::Value *, 8> args{c0};
    args.append(coor.coor().begin(), coor.coor().end());

    M::Value *retval = base;
    assert(offs.size() == args.size() && "must have same arity");
    unsigned pos = 0;
    for (unsigned i = 0, sz = offs.size(); i != sz; ++i) {
      assert(pos <= i);
      if (auto *defop = args[i]->getDefiningOp())
        if (auto field = dyn_cast<FieldIndexOp>(defop)) {
          auto memTy = unwrap(convertType(baseTy)).getPointerTo();
          M::Value *gep = retval;
          if (i - pos > 0)
            gep = genGEP(loc, memTy, rewriter, gep, arguments(offs, pos, i));
          auto bc = rewriter.create<M::LLVM::BitcastOp>(loc, voidPtrTy(), gep);
          auto gep_ = genGEP(loc, voidPtrTy(), rewriter, bc, offs[i]);
          pos = i + 1;
          baseTy = baseTy.cast<RecordType>().getType(field.field_id());
          retval = rewriter.create<M::LLVM::BitcastOp>(loc, convertType(baseTy),
                                                       gep_);
          continue;
        }
      if (auto ptrTy = baseTy.dyn_cast<ReferenceType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto ptrTy = baseTy.dyn_cast<fir::PointerType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto ptrTy = baseTy.dyn_cast<HeapType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto arrTy = baseTy.dyn_cast<SequenceType>()) {
        // FIXME: unchecked advance over array dims
        i += arrTy.getDimension() - 1;
        baseTy = arrTy.getEleTy();
      } else if (auto strTy = baseTy.dyn_cast<RecordType>()) {
        baseTy = strTy.getType(getIntValue(offs[i]));
      } else if (auto strTy = baseTy.dyn_cast<M::TupleType>()) {
        baseTy = strTy.getType(getIntValue(offs[i]));
      } else {
        assert(false && "unhandled type");
      }
    }
    if (pos < offs.size())
      retval = genGEP(loc, unwrap(ty), rewriter, retval,
                      arguments(offs, pos, offs.size()));
    rewriter.replaceOp(coor, retval);
    return matchSuccess();
  }

  L::SmallVector<M::Value *, 8> arguments(L::ArrayRef<M::Value *> vec,
                                          unsigned s, unsigned e) const {
    return {vec.begin() + s, vec.begin() + e};
  }

  int64_t getIntValue(M::Value *val) const {
    if (val)
      if (auto *defop = val->getDefiningOp())
        if (auto constOp = dyn_cast<M::ConstantIntOp>(defop))
          return constOp.getValue();
    assert(false && "must be a constant");
    return 0;
  }
};

/// convert a field index to a runtime function that computes the byte offset of
/// the dynamic field
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto field = M::cast<FieldIndexOp>(op);
    // call the compiler generated function to determine the byte offset of
    // the field at runtime
    auto symAttr = M::SymbolRefAttr::get(methodName(field), field.getContext());
    L::SmallVector<M::NamedAttribute, 1> attrs{
        rewriter.getNamedAttr("callee", symAttr)};
    auto ty = lowering.offsetType();
    rewriter.replaceOpWithNewOp<M::LLVM::CallOp>(field, ty, operands, attrs);
    return matchSuccess();
  }

  // constructing the name of the method
  inline static std::string methodName(FieldIndexOp field) {
    L::Twine fldName = field.field_id();
    // note: using std::string to dodge a bug in g++ 7.4.0
    std::string tyName = field.on_type().cast<RecordType>().getName();
    L::Twine methodName = "_QQOFFSETOF_" + tyName + "_" + fldName;
    return methodName.str();
  }
};

struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto lenp = M::cast<LenParamIndexOp>(op);
    auto ity = lowering.indexType();
    auto onty = lenp.getOnType();
    // size of portable descriptor
    const unsigned boxsize = 24; // FIXME
    unsigned offset = boxsize;
    // add the size of the rows of triples
    if (auto arr = onty.dyn_cast<SequenceType>()) {
      offset += 3 * arr.getDimension();
    }
    // advance over some addendum fields
    const unsigned addendumOffset = sizeof(void *) + sizeof(uint64_t);
    offset += addendumOffset;
    // add the offset into the LENs
    offset += 0; // FIXME
    auto attr = rewriter.getI64IntegerAttr(offset);
    rewriter.replaceOpWithNewOp<M::LLVM::ConstantOp>(lenp, ity, attr);
    return matchSuccess();
  }
};

/// lower the fir.end operation to a null (erasing it)
struct FirEndOpConversion : public FIROpConversion<FirEndOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

/// lower a gendims operation into a sequence of writes to a temp
struct GenDimsOpConversion : public FIROpConversion<GenDimsOp> {
  using FIROpConversion::FIROpConversion;

  // gendims(args:index, ...) ==> %v = ... : [size x <3 x index>]
  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto gendims = M::cast<GenDimsOp>(op);
    auto loc = gendims.getLoc();
    auto ty = convertType(gendims.getType());
    auto ptrTy = unwrap(ty).getPointerTo();
    auto alloca = genAlloca(loc, ptrTy, defaultAlign, rewriter);
    unsigned offIndex = 0;
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto ipty = lowering.indexType().getPointerTo();
    for (auto op : operands) {
      auto offset = genConstantOffset(loc, rewriter, offIndex);
      auto gep = genGEP(loc, ipty, rewriter, alloca, c0, offset);
      rewriter.create<M::LLVM::StoreOp>(loc, op, gep);
    }
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(gendims, ptrTy, alloca);
    return matchSuccess();
  }

  // Generate an alloca of size `size` and cast it to type `toTy`
  M::LLVM::AllocaOp genAlloca(M::Location loc, M::LLVM::LLVMType toTy,
                              unsigned alignment,
                              M::ConversionPatternRewriter &rewriter) const {
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto func = M::cast<M::LLVM::LLVMFuncOp>(thisBlock->getParentOp());
    rewriter.setInsertionPointToStart(&func.front());
    auto size = genConstantOffset(loc, rewriter, 1);
    auto rv = rewriter.create<M::LLVM::AllocaOp>(loc, toTy, size, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return rv;
  }
};

struct GenTypeDescOpConversion : public FIROpConversion<GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto gentypedesc = M::cast<GenTypeDescOp>(op);
    auto ty = unwrap(convertType(gentypedesc.getInType())).getPointerTo();
    std::string name = "fixme"; // FIXME: get the uniqued name
    rewriter.replaceOpWithNewOp<M::LLVM::AddressOfOp>(gentypedesc, ty, name);
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
    auto tyAttr = unwrap(convertType(global.getType()));
    bool isConst = global.getAttr("constant") ? true : false;
    auto name =
        global.getAttrOfType<M::StringAttr>(M::SymbolTable::getSymbolAttrName())
            .getValue();
    M::Attribute value;
    rewriter.replaceOpWithNewOp<M::LLVM::GlobalOp>(
        global, tyAttr, isConst, M::LLVM::Linkage::External, name, value);
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
    auto ty = convertType(load.getType());
    auto at = load.getAttrs();
    rewriter.replaceOpWithNewOp<M::LLVM::LoadOp>(op, ty, operands, at);
    return matchSuccess();
  }
};

// FIXME: how do we want to enforce this in LLVM-IR?
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

void genCaseLadderStep(M::Location loc, M::Value *cmp, M::Block *dest,
                       OperandTy destOps,
                       M::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock = rewriter.createBlock(dest);
  rewriter.setInsertionPointToEnd(thisBlock);
  L::SmallVector<M::Block *, 2> dest_{dest, newBlock};
  L::SmallVector<M::ValueRange, 2> destOps_{destOps, {}};
  rewriter.create<M::LLVM::CondBrOp>(loc, M::ValueRange{cmp}, dest_, destOps_);
  rewriter.setInsertionPointToEnd(newBlock);
}

/// Conversion of `fir.select_case`
///
/// TODO: lowering of CHARACTER type cases
struct SelectCaseOpConversion : public FIROpConversion<SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto selectcase = M::cast<SelectCaseOp>(op);
    auto conds = selectcase.getNumConditions();
    auto attrName = SelectCaseOp::AttrName;
    auto caseAttr = selectcase.getAttrOfType<M::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    auto ty = selectcase.getSelector()->getType();
    (void)ty;
    auto &selector = operands[0];
    unsigned nextOp = 1;
    auto loc = selectcase.getLoc();
    assert(conds > 0 && "selectcase must have cases");
    for (unsigned t = 0; t != conds; ++t) {
      auto &attr = cases[t];
      if (attr.dyn_cast_or_null<fir::PointIntervalAttr>()) {
        auto cmp = rewriter.create<M::LLVM::ICmpOp>(
            loc, M::LLVM::ICmpPredicate::eq, selector, operands[nextOp++]);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::LowerBoundAttr>()) {
        auto cmp = rewriter.create<M::LLVM::ICmpOp>(
            loc, M::LLVM::ICmpPredicate::sle, operands[nextOp++], selector);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::UpperBoundAttr>()) {
        auto cmp = rewriter.create<M::LLVM::ICmpOp>(
            loc, M::LLVM::ICmpPredicate::sle, selector, operands[nextOp++]);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
        auto cmp = rewriter.create<M::LLVM::ICmpOp>(
            loc, M::LLVM::ICmpPredicate::sle, operands[nextOp++], selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = rewriter.createBlock(destinations[t]);
        auto *newBlock2 = rewriter.createBlock(destinations[t]);
        rewriter.setInsertionPointToEnd(thisBlock);
        L::SmallVector<M::Block *, 2> dests{newBlock1, newBlock2};
        L::SmallVector<M::ValueRange, 2> destOps{{}, {}};
        rewriter.create<M::LLVM::CondBrOp>(loc, M::ValueRange{cmp}, dests,
                                           destOps);
        rewriter.setInsertionPointToEnd(newBlock1);
        auto cmp_ = rewriter.create<M::LLVM::ICmpOp>(
            loc, M::LLVM::ICmpPredicate::sle, selector, operands[nextOp++]);
        L::SmallVector<M::Block *, 2> dest2{destinations[t], newBlock2};
        L::SmallVector<M::ValueRange, 2> destOp2{destOperands[t], {}};
        rewriter.create<M::LLVM::CondBrOp>(loc, M::ValueRange{cmp_}, dest2,
                                           destOp2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(attr.dyn_cast_or_null<M::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<M::LLVM::BrOp>(
          selectcase, M::ValueRange{}, destinations[t],
          M::ValueRange{destOperands[t]});
    }
    return matchSuccess();
  }
};

template <typename OP>
void selectMatchAndRewrite(FIRToLLVMTypeConverter &lowering, M::Operation *op,
                           OperandTy operands,
                           L::ArrayRef<M::Block *> destinations,
                           L::ArrayRef<OperandTy> destOperands,
                           M::ConversionPatternRewriter &rewriter) {
  auto select = M::cast<OP>(op);

  // We could target the LLVM switch instruction, but it isn't part of the
  // LLVM IR dialect.  Create an if-then-else ladder instead.
  auto conds = select.getNumConditions();
  auto attrName = OP::AttrName;
  auto caseAttr = select.template getAttrOfType<M::ArrayAttr>(attrName);
  auto cases = caseAttr.getValue();
  auto ty = select.getSelector()->getType();
  auto ity = lowering.convertType(ty);
  auto &selector = operands[0];
  auto loc = select.getLoc();
  assert(conds > 0 && "select must have cases");
  for (unsigned t = 0; t != conds; ++t) {
    auto &attr = cases[t];
    if (auto intAttr = attr.template dyn_cast_or_null<M::IntegerAttr>()) {
      auto ci = rewriter.create<M::LLVM::ConstantOp>(
          loc, ity, rewriter.getIntegerAttr(ty, intAttr.getInt()));
      auto cmp = rewriter.create<M::LLVM::ICmpOp>(
          loc, M::LLVM::ICmpPredicate::eq, selector, ci);
      genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
      continue;
    }
    assert(attr.template dyn_cast_or_null<M::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    rewriter.replaceOpWithNewOp<M::LLVM::BrOp>(
        select, M::ValueRange{}, destinations[t],
        M::ValueRange{destOperands[t]});
  }
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowering, op, operands, destinations,
                                         destOperands, rewriter);
    return matchSuccess();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  L::ArrayRef<M::Block *> destinations,
                  L::ArrayRef<OperandTy> destOperands,
                  M::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(
        lowering, op, operands, destinations, destOperands, rewriter);
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
    auto selecttype = M::cast<SelectTypeOp>(op);
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
    rewriter.replaceOpWithNewOp<M::LLVM::UndefOp>(undef,
                                                  convertType(undef.getType()));
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
    rewriter.replaceOpWithNewOp<M::LLVM::UnreachableOp>(
        unreach, operands, L::ArrayRef<M::Block *>{},
        L::ArrayRef<M::ValueRange>{}, unreach.getAttrs());
    return matchSuccess();
  }
};

//
// Primitive operations on Real (floating-point) types
//

/// Convert a floating-point primitive
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

struct NegfOpConversion : public FIROpConversion<fir::NegfOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto neg = M::cast<fir::NegfOp>(op);
    auto ty = convertType(neg.getType());
    rewriter.replaceOpWithNewOp<M::LLVM::FNegOp>(neg, ty, operands);
    return matchSuccess();
  }
};

//
// Primitive operations on Complex types
//

/// Generate code for complex addition/subtraction
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

/// Inlined complex multiply
struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto mulc = M::cast<fir::MulcOp>(op);
    // FIXME: should this just call __muldc3 ?
    // result: (xx'-yy')+i(xy'+yx')
    auto a = operands[0];
    auto b = operands[1];
    auto loc = mulc.getLoc();
    auto ctx = mulc.getContext();
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(mulc.getType());
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

/// Inlined complex division
struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto divc = M::cast<fir::DivcOp>(op);
    // FIXME: should this just call __divdc3 ?
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    auto a = operands[0];
    auto b = operands[1];
    auto loc = divc.getLoc();
    auto ctx = divc.getContext();
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(divc.getType());
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

struct NegcOpConversion : public FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  M::PatternMatchResult
  matchAndRewrite(M::Operation *op, OperandTy operands,
                  M::ConversionPatternRewriter &rewriter) const override {
    auto neg = M::cast<fir::NegcOp>(op);
    auto ctxt = neg.getContext();
    auto ty = convertType(neg.getType());
    auto loc = neg.getLoc();
    auto c0 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    auto &o0 = operands[0];
    auto rp = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, o0, c0);
    auto nrp = rewriter.create<M::LLVM::FNegOp>(loc, ty, rp);
    auto c1 = M::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    auto ip = rewriter.create<M::LLVM::ExtractValueOp>(loc, ty, o0, c1);
    auto nip = rewriter.create<M::LLVM::FNegOp>(loc, ty, ip);
    auto r = rewriter.create<M::LLVM::InsertValueOp>(loc, ty, o0, nrp, c0);
    rewriter.replaceOpWithNewOp<M::LLVM::InsertValueOp>(neg, ty, r, nip, c1);
    return matchSuccess();
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect.  An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
struct FIRToLLVMLoweringPass : public M::ModulePass<FIRToLLVMLoweringPass> {
  FIRToLLVMLoweringPass(NameUniquer &uniquer) : uniquer{uniquer} {}

  void runOnModule() override {
    if (ClDisableFirToLLVMIR)
      return;

    auto *context{&getContext()};
    FIRToLLVMTypeConverter typeConverter{context, uniquer};
    M::OwningRewritePatternList patterns;
    patterns.insert<
        AddcOpConversion, AddfOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeDescOpConversion,
        CallOpConversion, CmpcOpConversion, CmpfOpConversion,
        ConstantOpConversion, ConvertOpConversion, CoordinateOpConversion,
        DispatchOpConversion, DispatchTableOpConversion, DivcOpConversion,
        DivfOpConversion, DTEntryOpConversion, EmboxCharOpConversion,
        EmboxOpConversion, EmboxProcOpConversion, FieldIndexOpConversion,
        FirEndOpConversion, ExtractValueOpConversion, FreeMemOpConversion,
        GenDimsOpConversion, GenTypeDescOpConversion, GlobalEntryOpConversion,
        GlobalOpConversion, InsertValueOpConversion, LenParamIndexOpConversion,
        LoadOpConversion, ModfOpConversion, MulcOpConversion, MulfOpConversion,
        NegcOpConversion, NegfOpConversion, NoReassocOpConversion,
        SelectCaseOpConversion, SelectOpConversion, SelectRankOpConversion,
        SelectTypeOpConversion, StoreOpConversion, SubcOpConversion,
        SubfOpConversion, UnboxCharOpConversion, UnboxOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion>(
        context, typeConverter);
    M::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    M::ConversionTarget target{*context};
    target.addLegalDialect<M::LLVM::LLVMDialect>();

    // required NOP stubs for applying a full conversion
    target.addDynamicallyLegalOp<M::ModuleOp>(
        [&](M::ModuleOp) { return true; });
    target.addDynamicallyLegalOp<M::ModuleTerminatorOp>(
        [&](M::ModuleTerminatorOp) { return true; });

    genDispatchTableMap();

    // apply the patterns
    if (M::failed(M::applyFullConversion(
            getModule(), target, std::move(patterns), &typeConverter))) {
      M::emitError(M::UnknownLoc::get(context),
                   "error in converting to LLVM-IR dialect\n");
      signalPassFailure();
    }
  }

private:
  void genDispatchTableMap() {
    for (auto dt : getModule().getOps<DispatchTableOp>()) {
      // FIXME
      (void)dt;
    }
  }

  NameUniquer &uniquer;
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass : public M::ModulePass<LLVMIRLoweringPass> {
  LLVMIRLoweringPass(L::StringRef outputName) : outputName{outputName} {}

  void runOnModule() override {
    if (ClDisableLLVM)
      return;

    if (auto llvmModule{M::translateModuleToLLVMIR(getModule())}) {
      std::error_code ec;
      L::raw_fd_ostream stream(outputName, ec, L::sys::fs::F_None);
      stream << *llvmModule << '\n';
      L::errs() << outputName << " written\n";
      return;
    }

    auto ctxt{getModule().getContext()};
    M::emitError(M::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  L::StringRef outputName;
};

} // namespace

std::unique_ptr<M::Pass>
fir::createFIRToLLVMPass(fir::NameUniquer &nameUniquer) {
  return std::make_unique<FIRToLLVMLoweringPass>(nameUniquer);
}

std::unique_ptr<M::Pass> fir::createLLVMDialectToLLVMPass(L::StringRef output) {
  return std::make_unique<LLVMIRLoweringPass>(output);
}
