//===-- lib/lower/io.cc -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io.h"
#include "../parser/parse-tree.h"
#include "../semantics/tools.h"
#include "bridge.h"
#include "builder.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "runtime.h"
#include <cassert>

namespace Br = Fortran::lower;
namespace M = mlir;
namespace Pa = Fortran::parser;

using namespace Fortran;
using namespace Fortran::lower;

namespace {

/// Define actions to sort runtime functions. One actions
/// may be associated to one or more runtime function.
/// Actions are the keys in the StaticMultimapView used to
/// hold the io runtime description in a static constexpr way.
enum class IOAction { BeginExternalList, Output, EndIO };

class IORuntimeDescription : public RuntimeStaticDescription {
public:
  using Key = IOAction;
  constexpr IORuntimeDescription(IOAction act, const char *s, MaybeTypeCode r,
                                 TypeCodeVector a)
      : RuntimeStaticDescription{s, r, a}, key{act} {}
  static M::Type getIOCookieType(M::MLIRContext *context) {
    return getMLIRType(TypeCode::IOCookie, context);
  }
  IOAction key;
};

using IORuntimeMap = StaticMultimapView<IORuntimeDescription>;

using RT = RuntimeStaticDescription;
using RType = typename RT::TypeCode;
using Args = typename RT::TypeCodeVector;
using IOA = IOAction;

/// This is were the IO runtime are to be described.
/// The array need to be sorted on the Actions.
/// Experimental runtime for now.
static constexpr IORuntimeDescription ioRuntimeTable[]{
    {IOA::BeginExternalList, "__F18IOa_BeginExternalListOutput",
     RType::IOCookie, Args::create<RType::i32>()},
    {IOA::Output, "__F18IOa_OutputInteger64", RT::voidTy,
     Args::create<RType::IOCookie, RType::i64>()},
    {IOA::Output, "__F18IOa_OutputReal64", RT::voidTy,
     Args::create<RType::IOCookie, RType::f64>()},
    {IOA::EndIO, "__F18IOa_EndIOStatement", RT::voidTy,
     Args::create<RType::IOCookie>()},
};

static constexpr IORuntimeMap ioRuntimeMap{ioRuntimeTable};

/// This helper can be used to access io runtime functions that
/// are mapped to an IOAction that must be mapped to one and
/// exactly one runtime function. This constraint is enforced
/// at compile time. This search is resolved at compile time.
template <IORuntimeDescription::Key key>
static M::FuncOp getIORuntimeFunction(M::OpBuilder &builder) {
  static constexpr auto runtimeDescription{ioRuntimeMap.find(key)};
  static_assert(runtimeDescription != ioRuntimeMap.end());
  return runtimeDescription->getFuncOp(builder);
}

/// This helper can be used to access io runtime functions that
/// are mapped to Output IOAction that must be mapped to at least one
/// runtime function but can be mapped to more functions.
/// This helper returns the function that has the same
/// M::FunctionType as the one seeked. It may therefore dynamically fail
/// if no function mapped to the Action has the seeked M::FunctionType.
static M::FuncOp getOutputRuntimeFunction(M::OpBuilder &builder, M::Type type) {
  static constexpr auto descriptionRange{ioRuntimeMap.getRange(IOA::Output)};
  static_assert(!descriptionRange.empty());

  M::MLIRContext *context{getModule(&builder).getContext()};
  llvm::SmallVector<M::Type, 2> argTypes{
      IORuntimeDescription::getIOCookieType(context), type};

  M::FunctionType seekedType{M::FunctionType::get(argTypes, {}, context)};
  for (const auto &description : descriptionRange) {
    if (description.getMLIRFunctionType(context) == seekedType) {
      return description.getFuncOp(builder);
    }
  }
  assert(false && "IO output runtime function not defined for this type");
  return {};
}

/// Lower print statement assuming a dummy runtime interface for now.
void lowerPrintStatement(M::OpBuilder &builder, M::Location loc,
                         M::ValueRange args) {
  M::ModuleOp module{getModule(&builder)};
  M::MLIRContext *mlirContext{module.getContext()};

  M::FuncOp beginFunc{
      getIORuntimeFunction<IOAction::BeginExternalList>(builder)};

  // Initiate io
  M::Type externalUnitType{M::IntegerType::get(32, mlirContext)};
  M::Value defaultUnit{builder.create<M::ConstantOp>(
      loc, builder.getIntegerAttr(externalUnitType, 1))};
  llvm::SmallVector<M::Value, 1> beginArgs{defaultUnit};
  M::Value cookie{
      builder.create<M::CallOp>(loc, beginFunc, beginArgs).getResult(0)};

  // Call data transfer runtime function
  for (M::Value arg : args) {
    llvm::SmallVector<M::Value, 1> operands{cookie, arg};
    M::FuncOp outputFunc{getOutputRuntimeFunction(builder, arg->getType())};
    builder.create<M::CallOp>(loc, outputFunc, operands);
  }

  // Terminate IO
  M::FuncOp endIOFunc{getIORuntimeFunction<IOAction::EndIO>(builder)};
  llvm::SmallVector<M::Value, 1> endArgs{cookie};
  builder.create<M::CallOp>(loc, endIOFunc, endArgs);
}

} // namespace

void Br::genBackspaceStatement(AbstractConverter &, const Pa::BackspaceStmt &) {
  assert(false);
}

void Br::genCloseStatement(AbstractConverter &, const Pa::CloseStmt &) {
  assert(false);
}

void Br::genEndfileStatement(AbstractConverter &, const Pa::EndfileStmt &) {
  assert(false);
}

void Br::genFlushStatement(AbstractConverter &, const Pa::FlushStmt &) {
  assert(false);
}

void Br::genInquireStatement(AbstractConverter &, const Pa::InquireStmt &) {
  assert(false);
}

void Br::genOpenStatement(AbstractConverter &, const Pa::OpenStmt &) {
  assert(false);
}

void Br::genPrintStatement(Br::AbstractConverter &converter,
                           const Pa::PrintStmt &stmt) {
  llvm::SmallVector<M::Value, 4> args;
  for (auto &item : std::get<std::list<Pa::OutputItem>>(stmt.t)) {
    if (auto *pe{std::get_if<Pa::Expr>(&item.u)}) {
      auto loc{converter.genLocation(pe->source)};
      args.push_back(converter.genExprValue(*semantics::GetExpr(*pe), &loc));
    } else {
      assert(false); // TODO implied do
    }
  }
  lowerPrintStatement(converter.getOpBuilder(), converter.getCurrentLocation(),
                      args);
}

void Br::genReadStatement(AbstractConverter &, const Pa::ReadStmt &) {
  assert(false);
}

void Br::genRewindStatement(AbstractConverter &, const Pa::RewindStmt &) {
  assert(false);
}

void Br::genWriteStatement(AbstractConverter &, const Pa::WriteStmt &) {
  assert(false);
}
