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

#include "io.h"
#include "builder.h"
#include "runtime.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"

namespace Fortran::burnside {

/// Define actions to sort runtime functions. One actions
/// may be associated to one or more runtime function.
/// Actions are the keys in the StaticMultimapView used to
/// hold the io runtime description in a static constexpr way.
enum class IOAction { BeginExternalList, Output, EndIO };

class IORuntimeDescription : public RuntimeStaticDescription {
public:
  using Key = IOAction;
  constexpr IORuntimeDescription(
      IOAction act, const char *s, MaybeTypeCode r, TypeCodeVector a)
    : RuntimeStaticDescription{s, r, a}, key{act} {}
  static mlir::Type getIOCookieType(mlir::MLIRContext *context) {
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
template<IORuntimeDescription::Key key>
static mlir::FuncOp getIORuntimeFunction(mlir::OpBuilder &builder) {
  static constexpr auto runtimeDescription{ioRuntimeMap.find(key)};
  static_assert(runtimeDescription != ioRuntimeMap.end());
  return runtimeDescription->getFuncOp(builder);
}

/// This helper can be used to access io runtime functions that
/// are mapped to Output IOAction that must be mapped to at least one
/// runtime function but can be mapped to more functions.
/// This helper returns the function that has the same
/// mlir::FunctionType as the one seeked. It may therefore dynamically fail
/// if no function mapped to the Action has the seeked mlir::FunctionType.
static mlir::FuncOp getOutputRuntimeFunction(
    mlir::OpBuilder &builder, mlir::Type type) {
  static constexpr auto descriptionRange{ioRuntimeMap.getRange(IOA::Output)};
  static_assert(!descriptionRange.empty());

  mlir::MLIRContext *context{getModule(&builder).getContext()};
  llvm::SmallVector<mlir::Type, 2> argTypes{
      IORuntimeDescription::getIOCookieType(context), type};

  mlir::FunctionType seekedType{mlir::FunctionType::get(argTypes, {}, context)};
  for (const auto &description : descriptionRange) {
    if (description.getMLIRFunctionType(context) == seekedType) {
      return description.getFuncOp(builder);
    }
  }
  assert(false && "IO output runtime function not defined for this type");
  return {};
}

/// Lower print statement assuming a dummy runtime interface for now.
void genPrintStatement(mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value *> args) {
  mlir::ModuleOp module{getModule(&builder)};
  mlir::MLIRContext *mlirContext{module.getContext()};

  mlir::FuncOp beginFunc{
      getIORuntimeFunction<IOAction::BeginExternalList>(builder)};

  // Initiate io
  mlir::Type externalUnitType{mlir::IntegerType::get(32, mlirContext)};
  mlir::Value *defaultUnit{builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(externalUnitType, 1))};
  llvm::SmallVector<mlir::Value *, 1> beginArgs{defaultUnit};
  mlir::Value *cookie{
      builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0)};

  // Call data transfer runtime function
  for (mlir::Value *arg : args) {
    llvm::SmallVector<mlir::Value *, 1> operands{cookie, arg};
    mlir::FuncOp outputFunc{getOutputRuntimeFunction(builder, arg->getType())};
    builder.create<mlir::CallOp>(loc, outputFunc, operands);
  }

  // Terminate IO
  mlir::FuncOp endIOFunc{getIORuntimeFunction<IOAction::EndIO>(builder)};
  llvm::SmallVector<mlir::Value *, 1> endArgs{cookie};
  builder.create<mlir::CallOp>(loc, endIOFunc, endArgs);
}

}
