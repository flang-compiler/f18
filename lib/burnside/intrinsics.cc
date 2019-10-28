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

#include "intrinsics.h"
#include "builder.h"
#include "complex-handler.h"
#include "fe-helper.h"
#include "fir/FIROps.h"
#include "fir/FIRType.h"
#include "runtime.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <unordered_map>
#include <utility>

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace Fortran::burnside {

/// MathRuntimeLibrary maps Fortran generic intrinsic names to runtime function
/// signatures. There is no guarantee that that runtime functions are available
/// for all intrinsic functions and possible types.
/// To be easy and fast to use, this class holds a map and uses
/// mlir::FunctionType to represent the runtime function type. This imply that
/// MathRuntimeLibrary cannot be constexpr built and requires an
/// mlir::MLIRContext to be built. Its constructor uses a constexpr table
/// description of the runtime. The runtime functions are not declared into the
/// mlir::module until there is a query that needs them. This is to avoid
/// polluting the FIR/LLVM IR dumps with unused functions.
class MathRuntimeLibrary {
public:
  /// The map key are Fortran generic intrinsic names.
  using Key = llvm::StringRef;
  struct Hash {  // Need custom hash for this kind of key
    size_t operator()(const Key &k) const { return llvm::hash_value(k); }
  };
  /// Runtime function description that is sufficient to build an
  /// mlir::FuncOp and to compare function types.
  struct RuntimeFunction {
    RuntimeFunction(llvm::StringRef n, mlir::FunctionType t)
      : symbol{n}, type{t} {};
    llvm::StringRef symbol;
    mlir::FunctionType type;
  };
  using Map = std::unordered_multimap<Key, RuntimeFunction, Hash>;

  MathRuntimeLibrary(IntrinsicLibrary::Version, mlir::MLIRContext &);

  /// Probe the intrinsic library for a certain intrinsic and get/build the
  /// related mlir::FuncOp if a runtime description is found.
  /// Also add a unit attribute "fir.runtime" to the function so that later
  /// it is possible to quickly know what function are intrinsics vs users.
  llvm::Optional<mlir::FuncOp> getFunction(
      mlir::OpBuilder &, llvm::StringRef, mlir::FunctionType) const;

private:
  mlir::FuncOp getFuncOp(
      mlir::OpBuilder &builder, const RuntimeFunction &runtime) const;
  Map library;
};

/// enum used to templatized code gen for intrinsics that are alike.
enum class Extremum { Min, Max };

/// The implementation of IntrinsicLibrary is based on a map that associates
/// Fortran intrinsics generic names to the related FIR generator functions.
/// All generator functions are member functions of the Implementation class
/// and they all take the same context argument that contains the name and
/// arguments of the Fortran intrinsics call to lower among other things.
/// A same FIR generator function may be able to generate the FIR for several
/// intrinsics. For instance genRuntimeCall tries to find a runtime
/// functions that matches the Fortran intrinsic call and generate the
/// operations to call this functions if it was found.
/// IntrinsicLibrary holds a constant MathRuntimeLibrary that it uses to
/// find and place call to math runtime functions. This library is built
/// when the Implementation is built. Because of this, Implementation is
/// not cheap to build and it should be kept as long as possible.

// TODO it is unclear how optional argument are handled
// TODO error handling -> return a code or directly emit messages ?
class IntrinsicLibrary::Implementation {
public:
  Implementation(Version v, mlir::MLIRContext &c) : runtime{v, c} {}
  inline mlir::Value *genval(mlir::Location loc, mlir::OpBuilder &builder,
      llvm::StringRef name, mlir::Type resultType,
      llvm::ArrayRef<mlir::Value *> args) const;

private:
  // Info needed by Generators is passed in Context struct to keep Generator
  // signatures modification easy.
  struct Context {
    mlir::Location loc;
    mlir::OpBuilder *builder{nullptr};
    llvm::StringRef name;
    llvm::ArrayRef<mlir::Value *> arguments;
    mlir::FunctionType funcType;
    mlir::ModuleOp getModuleOp() { return getModule(builder); }
    mlir::MLIRContext *getMLIRContext() {
      return getModule(builder).getContext();
    }
    mlir::Type getResultType() {
      assert(funcType.getNumResults() == 1);
      return funcType.getResult(0);
    }
  };

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using Generator = mlir::Value *(Implementation::*)(Context &) const;
  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  mlir::Value *genRuntimeCall(Context &) const;
  /// All generators can be combined with genWrapperCall that will build a
  /// function named "fir."+ <generic name> + "." + <result type code> and
  /// generate the intrinsic implementation inside instead of at the intrinsic
  /// call sites. This can be used to keep the FIR more readable.
  template<Generator g> mlir::Value *genWrapperCall(Context &c) const {
    return outlineInWrapper(g, c);
  }
  /// The defaultGenerator is always attempted if no mapping was found for the
  /// generic name provided.
  mlir::Value *defaultGenerator(Context &c) const {
    return genWrapperCall<&I::genRuntimeCall>(c);
  }
  mlir::Value *genConjg(Context &) const;
  template<Extremum extremum> mlir::Value *genExtremum(Context &) const;

  struct IntrinsicHanlder {
    const char *name;
    Generator generator{&I::defaultGenerator};
  };
  using I = Implementation;
  /// Table that drives the fir generation depending on the intrinsic.
  /// one to one mapping with Fortran arguments. If no mapping is
  /// defined here for a generic intrinsic, the defaultGenerator will
  /// be attempted.
  static constexpr IntrinsicHanlder handlers[]{
      {"conjg", &I::genConjg},
      {"max", &I::genExtremum<Extremum::Max>},
      {"min", &I::genExtremum<Extremum::Min>},
  };

  // helpers
  mlir::Value *outlineInWrapper(Generator, Context &) const;

  MathRuntimeLibrary runtime;
};

// helpers
static std::string getIntrinsicWrapperName(
    const llvm::StringRef &intrinsic, mlir::FunctionType funTy);
static mlir::FunctionType getFunctionType(mlir::Type resultType,
    llvm::ArrayRef<mlir::Value *> arguments, mlir::OpBuilder &);

/// Define a simple static runtime description that will be transformed into
/// RuntimeFunction when building the IntrinsicLibrary.
class MathsRuntimeStaticDescription : public RuntimeStaticDescription {
public:
  constexpr MathsRuntimeStaticDescription(
      const char *n, const char *s, MaybeTypeCode r, TypeCodeVector a)
    : RuntimeStaticDescription{s, r, a}, name{n} {}
  llvm::StringRef getName() const { return name; }

private:
  // Generic math function name
  const char *name{nullptr};
};

/// Description of the runtime functions available on the target.
using RType = typename RuntimeStaticDescription::TypeCode;
using Args = typename RuntimeStaticDescription::TypeCodeVector;
static constexpr MathsRuntimeStaticDescription llvmRuntime[] = {
    {"abs", "llvm.fabs.f32", RType::f32, Args::create<RType::f32>()},
    {"abs", "llvm.fabs.f64", RType::f64, Args::create<RType::f64>()},
    {"acos", "acosf", RType::f32, Args::create<RType::f32>()},
    {"acos", "acos", RType::f64, Args::create<RType::f64>()},
    {"atan", "atan2f", RType::f32, Args::create<RType::f32, RType::f32>()},
    {"atan", "atan2", RType::f64, Args::create<RType::f64, RType::f64>()},
    {"sqrt", "llvm.sqrt.f32", RType::f32, Args::create<RType::f32>()},
    {"sqrt", "llvm.sqrt.f64", RType::f64, Args::create<RType::f64>()},
    {"cos", "llvm.cos.f32", RType::f32, Args::create<RType::f32>()},
    {"cos", "llvm.cos.f64", RType::f64, Args::create<RType::f64>()},
    {"sin", "llvm.sin.f32", RType::f32, Args::create<RType::f32>()},
    {"sin", "llvm.sin.f64", RType::f64, Args::create<RType::f64>()},
};

static constexpr MathsRuntimeStaticDescription pgmathPreciseRuntime[] = {
    {"acos", "__pc_acos_1", RType::c32, Args::create<RType::c32>()},
    {"acos", "__pz_acos_1", RType::c64, Args::create<RType::c64>()},
    {"pow", "__pc_pow_1", RType::c32, Args::create<RType::c32, RType::c32>()},
    {"pow", "__pc_powi_1", RType::c32, Args::create<RType::c32, RType::i32>()},
    {"pow", "__pc_powk_1", RType::c32, Args::create<RType::c32, RType::i64>()},
    {"pow", "__pd_pow_1", RType::f64, Args::create<RType::f64, RType::f64>()},
    {"pow", "__pd_powi_1", RType::f64, Args::create<RType::f64, RType::i32>()},
    {"pow", "__pd_powk_1", RType::f64, Args::create<RType::f64, RType::i64>()},
    {"pow", "__ps_pow_1", RType::f32, Args::create<RType::f32, RType::f32>()},
    {"pow", "__ps_powi_1", RType::f32, Args::create<RType::f32, RType::i32>()},
    {"pow", "__ps_powk_1", RType::f32, Args::create<RType::f32, RType::i64>()},
    {"pow", "__pz_pow_1", RType::c64, Args::create<RType::c64, RType::c64>()},
    {"pow", "__pz_powi_1", RType::c64, Args::create<RType::c64, RType::i32>()},
    {"pow", "__pz_powk_1", RType::c64, Args::create<RType::c64, RType::i64>()},
    {"pow", "__mth_i_ipowi", RType::i32,
        Args::create<RType::i32, RType::i32>()},
    {"pow", "__mth_i_kpowi", RType::i64,
        Args::create<RType::i64, RType::i32>()},
    {"pow", "__mth_i_kpowk", RType::i64,
        Args::create<RType::i64, RType::i64>()},
};

// TODO : Tables above should be generated in a clever ways and probably shared
// with lib/evaluate intrinsic folding.

// Implementations

// IntrinsicLibrary implementation

IntrinsicLibrary IntrinsicLibrary::create(
    IntrinsicLibrary::Version v, mlir::MLIRContext &context) {
  IntrinsicLibrary lib{};
  lib.impl = new Implementation(v, context);
  return lib;
}

IntrinsicLibrary::~IntrinsicLibrary() { delete impl; }

mlir::Value *IntrinsicLibrary::genval(mlir::Location loc,
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<mlir::Value *> args) const {
  assert(impl);
  return impl->genval(loc, builder, name, resultType, args);
}

// MathRuntimeLibrary implementation

// Create the runtime description for the targeted library version.
// So far ignore the version an only load the dummy llvm lib and pgmath precise
MathRuntimeLibrary::MathRuntimeLibrary(
    IntrinsicLibrary::Version, mlir::MLIRContext &mlirContext) {
  for (const MathsRuntimeStaticDescription &func : llvmRuntime) {
    RuntimeFunction impl{
        func.getSymbol(), func.getMLIRFunctionType(&mlirContext)};
    library.insert({Key{func.getName()}, impl});
  }
  for (const MathsRuntimeStaticDescription &func : pgmathPreciseRuntime) {
    RuntimeFunction impl{
        func.getSymbol(), func.getMLIRFunctionType(&mlirContext)};
    library.insert({Key{func.getName()}, impl});
  }
}

mlir::FuncOp MathRuntimeLibrary::getFuncOp(
    mlir::OpBuilder &builder, const RuntimeFunction &runtime) const {
  auto module{getModule(&builder)};
  mlir::FuncOp function{getNamedFunction(module, runtime.symbol)};
  if (!function) {
    function = createFunction(module, runtime.symbol, runtime.type);
    function.setAttr("fir.runtime", builder.getUnitAttr());
  }
  return function;
}

llvm::Optional<mlir::FuncOp> MathRuntimeLibrary::getFunction(
    mlir::OpBuilder &builder, llvm::StringRef name,
    mlir::FunctionType funcType) const {
  auto range{library.equal_range(name)};
  const RuntimeFunction *bestNearMatch{nullptr};
  for (auto iter{range.first}; iter != range.second; ++iter) {
    const RuntimeFunction &impl{iter->second};
    if (funcType == impl.type) {
      return getFuncOp(builder, impl);  // exact match
    } else {
      if (!bestNearMatch &&
          impl.type.getNumResults() == funcType.getNumResults() &&
          impl.type.getNumInputs() == funcType.getNumInputs()) {
        bestNearMatch = &impl;
      } else {
        // TODO the best near match should not be the first hit.
        // It should apply rules:
        //  -> Non narrowing argument conversion are better
        //  -> The "nearest" conversions are better
      }
    }
  }
  if (bestNearMatch != nullptr) {
    return getFuncOp(builder, *bestNearMatch);
  } else {
    return {};
  }
}

// IntrinsicLibrary::Implementation implementation

mlir::Value *IntrinsicLibrary::Implementation::genval(mlir::Location loc,
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<mlir::Value *> args) const {
  Context context{
      loc, &builder, name, args, getFunctionType(resultType, args, builder)};
  for (const IntrinsicHanlder &handler : handlers) {
    if (name == handler.name) {
      assert(handler.generator != nullptr);
      return (this->*handler.generator)(context);
    }
  }
  // Try the default generator if no special handler was defined for the
  // intrinsic being called.
  return this->defaultGenerator(context);
}

static mlir::FunctionType getFunctionType(mlir::Type resultType,
    llvm::ArrayRef<mlir::Value *> arguments, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Type, 2> argumentTypes;
  for (const auto &arg : arguments) {
    assert(arg != nullptr);  // TODO think about optionals
    argumentTypes.push_back(arg->getType());
  }
  return mlir::FunctionType::get(
      argumentTypes, resultType, getModule(&builder).getContext());
}

static std::string getIntrinsicWrapperName(
    const llvm::StringRef &intrinsic, mlir::FunctionType funTy) {
  // TODO find nicer type to string infra or move this in a mangling utility
  auto addSuffix{[&](const llvm::Twine &suffix) -> std::string {
    return ("fir." + intrinsic + suffix).str();
  }};
  assert(funTy.getNumResults() == 1 && "only function mangling supported");
  mlir::Type resultType{funTy.getResult(0)};
  if (auto f{resultType.dyn_cast<mlir::FloatType>()}) {
    return addSuffix(".f" + llvm::Twine(f.getWidth()));
  } else if (auto i{resultType.dyn_cast<mlir::IntegerType>()}) {
    return addSuffix(".i" + llvm::Twine(i.getWidth()));
  } else if (auto cplx{resultType.dyn_cast<fir::CplxType>()}) {
    return addSuffix(".c" + llvm::Twine(cplx.getFKind()));
  } else if (auto real{resultType.dyn_cast<fir::RealType>()}) {
    return addSuffix(".r" + llvm::Twine(real.getFKind()));
  } else {
    assert(false);
    return addSuffix(".unknown");
  }
}

mlir::Value *IntrinsicLibrary::Implementation::outlineInWrapper(
    Generator generator, Context &context) const {
  mlir::ModuleOp module{getModule(context.builder)};
  mlir::MLIRContext *mlirContext{module.getContext()};
  std::string wrapperName{
      getIntrinsicWrapperName(context.name, context.funcType)};
  mlir::FuncOp function{getNamedFunction(module, wrapperName)};
  if (!function) {
    // First time this wrapper is needed, build it.
    function = createFunction(module, wrapperName, context.funcType);
    function.setAttr("fir.intrinsic", context.builder->getUnitAttr());
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    Context localContext{context};
    auto localBuilder{std::make_unique<mlir::OpBuilder>(function)};
    localBuilder->setInsertionPointToStart(&function.front());
    localContext.builder = &(*localBuilder);
    llvm::SmallVector<mlir::Value *, 2> localArguments;
    for (mlir::BlockArgument *bArg : function.front().getArguments()) {
      localArguments.push_back(bArg);
    }
    localContext.arguments = localArguments;
    localContext.loc = mlir::UnknownLoc::get(mlirContext);

    mlir::Value *result{(this->*generator)(localContext)};
    localBuilder->create<mlir::ReturnOp>(localContext.loc, result);
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getType() == context.funcType);
  }
  auto call{context.builder->create<mlir::CallOp>(
      context.loc, function, context.arguments)};
  return call.getResult(0);
}

mlir::Value *IntrinsicLibrary::Implementation::genRuntimeCall(
    Context &context) const {
  // Look up runtime
  mlir::FunctionType soughtFuncType{context.funcType};
  if (auto funcOp{runtime.getFunction(
          *context.builder, context.name, soughtFuncType)}) {
    mlir::FunctionType actualFuncType{funcOp->getType()};
    if (actualFuncType.getNumResults() != soughtFuncType.getNumResults() ||
        actualFuncType.getNumInputs() != soughtFuncType.getNumInputs() ||
        actualFuncType.getNumInputs() != context.arguments.size() ||
        actualFuncType.getNumResults() != 1) {
      assert(false);  // TODO better error handling
      return nullptr;
    }
    llvm::SmallVector<mlir::Value *, 2> convertedArguments;
    int i{0};
    for (mlir::Value *arg : context.arguments) {
      mlir::Type actualType{actualFuncType.getInput(i)};
      if (soughtFuncType.getInput(i) != actualType) {
        auto castedArg{context.builder->create<fir::ConvertOp>(
            context.loc, actualType, arg)};
        convertedArguments.push_back(castedArg.getResult());
      } else {
        convertedArguments.push_back(arg);
      }
      ++i;
    }
    auto call{context.builder->create<mlir::CallOp>(
        context.loc, *funcOp, convertedArguments)};
    mlir::Type soughtType{soughtFuncType.getResult(0)};
    mlir::Value *res{call.getResult(0)};
    if (actualFuncType.getResult(0) != soughtType) {
      auto castedRes{context.builder->create<fir::ConvertOp>(
          context.loc, soughtType, res)};
      return castedRes.getResult();
    } else {
      return res;
    }
  } else {
    // could not find runtime function
    assert(false && "no runtime found for this intrinsics");
    // TODO: better error handling ?
    //  - Try to have compile time check of runtime compltness ?
  }
}

// CONJG
mlir::Value *IntrinsicLibrary::Implementation::genConjg(
    Context &genCtxt) const {
  assert(genCtxt.arguments.size() == 1);
  mlir::Type resType{genCtxt.getResultType()};
  assert(resType == genCtxt.arguments[0]->getType());
  mlir::OpBuilder &builder{*genCtxt.builder};
  ComplexHandler cplxHandler{builder, genCtxt.loc};

  // TODO a negation unary would be better than a sub to zero ?
  mlir::Value *cplx{genCtxt.arguments[0]};
  mlir::Type realType{cplxHandler.getComplexPartType(cplx)};
  mlir::Value *zero{builder.create<mlir::ConstantOp>(
      genCtxt.loc, realType, builder.getZeroAttr(realType))};
  mlir::Value *imag{cplxHandler.extract<ComplexHandler::Part::Imag>(cplx)};
  mlir::Value *negImag{
      genCtxt.builder->create<mlir::SubFOp>(genCtxt.loc, zero, imag)};
  return cplxHandler.insert<ComplexHandler::Part::Imag>(cplx, negImag);
}

static mlir::FuncOp getMergeFunc(mlir::Type type, mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Type, 3> argTypes{
      type, type, fir::LogicalType::get(module.getContext(), 4)};
  auto mergeType{
      mlir::FunctionType::get(argTypes, {type}, module.getContext())};
  auto mergeName{getIntrinsicWrapperName("merge", mergeType)};
  return createFunction(module, mergeName, mergeType);
}

template<Extremum extremum>
static mlir::Value *createCompare(mlir::Location loc, mlir::OpBuilder &builder,
    mlir::Value *left, mlir::Value *right) {
  static constexpr auto integerPredicate{extremum == Extremum::Max
          ? mlir::CmpIPredicate::sgt
          : mlir::CmpIPredicate::slt};
  static constexpr auto realPredicate{extremum == Extremum::Max
          ? fir::CmpFPredicate::UGT
          : fir::CmpFPredicate::ULT};
  auto type{left->getType()};
  mlir::Value *result{nullptr};
  if (type.isa<mlir::FloatType>() || type.isa<fir::RealType>()) {
    // TODO: The type we get from CmpfOp is a bit weird (it is the operands
    // types, and not a boolean).
    result = builder.create<fir::CmpfOp>(loc, realPredicate, left, right);
  } else if (type.isa<mlir::IntegerType>()) {
    result = builder.create<mlir::CmpIOp>(loc, integerPredicate, left, right);
  } else if (type.isa<fir::CharacterType>()) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
  }
  assert(result);
  // TODO which logical type should comparisons return ?
  // This is also currently not observed in convert-expr
  auto resTy{fir::LogicalType::get(getModule(&builder).getContext(), 4)};
  return builder.create<fir::ConvertOp>(loc, resTy, result);
}

// MIN and MAX
// Extremum intrinsic are lowered using the MERGE intrinsic
// to avoid introducing branches.
template<Extremum extremum>
mlir::Value *IntrinsicLibrary::Implementation::genExtremum(
    Context &genCtxt) const {
  auto &builder{*genCtxt.builder};
  auto loc{genCtxt.loc};
  assert(genCtxt.arguments.size() >= 2);
  auto *result{genCtxt.arguments[0]};
  auto mergeFunc{getMergeFunc(result->getType(), genCtxt.getModuleOp())};
  for (auto *arg : genCtxt.arguments.drop_front()) {
    auto mask{createCompare<extremum>(loc, builder, result, arg)};
    llvm::SmallVector<mlir::Value *, 3> mergeArgs{result, arg, mask};
    result =
        builder.create<mlir::CallOp>(loc, mergeFunc, mergeArgs).getResult(0);
  }
  return result;
}

}  // namespace Fortran::burnside
