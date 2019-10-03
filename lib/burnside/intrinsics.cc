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

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace Fortran::burnside {

// Define a simple static runtime description that will be transformed into
// IntrinsicImplementation when building the IntrinsicLibrary.
namespace runtime {
enum class Type { f32, f64 };
struct StaticDescription {
  const char *name;
  const char *symbol;
  Type resultType;
  std::vector<Type> argumentTypes;
};

// TODO
// This table should be generated in a clever ways and probably shared with
// lib/evaluate intrinsic folding.
static const StaticDescription llvm[] = {
    {"abs", "llvm.fabs.f32", Type::f32, {Type::f32}},
    {"abs", "llvm.fabs.f64", Type::f64, {Type::f64}},
    {"acos", "acosf", Type::f32, {Type::f32}},
    {"acos", "acos", Type::f64, {Type::f64}},
    {"atan", "atan2f", Type::f32, {Type::f32, Type::f32}},
    {"atan", "atan2", Type::f64, {Type::f64, Type::f64}},
    {"sqrt", "llvm.sqrt.f32", Type::f32, {Type::f32}},
    {"sqrt", "llvm.sqrt.f64", Type::f64, {Type::f64}},
    {"cos", "llvm.cos.f32", Type::f32, {Type::f32}},
    {"cos", "llvm.cos.f64", Type::f64, {Type::f64}},
    {"sin", "llvm.sin.f32", Type::f32, {Type::f32}},
    {"sin", "llvm.sin.f64", Type::f64, {Type::f64}},
};

// Conversion between types of the static representation and MLIR types.
mlir::Type toMLIRType(Type t, mlir::MLIRContext &context) {
  switch (t) {
  case Type::f32: return mlir::FloatType::getF32(&context);
  case Type::f64: return mlir::FloatType::getF64(&context);
  }
}
mlir::FunctionType toMLIRFunctionType(
    const StaticDescription &func, mlir::MLIRContext &context) {
  std::vector<mlir::Type> argMLIRTypes;
  for (runtime::Type t : func.argumentTypes) {
    argMLIRTypes.push_back(toMLIRType(t, context));
  }
  return mlir::FunctionType::get(
      argMLIRTypes, toMLIRType(func.resultType, context), &context);
}
}  // runtime

std::optional<mlir::FuncOp> IntrinsicLibrary::getFunction(
    const std::string &name, const mlir::Type &type,
    mlir::OpBuilder &builder) const {
  auto module{getModule(&builder)};
  if (const auto &it{lib.find({name, type})}; it != lib.end()) {
    const IntrinsicImplementation &impl{it->second};
    if (mlir::FuncOp func{getNamedFunction(impl.symbol)}) {
      return func;
    }
    mlir::FuncOp function{createFunction(module, impl.symbol, impl.type)};
    function.setAttr("fir.intrinsic", builder.getUnitAttr());
    return function;
  } else {
    return std::nullopt;
  }
}

// So far ignore the version an only load the dummy llvm lib.
IntrinsicLibrary IntrinsicLibrary::create(
    IntrinsicLibrary::Version, mlir::MLIRContext &context) {
  Map map;
  for (const auto &func : runtime::llvm) {
    IntrinsicLibrary::Key key{
        func.name, runtime::toMLIRType(func.resultType, context)};
    IntrinsicImplementation impl{
        func.symbol, runtime::toMLIRFunctionType(func, context)};
    map.insert({key, impl});
  }
  return IntrinsicLibrary{std::move(map)};
}
}
