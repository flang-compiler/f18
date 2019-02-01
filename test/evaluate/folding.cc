// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "testing.h"
#include "../../lib/evaluate/call.h"
#include "../../lib/evaluate/expression.h"
#include "../../lib/evaluate/fold.h"
#include "../../lib/evaluate/host.h"
#include "../../lib/evaluate/type.h"
#include "../../lib/evaluate/variable.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <tuple>
// below headers needed to create a context
#include "../../lib/evaluate/common.h"
#include "../../lib/parser/char-block.h"
#include "../../lib/parser/message.h"

using namespace Fortran::evaluate;

// helper to call functions on all types from tuple
template<typename... T> struct RunOnTypes {};
template<typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

// helper to get an empty context to give to fold
FoldingContext getTestFoldingContext(Fortran::parser::Messages &m) {
  Fortran::parser::CharBlock at{};
  Fortran::parser::ContextualMessages cm{at, &m};
  return Fortran::evaluate::FoldingContext(cm);
}

// test for fold.h GetScalarConstantValue function
struct TestGetScalarConstantValue {
  template<typename T> static void Run() {
    Expr<T> exprFullyTyped{Constant<T>{Scalar<T>{}}};
    Expr<SomeKind<T::category>> exprSomeKind{exprFullyTyped};
    Expr<SomeType> exprSomeType{exprSomeKind};
    TEST(GetScalarConstantValue<T>(exprFullyTyped) != nullptr);
    TEST(GetScalarConstantValue<T>(exprSomeKind) != nullptr);
    TEST(GetScalarConstantValue<T>(exprSomeType) != nullptr);
  }
};

// test for fold.h Fold function
template<typename TR, typename TA> struct ElementalIntrinsic {
  ElementalIntrinsic(
      std::string name, Host::HostType<TR> hr, Host::HostType<TA> hx)
    : intrinsic{name}, expectedResult{Host::CastHostToFortran<TR>(hr)},
      arg{Host::CastHostToFortran<TA>(hx)} {};
  ElementalIntrinsic(std::string name, Scalar<TR> sr, Scalar<TA> sx)
    : intrinsic{name}, expectedResult{sr}, arg{sx} {};
  SpecificIntrinsic intrinsic;
  Scalar<TR> expectedResult;
  Scalar<TA> arg;
};

template<typename TR, typename TA>
void TestElementalIntrinsicFold(ElementalIntrinsic<TR, TA> &call) {
  Fortran::parser::Messages m;
  FoldingContext context{getTestFoldingContext(m)};
  std::optional<ActualArgument> x{Expr<SomeType>{
      Expr<SomeKind<TA::category>>{Expr<TA>{Constant<TA>{call.arg}}}}};
  Expr<TR> expr{Fold(context,
      Expr<TR>{FunctionRef<TR>{
          ProcedureDesignator{call.intrinsic}, ActualArguments{x}}})};
  auto *res{GetScalarConstantValue(expr)};
  if constexpr (TR::category == TypeCategory::Real) {
    std::cout << "REAL" << std::endl;
    auto *res2{GetScalarConstantValue(expr)};
    std::cout << res2 << std::endl;
    if (res) {
      std::cout << "res not null" << std::endl;
      res->AsFortran(std::cout, TR::kind) << std::endl;
    }
  }
  TEST((res && (*res == call.expectedResult)));
}

void TestIntrinsicFolding() {
  using Int4 = Type<TypeCategory::Integer, 4>;
  using Int8 = Type<TypeCategory::Integer, 8>;
  using Real3 = Type<TypeCategory::Real, 3>;
  using Complex8 = Type<TypeCategory::Complex, 8>;
  using Real4 = Type<TypeCategory::Real, 4>;
  //  using Real8 = Type<TypeCategory::Real, 8>;
  using Char1 = Type<TypeCategory::Character, 1>;

  ElementalIntrinsic<Int4, Int4> kind{"kind", 4, 0};
  TestElementalIntrinsicFold(kind);

  std::string s{"a-test-string"};
  // TODO Fill a bug against len implementation. It should accept any return
  // kind
  // ElementalIntrinsic<Int4, Char1> len{"len", static_cast<int>(s.length()),
  // s};
  ElementalIntrinsic<Int8, Char1> len{"len", static_cast<int>(s.length()), s};

  ElementalIntrinsic<Real4, Real4> acos{"acos", std::acos(0.5f), 0.5f};
  TestElementalIntrinsicFold(acos);

  auto x16{Scalar<Real3>::Convert(Host::CastHostToFortran<Real4>(1.5f)).value};
  x16.AsFortran(std::cout << "x16: ", 3) << std::endl;
  auto x4{Scalar<Real4>::Convert(x16).value};
  x4.AsFortran(std::cout << "x4: ", 4) << std::endl;
  auto res16{Scalar<Real3>::Convert(Host::CastHostToFortran<Real4>(std::acosh(
                                        Host::CastFortranToHost<Real4>(x4))))
                 .value};

  ElementalIntrinsic<Real3, Real3> acosh{"acosh", res16, x16};
  TestElementalIntrinsicFold(acosh);

  ElementalIntrinsic<Real4, Real4> acosh4{"acosh", std::acosh(1.5f), 1.5f};
  TestElementalIntrinsicFold(acosh4);

  std::complex<double> cx{-0.5, 0.5};
  ElementalIntrinsic<Complex8, Complex8> acoshC8{"acosh", std::acosh(cx), cx};
  TestElementalIntrinsicFold(acoshC8);
}

int main() {
  RunOnTypes<TestGetScalarConstantValue, AllIntrinsicTypes>::Run();
  TestIntrinsicFolding();
  return testing::Complete();
}
