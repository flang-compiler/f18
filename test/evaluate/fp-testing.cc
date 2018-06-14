// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "fp-testing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Fortran::evaluate::RealFlag;

ScopedHostFloatingPointEnvironment::ScopedHostFloatingPointEnvironment(
    bool treatDenormalOperandsAsZero, bool flushDenormalResultsToZero) {
  errno = 0;
  if (feholdexcept(&originalFenv_) != 0) {
    std::fprintf(stderr, "feholdexcept() failed: %s\n", std::strerror(errno));
    std::abort();
  }
  if (fegetenv(&currentFenv_) != 0) {
    std::fprintf(stderr, "fegetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
#if defined __x86_64__
  if (treatDenormalOperandsAsZero) {
    currentFenv_.__mxcsr |= 0x0040;
  } else {
    currentFenv_.__mxcsr &= ~0x0040;
  }
  if (flushDenormalResultsToZero) {
    currentFenv_.__mxcsr |= 0x8000;
  } else {
    currentFenv_.__mxcsr &= ~0x8000;
  }
#else
    // TODO others
#endif
  errno = 0;
  if (fesetenv(&currentFenv_) != 0) {
    std::fprintf(stderr, "fesetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
}

ScopedHostFloatingPointEnvironment::~ScopedHostFloatingPointEnvironment() {
  errno = 0;
  if (fesetenv(&originalFenv_) != 0) {
    std::fprintf(stderr, "fesetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
}

void ScopedHostFloatingPointEnvironment::ClearFlags() const {
  feclearexcept(FE_ALL_EXCEPT);
}

RealFlags ScopedHostFloatingPointEnvironment::CurrentFlags() {
  int exceptions = fetestexcept(FE_ALL_EXCEPT);
  RealFlags flags;
  if (exceptions & FE_INVALID) {
    flags.set(RealFlag::InvalidArgument);
  }
  if (exceptions & FE_DIVBYZERO) {
    flags.set(RealFlag::DivideByZero);
  }
  if (exceptions & FE_OVERFLOW) {
    flags.set(RealFlag::Overflow);
  }
  if (exceptions & FE_UNDERFLOW) {
    flags.set(RealFlag::Underflow);
  }
  if (exceptions & FE_INEXACT) {
    flags.set(RealFlag::Inexact);
  }
  return flags;
}

void ScopedHostFloatingPointEnvironment::SetRounding(Rounding rounding) {
  switch (rounding) {
  case Rounding::TiesToEven: fesetround(FE_TONEAREST); break;
  case Rounding::ToZero: fesetround(FE_TOWARDZERO); break;
  case Rounding::Up: fesetround(FE_UPWARD); break;
  case Rounding::Down: fesetround(FE_DOWNWARD); break;
  case Rounding::TiesAwayFromZero:
    std::fprintf(stderr, "SetRounding: TiesAwayFromZero not available");
    std::abort();
    break;
  }
}
