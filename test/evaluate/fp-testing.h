#ifndef FORTRAN_TEST_EVALUATE_FP_TESTING_H_
#define FORTRAN_TEST_EVALUATE_FP_TESTING_H_

#include "flang/evaluate/common.h"
#include <fenv.h>

using Fortran::common::RoundingMode;
using Fortran::evaluate::RealFlags;
using Fortran::evaluate::Rounding;

class ScopedHostFloatingPointEnvironment {
public:
  ScopedHostFloatingPointEnvironment(bool treatSubnormalOperandsAsZero = false,
      bool flushSubnormalResultsToZero = false);
  ~ScopedHostFloatingPointEnvironment();
  void ClearFlags() const;
  static RealFlags CurrentFlags();
  static void SetRounding(Rounding rounding);

private:
  fenv_t originalFenv_;
  fenv_t currentFenv_;
};

#endif  // FORTRAN_TEST_EVALUATE_FP_TESTING_H_
