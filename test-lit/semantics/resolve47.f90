! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Logical constant '.true.' may not be used as a defined operator
  interface operator(.TRUE.)
  end interface
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Logical constant '.false.' may not be used as a defined operator
  generic :: operator(.false.) => bar
end

module m2
  interface operator(+)
    module procedure foo
  end interface
  interface operator(.foo.)
    module procedure foo
  end interface
  interface operator(.ge.)
    module procedure bar
  end interface
contains
  integer function foo(x, y)
    logical, intent(in) :: x, y
    foo = 0
  end
  logical function bar(x, y)
    complex, intent(in) :: x, y
    bar = .false.
  end
end

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Intrinsic operator '.le.' may not be used as a defined operator
use m2, only: operator(.le.) => operator(.ge.)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Intrinsic operator '.not.' may not be used as a defined operator
use m2, only: operator(.not.) => operator(.foo.)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Logical constant '.true.' may not be used as a defined operator
use m2, only: operator(.true.) => operator(.foo.)
end
