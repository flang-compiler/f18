! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  type :: t
    real :: y
  end type
end module

use m
implicit type(t)(x)
z = x%y  !OK: x is type(t)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'w' is not an object of derived type; it is implicitly typed
z = w%y
end
