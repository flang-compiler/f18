! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Declaration of 'x' conflicts with its use as internal procedure
  real :: x
contains
  subroutine x
  end
end

module m
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Declaration of 'x' conflicts with its use as module procedure
  real :: x
contains
  subroutine x
  end
end
