! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  integer :: x
  integer :: y
  integer :: z
end
module m2
  real :: y
  real :: z
  real :: w
end

use m1, xx => x, y => z
use m2
volatile w
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot change CONTIGUOUS attribute on use-associated 'w'
contiguous w
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' is use-associated from module 'm2' and cannot be re-declared
integer z
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Reference to 'y' is ambiguous
y = 1
end
