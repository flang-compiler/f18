! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  integer :: x
  integer, private :: y
  interface operator(.foo.)
    module procedure ifoo
  end interface
  interface operator(-)
    module procedure ifoo
  end interface
  interface operator(.priv.)
    module procedure ifoo
  end interface
  interface operator(*)
    module procedure ifoo
  end interface
  private :: operator(.priv.), operator(*)
contains
  integer function ifoo(x, y)
    logical, intent(in) :: x, y
  end
end

use m1, local_x => x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' is PRIVATE in 'm1'
use m1, local_y => y
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' not found in module 'm1'
use m1, local_z => z
use m1, operator(.localfoo.) => operator(.foo.)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Operator '.bar.' not found in module 'm1'
use m1, operator(.localbar.) => operator(.bar.)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' is PRIVATE in 'm1'
use m1, only: y
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Operator '.priv.' is PRIVATE in 'm1'
use m1, only: operator(.priv.)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'operator(*)' is PRIVATE in 'm1'
use m1, only: operator(*)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' not found in module 'm1'
use m1, only: z
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' not found in module 'm1'
use m1, only: my_x => z
use m1, only: operator(.foo.)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Operator '.bar.' not found in module 'm1'
use m1, only: operator(.bar.)
use m1, only: operator(-) , ifoo
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'operator(+)' not found in module 'm1'
use m1, only: operator(+)

end
