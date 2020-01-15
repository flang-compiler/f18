! Test resolution of type-bound generics.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  type :: t
  contains
    procedure, pass(x) :: add1 => add
    procedure, nopass :: add2 => add
    procedure :: add_real
    generic :: g => add1, add2, add_real
  end type
contains
  integer function add(x, y)
    class(t), intent(in) :: x, y
  end
  integer function add_real(x, y)
    class(t), intent(in) :: x
    real, intent(in) :: y
  end
  subroutine test1(x, y, z)
    type(t) :: x
    integer :: y
    integer :: z
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(y)
  end
  subroutine test2(x, y, z)
    type(t) :: x
    real :: y
    integer :: z
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(x, y)
  end
end
