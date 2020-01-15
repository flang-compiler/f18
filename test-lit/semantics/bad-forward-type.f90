! Forward references to derived types (error cases)
! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s



!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undef' was forward-referenced but not defined
type(undef) function f1()
  call sub(f1)
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undef' was forward-referenced but not defined
type(undef) function f2() result(r)
  call sub(r)
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f3()
  call sub(f3)
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f4() result(r)
  call sub(f4)
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad' is not the name of a parameter for derived type 'pdt'
type(pdt(bad=1)) function f5()
  type :: pdt(good)
    integer, kind :: good = kind(0)
    integer(kind=good) :: n
  end type
end function

subroutine s1(q1)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undef' was forward-referenced but not defined
  implicit type(undef)(q)
end subroutine

subroutine s2(q1)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The derived type 'undefpdt' was forward-referenced but not defined
  implicit type(undefpdt(1))(q)
end subroutine

subroutine s3
  type :: t1
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'undef' not found
    type(undef) :: x
  end type
end subroutine

subroutine s4
  type :: t1
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'undefpdt' not found
    type(undefpdt(1)) :: x
  end type
end subroutine

subroutine s5(x)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'undef' not found
  type(undef) :: x
end subroutine

subroutine s6(x)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'undefpdt' not found
  type(undefpdt(1)) :: x
end subroutine

subroutine s7(x)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'undef' not found
  type, extends(undef) :: t
  end type
end subroutine
