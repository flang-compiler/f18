! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  type t1
  end type
  type t3
  end type
  interface
    subroutine s1(x)
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't1' from host is not accessible
      import :: t1
      type(t1) :: x
      integer :: t1
    end subroutine
    subroutine s2()
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't2' not found in host scope
      import :: t2
    end subroutine
    subroutine s3(x, y)
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 't1' not found
      type(t1) :: x, y
    end subroutine
    subroutine s4(x, y)
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't3' from host is not accessible
      import, all
      type(t1) :: x
      type(t3) :: y
      integer :: t3
    end subroutine
  end interface
contains
  subroutine s5()
  end
  subroutine s6()
    import, only: s5
    implicit none(external)
    call s5()
  end
  subroutine s7()
    import, only: t1
    implicit none(external)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's5' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
    call s5()
  end
end module
