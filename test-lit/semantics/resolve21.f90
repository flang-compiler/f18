! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  type :: t
    integer :: i
    integer :: s1
    integer :: t
  end type
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't' is already declared in this scoping unit
  integer :: t
  integer :: i, j
  type(t) :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 't2' not found
  type(t2) :: y
  external :: v
  type(t) :: v, w
  external :: w
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' is not an object of derived type; it is implicitly typed
  i = z%i
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's1' is an invalid base for a component reference
  i = s1%i
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'j' is not an object of derived type
  i = j%i
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Component 'j' not found in derived type 't'
  i = x%j
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'v' is an invalid base for a component reference
  i = v%i
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'w' is an invalid base for a component reference
  i = w%i
  i = x%i  !OK
end subroutine

subroutine s2
  type :: t1
    integer :: i
  end type
  type :: t2
    type(t1) :: x
  end type
  type(t2) :: y
  integer :: i
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Component 'j' not found in derived type 't1'
  k = y%x%j
  k = y%x%i !OK
end subroutine
