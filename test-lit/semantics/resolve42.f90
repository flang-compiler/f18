! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Array 'z' without ALLOCATABLE or POINTER attribute must have explicit shape
  common x, y(4), z(:)
end

subroutine s2
  common /c1/ x, y, z
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' is already in a COMMON block
  common y
end

subroutine s3
  procedure(real) :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'x' is already declared as a procedure
  common x
  common y
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' is already declared as an object
  procedure(real) :: y
end

subroutine s5
  integer x(2)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The dimensions of 'x' have already been declared
  common x(4), y(4)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The dimensions of 'y' have already been declared
  real y(2)
end

function f6(x) result(r)
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Dummy argument 'x' may not appear in a COMMON block
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ALLOCATABLE object 'y' may not appear in a COMMON block
  common x,y,z
  allocatable y
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Function result 'r' may not appear in a COMMON block
  common r
end

module m7
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Variable 'w' with BIND attribute may not appear in a COMMON block
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Variable 'z' with BIND attribute may not appear in a COMMON block
  common w,z
  integer, bind(c) :: z
  integer, bind(c,name="w") :: w
end

module m8
  type t
  end type
  class(*), pointer :: x
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unlimited polymorphic pointer 'x' may not appear in a COMMON block
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unlimited polymorphic pointer 'y' may not appear in a COMMON block
  common x, y
  class(*), pointer :: y
end

module m9
  integer x
end
subroutine s9
  use m9
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'x' is use-associated from module 'm9' and cannot be re-declared
  common x
end

module m10
  type t
  end type
  type(t) :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type 'x' in COMMON block must have the BIND or SEQUENCE attribute
  common x
end

module m11
  type t1
    sequence
    integer, allocatable :: a
  end type
  type t2
    sequence
    type(t1) :: b
    integer:: c
  end type
  type(t2) :: x2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type variable 'x2' may not appear in a COMMON block due to ALLOCATABLE component
  common x2
end

module m12
  type t1
    sequence
    integer :: a = 123
  end type
  type t2
    sequence
    type(t1) :: b
    integer:: c
  end type
  type(t2) :: x2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Derived type variable 'x2' may not appear in a COMMON block due to component with default initialization
  common x2
end

subroutine s13
  block
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMMON statement is not allowed in a BLOCK construct
    common x
  end block
end

subroutine s14
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'c' appears as a COMMON block in a BIND statement but not in a COMMON statement
  bind(c) :: /c/
end
