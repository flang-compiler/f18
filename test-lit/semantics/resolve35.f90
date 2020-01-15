! Construct names

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  real :: foo
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'foo' is already declared in this scoping unit
  foo: block
  end block foo
end

subroutine s2(x)
  logical :: x
  foo: if (x) then
  end if foo
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'foo' is already declared in this scoping unit
  foo: do i = 1, 10
  end do foo
end

subroutine s3
  real :: a(10,10), b(10,10)
  type y; end type
  integer(8) :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Index name 'y' conflicts with existing identifier
  forall(x=1:10, y=1:10)
    a(x, y) = b(x, y)
  end forall
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Index name 'y' conflicts with existing identifier
  forall(x=1:10, y=1:10) a(x, y) = b(x, y)
end

subroutine s4
  real :: a(10), b(10)
  complex :: x
  integer :: i(2)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is COMPLEX(4)
  forall(x=1:10)
    !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is COMPLEX(4)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is COMPLEX(4)
    a(x) = b(x)
  end forall
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
  forall(y=1:10)
    !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
    a(y) = b(y)
  end forall
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Index variable 'i' is not scalar
  forall(i=1:10)
    a(i) = b(i)
  end forall
end

subroutine s5
  real :: a(10), b(10)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' is already declared in this scoping unit
  forall(i=1:10, i=1:10)
    a(i) = b(i)
  end forall
end

subroutine s6
  integer, parameter :: n = 4
  real, dimension(n) :: x
  data(x(i), i=1, n) / n * 0.0 /
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Index name 't' conflicts with existing identifier
  forall(t=1:n) x(t) = 0.0
contains
  subroutine t
  end
end

subroutine s6b
  integer, parameter :: k = 4
  integer :: l = 4
  forall(integer(k) :: i = 1:10)
  end forall
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
  forall(integer(l) :: i = 1:10)
  end forall
end

subroutine s7
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' is already declared in this scoping unit
  do concurrent(integer::i=1:5) local(j, i) &
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'j' is already declared in this scoping unit
      local_init(k, j) &
      shared(a)
    a = j + 1
  end do
end

subroutine s8
  implicit none
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'i'
  do concurrent(i=1:5) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'j'
    local(j) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'k'
    local_init(k)
  end do
end

subroutine s9
  integer :: j
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' is already declared in this scoping unit
  do concurrent(integer::i=1:5) shared(i) &
      shared(j) &
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'j' is already declared in this scoping unit
      shared(j)
  end do
end

subroutine s10
  external bad1
  real, parameter :: bad2 = 1.0
  x = cos(0.)
  do concurrent(i=1:2) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad1' may not appear in a locality-spec because it is not definable
    local(bad1) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad2' may not appear in a locality-spec because it is not definable
    local(bad2) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad3' may not appear in a locality-spec because it is not definable
    local(bad3) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'cos' may not appear in a locality-spec because it is not definable
    local(cos)
  end do
  do concurrent(i=1:2) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The name 'bad1' must be a variable to appear in a locality-spec
    shared(bad1) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The name 'bad2' must be a variable to appear in a locality-spec
    shared(bad2) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The name 'bad3' must be a variable to appear in a locality-spec
    shared(bad3) &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The name 'cos' must be a variable to appear in a locality-spec
    shared(cos)
  end do
contains
  subroutine bad3
  end
end
