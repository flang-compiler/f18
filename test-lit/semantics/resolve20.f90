! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  abstract interface
    subroutine foo
    end subroutine
  end interface

  procedure() :: a
  procedure(integer) :: b
  procedure(foo) :: c
  procedure(bar) :: d
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'missing' must be an abstract interface or a procedure with an explicit interface
  procedure(missing) :: e
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'b' must be an abstract interface or a procedure with an explicit interface
  procedure(b) :: f
  procedure(c) :: g
  external :: h
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'h' must be an abstract interface or a procedure with an explicit interface
  procedure(h) :: i
  procedure(forward) :: j
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad1' must be an abstract interface or a procedure with an explicit interface
  procedure(bad1) :: k1
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad2' must be an abstract interface or a procedure with an explicit interface
  procedure(bad2) :: k2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bad3' must be an abstract interface or a procedure with an explicit interface
  procedure(bad3) :: k3

  abstract interface
    subroutine forward
    end subroutine
  end interface

  real :: bad1(1)
  real :: bad2
  type :: bad3
  end type

  type :: m ! the name of a module can be used as a local identifier
  end type m

  external :: a, b, c, d
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXTERNAL attribute not allowed on 'm'
  external :: m
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXTERNAL attribute not allowed on 'foo'
  external :: foo
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXTERNAL attribute not allowed on 'bar'
  external :: bar

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: PARAMETER attribute not allowed on 'm'
  parameter(m=2)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: PARAMETER attribute not allowed on 'foo'
  parameter(foo=2)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: PARAMETER attribute not allowed on 'bar'
  parameter(bar=2)

  type, abstract :: t1
    integer :: i
  contains
    !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'proc' must be an abstract interface or a procedure with an explicit interface
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure component 'p1' has invalid interface 'proc'
    procedure(proc), deferred :: p1
  end type t1

contains
  subroutine bar
  end subroutine
end module
