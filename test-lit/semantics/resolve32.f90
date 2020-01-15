! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m2
  public s2, s4
  private s3
contains
  subroutine s2
  end
  subroutine s3
  end
  subroutine s4
  end
end module

module m
  use m2
  external bar
  interface
    subroutine foo
    end subroutine
  end interface
  integer :: i
  type t1
    integer :: c
  contains
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The binding of 'a' ('missing') must be either an accessible module procedure or an external procedure with an explicit interface
    procedure, nopass :: a => missing
    procedure, nopass :: b => s, s2
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Type parameter, component, or procedure binding 'c' already defined in this type
    procedure, nopass :: c
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DEFERRED is only allowed when an interface-name is provided
    procedure, nopass, deferred :: d => s
    !Note: s3 not found because it's not accessible -- should we issue a message
    !to that effect?
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's3' must be either an accessible module procedure or an external procedure with an explicit interface
    procedure, nopass :: s3
    procedure, nopass :: foo
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bar' must be either an accessible module procedure or an external procedure with an explicit interface
    procedure, nopass :: bar
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' must be either an accessible module procedure or an external procedure with an explicit interface
    procedure, nopass :: i
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Type parameter, component, or procedure binding 'b' already defined in this type
    procedure, nopass :: b => s4
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DEFERRED is required when an interface-name is provided
    procedure(foo), nopass :: g
  end type
  type, abstract :: t1a ! DEFERRED valid only in ABSTRACT derived type
  contains
    procedure(foo), nopass, deferred :: e
    procedure(s), nopass, deferred :: f
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Type parameter, component, or procedure binding 'f' already defined in this type
    procedure(foo), nopass, deferred :: f
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'bar' must be an abstract interface or a procedure with an explicit interface
    procedure(bar), nopass, deferred :: h
  end type
  type t2
    integer :: i
  contains
    procedure, nopass :: b => s
    final :: f
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Type parameter, component, or procedure binding 'i' already defined in this type
    final :: i
  end type
  type t3
  contains
    private
    procedure, nopass :: b => s
    procedure, nopass, public :: f
  end type
contains
  subroutine s
  end
  subroutine f(x)
    type(t2) :: x
  end
end module
