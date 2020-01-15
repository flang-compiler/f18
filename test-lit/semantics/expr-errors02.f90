! Test specification expressions

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  type :: t(n)
    integer, len :: n = 1
    character(len=n) :: c
  end type
  interface
    integer function foo()
    end function
    pure real function realfunc(x)
      real, intent(in) :: x
    end function
    pure integer function hasProcArg(p)
      import realfunc
      procedure(realfunc) :: p
    end function
  end interface
  integer :: coarray[*]
 contains
  pure integer function modulefunc1(n)
    integer, value :: n
    modulefunc1 = n
  end function
  subroutine test(out, optional)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: reference to impure function 'foo'
    type(t(foo())) :: x1
    integer :: local
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: reference to local entity 'local'
    type(t(local)) :: x2
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The internal function 'internal' cannot be referenced in a specification expression
    type(t(internal(0))) :: x3
    integer, intent(out) :: out
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: reference to INTENT(OUT) dummy argument 'out'
    type(t(out)) :: x4
    integer, intent(in), optional :: optional
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: reference to OPTIONAL dummy argument 'optional'
    type(t(optional)) :: x5
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: dummy procedure argument
    type(t(hasProcArg(realfunc))) :: x6
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: coindexed reference
    type(t(coarray[1])) :: x7
    type(t(kind(foo()))) :: x101 ! ok
    type(t(modulefunc1(0))) :: x102 ! ok
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The module function 'modulefunc2' must have been previously defined when referenced in a specification expression
    type(t(modulefunc2(0))) :: x103 ! ok
   contains
    pure integer function internal(n)
      integer, value :: n
      internal = n
    end function
  end subroutine
  pure integer function modulefunc2(n)
    integer, value :: n
    modulefunc2 = n
  end function
end module
