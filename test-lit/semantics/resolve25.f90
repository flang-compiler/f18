! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  interface foo
    subroutine s1(x)
      real x
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's2' is not a module procedure
    module procedure s2
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 's3' not found
    procedure s3
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 's1' is already specified in generic 'foo'
    procedure s1
  end interface
  interface
    subroutine s4(x,y)
      real x,y
    end subroutine
    subroutine s2(x,y)
      complex x,y
    end subroutine
  end interface
  generic :: bar => s4
  generic :: bar => s2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 's4' is already specified in generic 'bar'
  generic :: bar => s4

  generic :: operator(.foo.)=> s4
  generic :: operator(.foo.)=> s2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 's4' is already specified in generic operator '.foo.'
  generic :: operator(.foo.)=> s4
end module

module m2
  interface
    integer function f(x, y)
      logical, intent(in) :: x, y
    end function
  end interface
  generic :: operator(+)=> f
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'f' is already specified in generic 'operator(+)'
  generic :: operator(+)=> f
end

module m3
  interface operator(.ge.)
    procedure f
  end interface
  interface operator(>=)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'f' is already specified in generic 'operator(.ge.)'
    procedure f
  end interface
  generic :: operator(>) => f
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'f' is already specified in generic 'operator(>)'
  generic :: operator(.gt.) => f
contains
  logical function f(x, y) result(result)
    logical, intent(in) :: x, y
    result = .true.
  end
end
