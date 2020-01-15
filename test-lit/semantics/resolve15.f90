! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  real :: var
  interface i
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'var' is not a subprogram
    procedure :: sub, var
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'bad' not found
    procedure :: bad
  end interface
  interface operator(.foo.)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'var' is not a subprogram
    procedure :: sub, var
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'bad' not found
    procedure :: bad
  end interface
contains
  subroutine sub
  end
end

subroutine s
  interface i
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'sub' is not a module procedure
    module procedure :: sub
  end interface
  interface assignment(=)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'sub' is not a module procedure
    module procedure :: sub
  end interface
contains
  subroutine sub(x, y)
    real, intent(out) :: x
    logical, intent(in) :: y
  end
end
