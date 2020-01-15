! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


program p
  integer :: p ! this is ok
end
module m
  integer :: m ! this is ok
end
submodule(m) sm
  integer :: sm ! this is ok
end
module m2
  type :: t
  end type
  interface
    subroutine s
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Module 'm2' cannot USE itself
      use m2, only: t
    end subroutine
  end interface
end module
subroutine s
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's' is already declared in this scoping unit
  integer :: s
end
function f() result(res)
  integer :: res
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'f' is already declared in this scoping unit
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The type of 'f' has already been declared
  real :: f
  res = 1
end
