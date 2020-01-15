! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  interface a
    subroutine s(x)
      real :: x
    end subroutine
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's' is already declared in this scoping unit
    subroutine s(x)
      integer :: x
    end subroutine
  end interface
end module

module m2
  interface s
    subroutine s(x)
      real :: x
    end subroutine
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 's' is already declared in this scoping unit
    subroutine s(x)
      integer :: x
    end subroutine
  end interface
end module
