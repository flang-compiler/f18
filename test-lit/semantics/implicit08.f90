! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  block
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT statement is not allowed in a BLOCK construct
    implicit logical(a)
  end block
end subroutine
