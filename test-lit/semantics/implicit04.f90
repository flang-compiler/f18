! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s
  parameter(a=1.0)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT NONE statement after PARAMETER statement
  implicit none
end subroutine
