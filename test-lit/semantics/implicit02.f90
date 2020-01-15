! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  implicit none
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT statement after IMPLICIT NONE or IMPLICIT NONE(TYPE) statement
  implicit integer(a-z)
end subroutine

subroutine s2
  implicit none(type)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT statement after IMPLICIT NONE or IMPLICIT NONE(TYPE) statement
  implicit integer(a-z)
end subroutine
