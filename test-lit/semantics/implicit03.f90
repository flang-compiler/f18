! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  implicit integer(a-z)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT NONE statement after IMPLICIT statement
  implicit none
end subroutine

subroutine s2
  implicit integer(a-z)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPLICIT NONE(TYPE) after IMPLICIT statement
  implicit none(type)
end subroutine
