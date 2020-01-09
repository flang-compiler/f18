! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  implicit none
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: More than one IMPLICIT NONE statement
  implicit none(type)
end subroutine

subroutine s2
  implicit none(external)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: More than one IMPLICIT NONE statement
  implicit none
end subroutine
