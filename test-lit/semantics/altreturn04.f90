! Functions cannot use alt return

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


REAL FUNCTION altreturn01(X)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
