! C1131 -- check valid and invalid DO loop naming

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


PROGRAM C1131
  IMPLICIT NONE
  ! Valid construct
  validDo: DO WHILE (.true.)
      PRINT *, "Hello"
    END DO ValidDo

  ! Missing name on END DO
  missingEndDo: DO WHILE (.true.)
      PRINT *, "Hello"
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DO construct name required but missing
    END DO

  ! Missing name on DO
  DO WHILE (.true.)
      PRINT *, "Hello"
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DO construct name unexpected
    END DO missingDO

END PROGRAM C1131
