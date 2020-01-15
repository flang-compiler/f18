! Check that computed goto express must be a scalar integer expression
! TODO: PGI, for example, accepts a float & converts the value to int.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


REAL R
COMPLEX Z
LOGICAL L
INTEGER, DIMENSION (2) :: B

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
GOTO (100) 1.5
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is LOGICAL(4)
GOTO (100) .TRUE.
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
GOTO (100) R
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is COMPLEX(4)
GOTO (100) Z
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
GOTO (100) B

100 CONTINUE

END
