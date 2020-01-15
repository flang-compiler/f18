! Check that non-logical expressions are not allowed.
! Check that non-scalar expressions are not allowed.
! TODO: Insure all non-logicals are prohibited.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


LOGICAL, DIMENSION (2) :: B

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is REAL(4)
IF (A) A = LOG (A)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
IF (B) A = LOG (A)

END
