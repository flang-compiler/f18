! Make sure arithmetic if expressions are non-complex numeric exprs.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


INTEGER I
COMPLEX Z
LOGICAL L
INTEGER, DIMENSION (2) :: B

if ( I ) 100, 200, 300
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ARITHMETIC IF expression must not be a COMPLEX expression
if ( Z ) 101, 201, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ARITHMETIC IF expression must be a numeric expression
if ( L ) 102, 202, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ARITHMETIC IF expression must be a scalar expression
if ( B ) 103, 203, 303
103 CONTINUE
203 CONTINUE
303 CONTINUE

END
