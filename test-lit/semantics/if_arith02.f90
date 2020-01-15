! Check that only labels are allowed in arithmetic if statements.
! TODO: Revisit error message "expected 'ASSIGN'" etc.
! TODO: Revisit error message "expected one of '0123456789'"

! TODO: BUG: Note that labels 500 and 600 do not exist and
! ought to be flagged as errors. This oversight may be the
! result of disabling semantic checking after syntax errors.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


if ( A ) 500, 600, 600
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: [[@LINE+5]]:{{[0-9]+}}:{{.*}}error: expected 'ASSIGN'
!ERROR: [[@LINE+4]]:{{[0-9]+}}:{{.*}}error: expected 'ALLOCATE ('
!ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: expected '=>'
!ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: expected '('
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: expected '='
if ( B ) A, 101, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: expected one of '0123456789'
if ( B ) 102, A, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: expected one of '0123456789'
if ( B ) 103, 103, A
103 CONTINUE
203 CONTINUE
303 CONTINUE

END
