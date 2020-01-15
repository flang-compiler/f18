! Check that if constructs only accept scalar logical expressions.
! TODO: expand the test to check this restriction for more types.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


INTEGER :: I
LOGICAL, DIMENSION (2) :: B

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 1
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 2
else
  a = 3
endif

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 4
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
else if( B ) then
  a = 5
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 6
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
else if( B ) then
  a = 7
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
elseif( B ) then
  a = 8
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 9
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
else if( B ) then
  a = 10
else
  a = 11
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
if ( B ) then
  a = 12
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
else if( B ) then
  a = 13
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a scalar value, but is a rank-1 array
else if( B ) then
  a = 14
end if


!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 1
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 2
else
  a = 3
endif

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 4
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
else if( I ) then
  a = 5
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 6
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
else if( I ) then
  a = 7
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
elseif( I ) then
  a = 8
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 9
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
else if( I ) then
  a = 10
else
  a = 11
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
if ( I ) then
  a = 12
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
else if( I ) then
  a = 13
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is INTEGER(4)
else if( I ) then
  a = 14
end if

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have LOGICAL type, but is REAL(4)
if (f()) then
  a = 15
end if

contains
  real function f()
    f = 1.0
  end
end
