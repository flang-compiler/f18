! Check for type errors in DEALLOCATE statements

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x

Real :: r
Integer :: s
Integer :: e
Integer :: pi
Character(256) :: ee
Procedure(Real) :: prp

Allocate(x)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have CHARACTER type, but is INTEGER(4)
Deallocate(x, stat=s, errmsg=e)

!ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have CHARACTER type, but is INTEGER(4)
Deallocate(x, stat=r, errmsg=e)

End Program
