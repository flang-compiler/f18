! Check for semantic errors in NULLIFY statements

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x(:)

Integer :: pi
Procedure(Real) :: prp

Allocate(x(3))
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: component in NULLIFY statement must have the POINTER attribute
Nullify(x(2)%p)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in NULLIFY statement must have the POINTER attribute
Nullify(pi)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in NULLIFY statement must have the POINTER attribute
Nullify(prp)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in NULLIFY statement must be a variable or procedure pointer name
Nullify(maxvalue)

End Program
