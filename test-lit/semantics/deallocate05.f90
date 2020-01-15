! Check for semantic errors in DEALLOCATE statements

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


Module share
  Real, Pointer :: rp
End Module share

Program deallocatetest
Use share

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x(:)

Real :: r
Integer :: s
Integer :: e
Integer :: pi
Character(256) :: ee
Procedure(Real) :: prp

Allocate(rp)
Deallocate(rp)

Allocate(x(3))

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x(2)%p)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(pi)

!ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x(2)%p, pi)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must be a variable name
Deallocate(prp)

!ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must be a variable name
Deallocate(pi, prp)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: name in DEALLOCATE statement must be a variable name
Deallocate(maxvalue)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x%p)

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: STAT may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, stat=s)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ERRMSG may not be duplicated in a DEALLOCATE statement
Deallocate(x, errmsg=ee, errmsg=ee)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: STAT may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, errmsg=ee, stat=s)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ERRMSG may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, errmsg=ee, errmsg=ee)

End Program deallocatetest
