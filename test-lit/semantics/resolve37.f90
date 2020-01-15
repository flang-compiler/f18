! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


integer, parameter :: k = 8
real, parameter :: l = 8.0
integer :: n = 2
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
parameter(m=n)
integer(k) :: x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
integer(l) :: y
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
integer(n) :: z
type t(k)
  integer, kind :: k
end type
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Type parameter 'k' lacks a value and has no default
type(t( &
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is LOGICAL(4)
  .true.)) :: w
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
real :: u(l*2)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
character(len=l) :: v
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Initialization expression for PARAMETER 'o' (o) cannot be computed as a constant value
real, parameter ::  o = o
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
integer, parameter ::  p = 0/0
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
integer, parameter ::  q = 1+2*(1/0)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
integer(kind=2/0) r
integer, parameter :: sok(*)=[1,2]/[1,2]
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
integer, parameter :: snok(*)=[1,2]/[1,0]
end
