! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


integer :: x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The type of 'x' has already been declared
real :: x
integer(8) :: i
parameter(i=1,j=2,k=3)
integer :: j
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The type of 'k' has already been implicitly declared
real :: k
end
