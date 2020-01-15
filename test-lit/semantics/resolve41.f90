! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  implicit none
  real, parameter :: a = 8.0
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
  integer :: aa = 2_a
  integer :: b = 8
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
  integer :: bb = 2_b
  !TODO: should get error -- not scalar
  !integer, parameter :: c(10) = 8
  !integer :: cc = 2_c
  integer, parameter :: d = 47
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: INTEGER(KIND=47) is not a supported type
  integer :: dd = 2_d
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Parameter 'e' not found
  integer :: ee = 2_e
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Missing initialization for parameter 'f'
  integer, parameter :: f
  integer :: ff = 2_f
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=23) is not a supported type
  real(d/2) :: g
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL*47 is not a supported type
  real*47 :: h
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX*47 is not a supported type
  complex*47 :: i
end
