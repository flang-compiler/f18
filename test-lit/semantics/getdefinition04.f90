! Tests -fget-definition with COMMON block with same name as variable.
!RUN: %test_generic %s %flang
program main
  integer :: x
  integer :: y
  common /x/ y
  x = y
end program

! EXEC: ${F18} -fget-definition 7 3 4 -fparse-only %s | ${FileCheck} %s
! CHECK:x:.*getdefinition04.f90, 4, 14-15
