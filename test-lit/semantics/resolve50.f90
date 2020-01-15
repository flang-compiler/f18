! Test coarray association in CHANGE TEAM statement

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  use iso_fortran_env
  type(team_type) :: t
  complex :: x[*]
  real :: y[*]
  real :: z
  ! OK
  change team(t, x[*] => y)
  end team
  ! C1116
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Selector in coarray association must name a coarray
  change team(t, x[*] => 1)
  end team
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Selector in coarray association must name a coarray
  change team(t, x[*] => z)
  end team
end

subroutine s2
  use iso_fortran_env
  type(team_type) :: t
  real :: y[10,*], y2[*], x[*]
  ! C1113
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The codimensions of 'x' have already been declared
  change team(t, x[10,*] => y, x[*] => y2)
  end team
end
