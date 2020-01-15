! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  integer x
  block
    import, none
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'x' from host scoping unit is not accessible due to IMPORT
    x = 1
  end block
end

subroutine s2
  block
    import, none
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' from host scoping unit is not accessible due to IMPORT
    y = 1
  end block
end

subroutine s3
  implicit none
  integer :: i, j
  block
    import, none
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'i'
    real :: a(16) = [(i, i=1, 16)]
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'j'
    data(a(j), j=1, 16) / 16 * 0.0 /
  end block
end

subroutine s4
  real :: i, j
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
  real :: a(16) = [(i, i=1, 16)]
  data(
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
    a(j), &
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must have INTEGER type, but is REAL(4)
    j=1, 16 &
  ) / 16 * 0.0 /
end
