!C1117

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine test1(a, i)
  integer i
  real a(10)
  one: critical
    if (a(i) < 0.0) then
      a(i) = 20.20
    end if
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: CRITICAL construct name mismatch
  end critical two
end subroutine test1

subroutine test2(a, i)
  integer i
  real a(10)
  critical
    if (a(i) < 0.0) then
      a(i) = 20.20
    end if
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: CRITICAL construct name unexpected
  end critical two
end subroutine test2
