!C1119

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine test1(a, i)
  integer i
  real a(10)
  critical
    if (a(i) < 0.0) then
      a(i) = 20.20
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Control flow escapes from CRITICAL
      goto 20
    end if
  end critical
20 a(i) = -a(i)
end subroutine test1

subroutine test2(i)
  integer i
  critical
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Control flow escapes from CRITICAL
    if (i) 10, 10, 20
    10 i = i + 1
  end critical
20 i = i - 1
end subroutine test2

subroutine test3(i)
  integer i
  critical
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Control flow escapes from CRITICAL
    goto (10, 10, 20) i
    10 i = i + 1
  end critical
20 i = i - 1
end subroutine test3
