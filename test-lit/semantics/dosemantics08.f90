! C1138 -- 
! A branch (11.2) within a DO CONCURRENT construct shall not have a branch
! target that is outside the construct.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1()
  do concurrent (i=1:10)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Control flow escapes from DO CONCURRENT
    goto 99
  end do

99 print *, "Hello"

end subroutine s1
