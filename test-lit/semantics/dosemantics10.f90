! C1134 A CYCLE statement must be within a DO construct
!
! C1166 An EXIT statement must be within a DO construct

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1()
! this one's OK
  do i = 1,10
    cycle
  end do

! this one's OK
  do i = 1,10
    exit
  end do

! all of these are OK
  outer: do i = 1,10
    cycle
    inner: do j = 1,10
      cycle
    end do inner
    cycle
  end do outer

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No matching DO construct for CYCLE statement
  cycle

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No matching construct for EXIT statement
  exit

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No matching DO construct for CYCLE statement
  if(.true.) cycle

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No matching construct for EXIT statement
  if(.true.) exit

end subroutine s1
