!C1132
! If the do-stmt is a nonlabel-do-stmt, the corresponding end-do shall be an
! end-do-stmt.
! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1()
  do while (.true.)
    print *, "Hello"
  continue
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: expected 'END DO'
end subroutine s1
