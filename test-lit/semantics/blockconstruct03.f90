! Tests implemented for this standard:
!            Block Construct
! C1109

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s5_c1109
  b1:block
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: BLOCK construct name mismatch
  end block b2
end

