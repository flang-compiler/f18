! C1131, C1133 -- check valid and invalid DO loop naming
! C1131 (R1119) If the do-stmt of a do-construct specifies a do-construct-name,
! the corresponding end-do shall be an end-do-stmt specifying the same
! do-construct-name. If the do-stmt of a do-construct does not specify a
! do-construct-name, the corresponding end-do shall not specify a
! do-construct-name.
!
! C1133 (R1119) If the do-stmt is a label-do-stmt, the corresponding end-do
! shall be identified with the same label.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1()
  implicit none
  ! Valid construct
  validdo: do while (.true.)
      print *, "hello"
      cycle validdo
      print *, "Weird to get here"
    end do validdo

  validdo: do while (.true.)
      print *, "Hello"
    end do validdo

  ! Missing name on initial DO
  do while (.true.)
      print *, "Hello"
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DO construct name unexpected
    end do formerlabelmissing

  dolabel: do while (.true.)
      print *, "Hello"
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: DO construct name mismatch
    end do differentlabel

  dowithcycle: do while (.true.)
      print *, "Hello"
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: CYCLE construct-name is not in scope
      cycle validdo
    end do dowithcycle

end subroutine s1
