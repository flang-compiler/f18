! C1003 - can't parenthesize function call returning procedure pointer
! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  type :: dt
    procedure(frpp), pointer, nopass :: pp
  end type dt
 contains
  subroutine boring
  end subroutine boring
  function frpp
    procedure(boring), pointer :: frpp
    frpp => boring
  end function frpp
  subroutine tests
    procedure(boring), pointer :: mypp
    type(dt) :: dtinst
    mypp => boring ! legal
    mypp => (boring) ! legal, not a function reference
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: A function reference that returns a procedure pointer may not be parenthesized
    mypp => (frpp()) ! C1003
    mypp => frpp() ! legal, not parenthesized
    dtinst%pp => frpp
    mypp => dtinst%pp() ! legal
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: A function reference that returns a procedure pointer may not be parenthesized
    mypp => (dtinst%pp())
  end subroutine tests
end module m1
