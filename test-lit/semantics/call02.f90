! 15.5.1 procedure reference constraints and restrictions

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s01(elem, subr)
  interface
    elemental real function elem(x)
      real, intent(in), value :: x
    end function
    subroutine subr(dummy)
      procedure(sin) :: dummy
    end subroutine
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: A dummy procedure may not be ELEMENTAL
    subroutine badsubr(dummy)
      import :: elem
      procedure(elem) :: dummy
    end subroutine
  end interface
  call subr(cos) ! not an error
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-intrinsic ELEMENTAL procedure 'elem' may not be passed as an actual argument
  call subr(elem) ! C1533
end subroutine

module m01
  procedure(sin) :: elem01
  interface
    elemental real function elem02(x)
      real, value :: x
    end function
    subroutine callme(f)
      external f
    end subroutine
  end interface
 contains
  elemental real function elem03(x)
    real, value :: x
  end function
  subroutine test
    call callme(cos) ! not an error
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-intrinsic ELEMENTAL procedure 'elem01' may not be passed as an actual argument
    call callme(elem01) ! C1533
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-intrinsic ELEMENTAL procedure 'elem02' may not be passed as an actual argument
    call callme(elem02) ! C1533
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-intrinsic ELEMENTAL procedure 'elem03' may not be passed as an actual argument
    call callme(elem03) ! C1533
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-intrinsic ELEMENTAL procedure 'elem04' may not be passed as an actual argument
    call callme(elem04) ! C1533
   contains
    elemental real function elem04(x)
      real, value :: x
    end function
  end subroutine
end module

module m02
  type :: t
    integer, pointer :: ptr
  end type
  type(t) :: coarray[*]
 contains
  subroutine callee(x)
    type(t), intent(in) :: x
  end subroutine
  subroutine test
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Coindexed object 'coarray' with POINTER ultimate component '%ptr' cannot be associated with dummy argument 'x='
    call callee(coarray[1]) ! C1537
  end subroutine
end module
