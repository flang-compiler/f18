! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! Test 15.7 (C1583-C1590, C1592-C1599) constraints and restrictions
! for PURE procedures.
! (C1591 is tested in call11.f90; C1594 in call12.f90.)

module m

  type :: impureFinal
   contains
    final :: impure
  end type
  type :: t
  end type
  type :: polyAlloc
    class(t), allocatable :: a
  end type

  real, volatile, target :: volatile

 contains

  subroutine impure(x)
    type(impureFinal) :: x
  end subroutine
  integer impure function notpure(n)
    integer, value :: n
    notpure = n
  end function

  pure real function f01(a)
    real, intent(in) :: a ! ok
  end function
  pure real function f02(a)
    real, value :: a ! ok
  end function
  pure real function f03(a) ! C1583
    !ERROR: non-POINTER dummy argument of PURE function must be INTENT(IN) or VALUE
    real :: a
  end function
  pure real function f03a(a)
    real, pointer :: a ! ok
  end function
  pure real function f04(a) ! C1583
    !ERROR: non-POINTER dummy argument of PURE function must be INTENT(IN) or VALUE
    real, intent(out) :: a
  end function
  pure real function f04a(a)
    real, pointer, intent(out) :: a ! ok if pointer
  end function
  pure real function f05(a) ! C1583
    real, value :: a ! weird, but ok (VALUE without INTENT)
  end function
  pure function f06() ! C1584
    !ERROR: Result of PURE function may not have an impure FINAL subroutine
    type(impureFinal) :: f06
  end function
  pure function f07() ! C1585
    !ERROR: Result of PURE function may not be both polymorphic and ALLOCATABLE
    class(t), allocatable :: f07
  end function
  pure function f08() ! C1585
    !ERROR: Result of PURE function may not have polymorphic ALLOCATABLE ultimate component '%a'
    type(polyAlloc) :: f08
  end function

  pure subroutine s01(a) ! C1586
    !ERROR: non-POINTER dummy argument of PURE subroutine must have INTENT() or VALUE attribute
    real :: a
  end subroutine
  pure subroutine s01a(a)
    real, pointer :: a
  end subroutine
  pure subroutine s02(a) ! C1587
    !ERROR: An INTENT(OUT) dummy argument of a PURE subroutine may not have an impure FINAL subroutine
    type(impureFinal), intent(out) :: a
  end subroutine
  pure subroutine s03(a) ! C1588
    !ERROR: An INTENT(OUT) dummy argument of a PURE subroutine may not be polymorphic
    class(t), intent(out) :: a
  end subroutine
  pure subroutine s04(a) ! C1588
    !ERROR: An INTENT(OUT) dummy argument of a PURE subroutine may not have a polymorphic ultimate component
    type(polyAlloc), intent(out) :: a
  end subroutine
  pure subroutine s05 ! C1589
    !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
    real, save :: v1
    !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
    real :: v2 = 0.
    !TODO: once we have DATA: !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
    real :: v3
    data v3/0./
    !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
    real :: v4
    common /blk/ v4
    save /blk/
    block
    !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
      real, save :: v5
    !ERROR: A PURE subprogram may not have a variable with the SAVE attribute
      real :: v6 = 0.
    end block
  end subroutine
  pure subroutine s06 ! C1589
    !ERROR: A PURE subprogram may not have a variable with the VOLATILE attribute
    real, volatile :: v1
    block
    !ERROR: A PURE subprogram may not have a variable with the VOLATILE attribute
      real, volatile :: v2
    end block
  end subroutine
  !ERROR: A dummy procedure of a PURE subprogram must be PURE
  pure subroutine s07(p) ! C1590
    procedure(impure) :: p
  end subroutine
  ! C1591 is tested in call11.f90.
  pure subroutine s08 ! C1592
   contains
    pure subroutine pure ! ok
    end subroutine
    !ERROR: An internal subprogram of a PURE subprogram must also be PURE
    subroutine impure1
    end subroutine
    !ERROR: An internal subprogram of a PURE subprogram must also be PURE
    impure subroutine impure2
    end subroutine
  end subroutine
  pure subroutine s09 ! C1593
    real :: x
    !ERROR: VOLATILE variable 'volatile' may not be referenced in PURE subprogram 's09'
    x = volatile
  end subroutine
  ! C1594 is tested in call12.f90.
  pure subroutine s10 ! C1595
    integer :: n
    !ERROR: Procedure 'notpure' referenced in PURE subprogram 's10' must be PURE too
    n = notpure(1)
  end subroutine
  pure subroutine s11(to) ! C1596
    ! Implicit deallocation at the end of the subroutine
    !ERROR: Deallocation of polymorphic object 'auto%a' is not permitted in a PURE subprogram
    type(polyAlloc) :: auto
    type(polyAlloc), intent(in out) :: to
    !ERROR: Deallocation of polymorphic non-coarray component '%a' is not permitted in a PURE subprogram
    to = auto
  end subroutine
  pure subroutine s12
    character(20) :: buff
    real :: x
    write(buff, *) 1.0 ! ok
    read(buff, *) x ! ok
    !ERROR: External I/O is not allowed in a PURE subprogram
    print *, 'hi' ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    open(1, file='launch-codes') ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    close(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    backspace(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    endfile(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    rewind(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    flush(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    wait(1) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    inquire(1, name=buff) ! C1597
    !ERROR: External I/O is not allowed in a PURE subprogram
    read(5, *) x ! C1598
    !ERROR: External I/O is not allowed in a PURE subprogram
    read(*, *) x ! C1598
    !ERROR: External I/O is not allowed in a PURE subprogram
    write(6, *) ! C1598
    !ERROR: External I/O is not allowed in a PURE subprogram
    write(*, *) ! C1598
  end subroutine
  pure subroutine s13
    !ERROR: An image control statement may not appear in a PURE subprogram
    sync all ! C1599
    ! TODO others from 11.6.1 (many)
  end subroutine

end module
