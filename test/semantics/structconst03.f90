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

! Error tests for structure constructors: C1594 violations
! from assigning globally-visible data to POINTER components.
! test/semantics/structconst04.f90 is this same test without type
! parameters.

module usefrom
  real, target :: usedfrom1
end module usefrom

module module1
  use usefrom
  implicit none
  type :: has_pointer1
    real, pointer :: ptop
    type(has_pointer1), allocatable :: link1 ! don't loop during analysis
  end type has_pointer1
  type :: has_pointer2
    type(has_pointer1) :: pnested
    type(has_pointer2), allocatable :: link2
  end type has_pointer2
  type, extends(has_pointer2) :: has_pointer3
    type(has_pointer3), allocatable :: link3
  end type has_pointer3
  type :: t1(k)
    integer, kind :: k
    real, pointer :: pt1
    type(t1(k)), allocatable :: link
  end type t1
  type :: t2(k)
    integer, kind :: k
    type(has_pointer1) :: hp1
    type(t2(k)), allocatable :: link
  end type t2
  type :: t3(k)
    integer, kind :: k
    type(has_pointer2) :: hp2
    type(t3(k)), allocatable :: link
  end type t3
  type :: t4(k)
    integer, kind :: k
    type(has_pointer3) :: hp3
    type(t4(k)), allocatable :: link
  end type t4
  real, target :: modulevar1
  type(has_pointer1) :: modulevar2
  type(has_pointer2) :: modulevar3
  type(has_pointer3) :: modulevar4

 contains

  pure real function pf1(dummy1, dummy2, dummy3, dummy4)
    real, target :: local1
    type(t1(0)) :: x1
    type(t2(0)) :: x2
    type(t3(0)) :: x3
    type(t4(0)) :: x4
    real, intent(in), target :: dummy1
    real, intent(inout), target :: dummy2
    real, pointer :: dummy3
    real, intent(inout), target :: dummy4[*]
    real, target :: commonvar1
    common /cblock/ commonvar1
    pf1 = 0.
    x1 = t1(0)(local1)
    !ERROR: Externally visible object 'usedfrom1' must not be associated with pointer component 'pt1' in a PURE function
    x1 = t1(0)(usedfrom1)
    !ERROR: Externally visible object 'modulevar1' must not be associated with pointer component 'pt1' in a PURE function
    x1 = t1(0)(modulevar1)
    !ERROR: Externally visible object 'cblock' must not be associated with pointer component 'pt1' in a PURE function
    x1 = t1(0)(commonvar1)
    !ERROR: Externally visible object 'dummy1' must not be associated with pointer component 'pt1' in a PURE function
    x1 = t1(0)(dummy1)
    x1 = t1(0)(dummy2)
    !ERROR: Externally visible object 'dummy3' must not be associated with pointer component 'pt1' in a PURE function
    x1 = t1(0)(dummy3)
! TODO when semantics handles coindexing:
! TODO !ERROR: Externally visible object must not be associated with a pointer in a PURE function
! TODO x1 = t1(0)(dummy4[0])
    x1 = t1(0)(dummy4)
    !ERROR: Externally visible object 'modulevar2' must not be associated with pointer component 'ptop' in a PURE function
    x2 = t2(0)(modulevar2)
    !ERROR: Externally visible object 'modulevar3' must not be associated with pointer component 'ptop' in a PURE function
    x3 = t3(0)(modulevar3)
    !ERROR: Externally visible object 'modulevar4' must not be associated with pointer component 'ptop' in a PURE function
    x4 = t4(0)(modulevar4)
   contains
    subroutine subr(dummy1a, dummy2a, dummy3a, dummy4a)
      real, target :: local1a
      type(t1(0)) :: x1a
      type(t2(0)) :: x2a
      type(t3(0)) :: x3a
      type(t4(0)) :: x4a
      real, intent(in), target :: dummy1a
      real, intent(inout), target :: dummy2a
      real, pointer :: dummy3a
      real, intent(inout), target :: dummy4a[*]
      x1a = t1(0)(local1a)
      !ERROR: Externally visible object 'usedfrom1' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(usedfrom1)
      !ERROR: Externally visible object 'modulevar1' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(modulevar1)
      !ERROR: Externally visible object 'cblock' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(commonvar1)
      !ERROR: Externally visible object 'dummy1' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(dummy1)
      !ERROR: Externally visible object 'dummy1a' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(dummy1a)
      x1a = t1(0)(dummy2a)
      !ERROR: Externally visible object 'dummy3' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(dummy3)
      !ERROR: Externally visible object 'dummy3a' must not be associated with pointer component 'pt1' in a PURE function
      x1a = t1(0)(dummy3a)
! TODO when semantics handles coindexing:
! TODO !ERROR: Externally visible object must not be associated with a pointer in a PURE function
! TODO x1a = t1(0)(dummy4a[0])
      x1a = t1(0)(dummy4a)
      !ERROR: Externally visible object 'modulevar2' must not be associated with pointer component 'ptop' in a PURE function
      x2a = t2(0)(modulevar2)
      !ERROR: Externally visible object 'modulevar3' must not be associated with pointer component 'ptop' in a PURE function
      x3a = t3(0)(modulevar3)
      !ERROR: Externally visible object 'modulevar4' must not be associated with pointer component 'ptop' in a PURE function
      x4a = t4(0)(modulevar4)
    end subroutine subr
  end function pf1

  impure real function ipf1(dummy1, dummy2, dummy3, dummy4)
    real, target :: local1
    type(t1(0)) :: x1
    type(t2(0)) :: x2
    type(t3(0)) :: x3
    type(t4(0)) :: x4
    real, intent(in), target :: dummy1
    real, intent(inout), target :: dummy2
    real, pointer :: dummy3
    real, intent(inout), target :: dummy4[*]
    real, target :: commonvar1
    common /cblock/ commonvar1
    ipf1 = 0.
    x1 = t1(0)(local1)
    x1 = t1(0)(usedfrom1)
    x1 = t1(0)(modulevar1)
    x1 = t1(0)(commonvar1)
    x1 = t1(0)(dummy1)
    x1 = t1(0)(dummy2)
    x1 = t1(0)(dummy3)
! TODO when semantics handles coindexing:
! TODO x1 = t1(0)(dummy4[0])
    x1 = t1(0)(dummy4)
    x2 = t2(0)(modulevar2)
    x3 = t3(0)(modulevar3)
    x4 = t4(0)(modulevar4)
  end function ipf1
end module module1
