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

subroutine s1
  !ERROR: The shape of common block object 'z' must be explicit
  common x, y(4), z(:)
end

subroutine s2
  common /c1/ x, y, z
  !ERROR: 'y' is already in a COMMON block
  common y
end

subroutine s3
  procedure(real) :: x
  !ERROR: 'x' is already declared as a procedure
  common x
  common y
  !ERROR: 'y' is already declared as an object
  procedure(real) :: y
end

subroutine s5
  integer x(2)
  !ERROR: The dimensions of 'x' have already been declared
  common x(4), y(4)
  !ERROR: The dimensions of 'y' have already been declared
  real y(2)
end

function f6(x) result(r)
  !ERROR: Dummy argument 'x' may not appear in a COMMON block
  !ERROR: ALLOCATABLE object 'y' may not appear in a COMMON block
  common x,y,z
  allocatable y
  !ERROR: Function result 'r' may not appear in a COMMON block
  common r
end

module m7
  !ERROR: Variable 'w' with BIND attribute may not appear in a COMMON block
  !ERROR: Variable 'z' with BIND attribute may not appear in a COMMON block
  common w,z
  integer, bind(c) :: z
  integer, bind(c,name="w") :: w
end

module m8
  type t
  end type
  class(*), pointer :: x
  !ERROR: Unlimited polymorphic pointer 'x' may not appear in a COMMON block
  !ERROR: Unlimited polymorphic pointer 'y' may not appear in a COMMON block
  common x, y
  class(*), pointer :: y
end

module m9
  integer x
end
subroutine s9
  use m9
  !ERROR: 'x' is use-associated from module 'm9' and cannot be re-declared
  common x
end

module m10
  type t
  end type
  type(t) :: x
  !ERROR: Derived type 'x' in COMMON block must have the BIND or SEQUENCE attribute
  common x
end

module m11
  type t1
    sequence
    integer, allocatable :: a
  end type
  type t2
    sequence
    type(t1) :: b
    integer:: c
  end type
  type(t2) :: x2
  !ERROR: Derived type variable 'x2' may not appear in a COMMON block due to ALLOCATABLE component
  common x2
end

module m12
  type t1
    sequence
    integer :: a = 123
  end type
  type t2
    sequence
    type(t1) :: b
    integer:: c
  end type
  type(t2) :: x2
  !ERROR: Derived type variable 'x2' may not appear in a COMMON block due to component with default initialization
  common x2
end

subroutine s13
  block
    !ERROR: COMMON statement is not allowed in a BLOCK construct
    common x
  end block
end

subroutine s14
  !ERROR: 'c' appears as a COMMON block in a BIND statement but not in a COMMON statement
  bind(c) :: /c/
end
