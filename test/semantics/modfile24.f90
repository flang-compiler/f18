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

! Test declarations with coarray-spec

! Different ways of declaring the same coarray.
module m1
  real :: a(1:5)[1:10,1:*]
  real, dimension(5) :: b[1:10,1:*]
  real, codimension[1:10,1:*] :: c(5)
  real, codimension[1:10,1:*], dimension(5) :: d
  codimension :: e[1:10,1:*]
  dimension :: e(5)
  real :: e
end
!Expect: m1.mod
!module m1
! real(4)::a(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::b(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::c(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::d(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::e(1_8:5_8)[1_8:10_8,1_8:*]
!end

! coarray-spec in codimension and target statements.
module m2
  codimension :: a[10,*], b[*]
  target :: c[10,*], d[*]
end
!Expect: m2.mod
!module m2
! real(4)::a[1_8:10_8,1_8:*]
! real(4)::b[1_8:*]
! real(4),target::c[1_8:10_8,1_8:*]
! real(4),target::d[1_8:*]
!end

! coarray-spec in components and with non-constants bounds
module m3
  type t
    real :: c(1:5)[1:10,1:*]
    complex, codimension[5,*] :: d
  end type
  real, allocatable :: e[:,:,:]
contains
  subroutine s(a, b, n)
    integer(8) :: n
    real :: a[1:n,2:*]
    real, codimension[1:n,2:*] :: b
  end
end
!Expect: m3.mod
!module m3
! type::t
!  real(4)::c(1_8:5_8)[1_8:10_8,1_8:*]
!  complex(4)::d[1_8:5_8,1_8:*]
! end type
! real(4),allocatable::e[:,:,:]
!contains
! subroutine s(a,b,n)
!  integer(8)::n
!  real(4)::a[1_8:n,2_8:*]
!  real(4)::b[1_8:n,2_8:*]
! end
!end

! coarray-spec in both attributes and entity-decl
module m4
  real, codimension[2:*], dimension(2:5) :: a, b(4,4), c[10,*], d(4,4)[10,*]
end
!Expect: m4.mod
!module m4
! real(4)::a(2_8:5_8)[2_8:*]
! real(4)::b(1_8:4_8,1_8:4_8)[2_8:*]
! real(4)::c(2_8:5_8)[1_8:10_8,1_8:*]
! real(4)::d(1_8:4_8,1_8:4_8)[1_8:10_8,1_8:*]
!end
