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

!OPTIONS: -fopenmp

! OpenMP Data-Mapping Attributes examples

module decl_tgt_tolink
  integer, parameter :: N = 128
  !$omp declare target to(A, B)
  real :: A(N), B(N)

contains
  subroutine addup()
    !$omp declare target

    !$omp parallel do
    do i=1, N
       A(i) = B(i) + 1.0
    enddo
  end subroutine addup
end module decl_tgt_tolink

! PREDETERMINED
! If a variable appears in a to or link clause on a declare target
! directive then it is treated as if it had appeared in a map clause
! with a map-type of tofrom.
subroutine decl_tgt_tolink_helper()
  use decl_tgt_tolink
  B = 1.0
  !! implicit map(tofrom:A,B)
  !$omp target
  call addup()
  !$omp end target
  print *,sum(A)  ! expecting N*2.
end subroutine decl_tgt_tolink_helper

! IMPLICIT
! 1) If a defaultmap(tofrom:scalar) clause is not present then a scalar
!    variable is not mapped, but instead has an implicit data-sharing
!    attribute of firstprivate
! 2) If a variable is not a scalar then it is treated as if it had appeared
!    in a map clause with a map-type of tofrom.
subroutine defaultmap_not_present_scalar
  real :: A(16)
  i = 1
  !$omp target
  A = i + 1  ! i follows rule 1); A follows rule 2)
  !$omp end target
  print *, sum(A)  ! value of A is copied out
end subroutine defaultmap_not_present_scalar

! IMPLICIT
! If a defaultmap(tofrom:scalar) clause is present then a scalar variable
! is treated as if it had appeared in a map clause with a map-type of tofrom.
subroutine defaultmap_present_scalar
  i = 1
  !$omp target defaultmap(tofrom:scalar)
  i = 2
  !$omp end target
  print *, i.eq.2
end subroutine defaultmap_present_scalar

program tt
  call decl_tgt_tolink_helper
  call defaultmap_not_present_scalar
  call defaultmap_present_scalar
end program tt
