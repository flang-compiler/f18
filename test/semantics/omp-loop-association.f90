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

! OPTIONS: -fopenmp

! Check the association between OpenMPLoopConstruct and DoConstruct

  integer :: b = 128
  integer :: c = 32
  integer, parameter :: num = 16
  N = 1024

! Labeled DO loops are accepted

  !$omp parallel
  !$omp do
  do 10 i=1, N
     a = 3.14
10   print *, a
  !$omp end parallel

  !$omp parallel do
  DO CONCURRENT (i = 1:N)
     a = 3.14
  END DO

! Loop association check

  !$omp parallel do copyin(a)
  do i = 1, N
     !$omp parallel do
     do j = 1, i
     enddo
     !$omp end parallel do
     a = 3.
  enddo
  !$omp end parallel do

  !$omp parallel do
  do i = 1, N
  enddo
  !$omp end parallel do
  !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
  !$omp end parallel do

  !$omp parallel
  a = 3.0
  !$omp do simd
  do i = 1, N
  enddo
  !$omp end do simd

  !$omp parallel do copyin(a)
  do i = 1, N
  enddo
  !$omp end parallel

  a = 0.0
  !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
  !$omp end parallel do
  !$omp parallel do private(c)
  do i = 1, N
     do j = 1, N
        !ERROR: DO loop is expected after the PARALLEL DO directive
        !$omp parallel do shared(b)
        a = 3.14
     enddo
     !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
     !$omp end parallel do
  enddo
  a = 1.414
  !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
  !$omp end parallel do

  do i = 1, N
     !$omp parallel do
     do j = 2*i*N, (2*i+1)*N
        a = 3.14
     enddo
  enddo
  !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
  !$omp end parallel do

  !ERROR: DO loop is expected after the PARALLEL DO directive
  !$omp parallel do private(c)
5 FORMAT (1PE12.4, I10)
  do i=1, N
     a = 3.14
  enddo
  !ERROR: The END PARALLEL DO must follow the DO loop associated with the loop construct
  !$omp end parallel do

  !$omp parallel do simd
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel do simd
  !ERROR: The END PARALLEL DO SIMD must follow the DO loop associated with the loop construct
  !$omp end parallel do simd
end
