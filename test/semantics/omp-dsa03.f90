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

! OpenMP Data-Sharing Attributes examples (implicitly determined)

! In a parallel construct, if no default clause is present,
! these variables are shared.
subroutine dsa_parallel_default()
  integer :: c(10)
  c = 1
  !$omp parallel
  a = 1 ! a is shared
  b = 2 ! b is shared
  c = 3 ! c is shared
  !$omp end parallel
  print *,a,b,sum(c)
end subroutine dsa_parallel_default

subroutine dsa_default_private_helper(A, N)
  real A(1:N,*)
  b = 10
  !$omp parallel do default(private)
  do i = 1, N
     do j = 1, N
        A(j,i) = j ! assumed-size array A is still shared
        b = 99 ! b is private because of default(private)
     end do
  enddo
  print *,sum(A(:,1)), b
end subroutine dsa_default_private_helper

subroutine dsa_default_private()
  real A(10,10)
  A = 0.0
  call dsa_default_private_helper(A, 10)
end subroutine dsa_default_private

subroutine dsa_default_firstprivate()
  a = 1
  b = 2
  !$omp parallel do default(firstprivate) shared(c)
  do i = 1, 10
     c = a
     a = 3
  enddo
  print *,a.eq.1, c.eq.1
end subroutine dsa_default_firstprivate

subroutine dsa_orphaned_task_helper(x)
  real x
  real a
  a = 2.0
  !$omp task shared(a)
  a = x ! dummy argument x is firstprivate
  x = 2.0
  !$omp end task
  print *,a,x ! should all be 1.0
end subroutine dsa_orphaned_task_helper

! In an orphaned task generating construct, if no default clause
! is present, dummy arguments are firstprivate.
subroutine dsa_orphaned_task()
  x = 1.0
  call dsa_orphaned_task_helper(x)
end subroutine dsa_orphaned_task

program mm
  call dsa_parallel_default
  call dsa_default_private
  call dsa_default_firstprivate
  call dsa_orphaned_task
end program mm
