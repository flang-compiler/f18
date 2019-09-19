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

! OpenMP Data-Sharing Attributes examples (PREDETERMINED)

! Variables and common blocks appearing in threadprivate directives
! are threadprivate.
subroutine dsa_threadprivate()
  use omp_lib
  common /c/ a, b
  integer :: a
  real :: b
  !$omp threadprivate(/c/)

  !$omp parallel
  a = omp_get_thread_num()
  b = a + 1.
  !$omp end parallel

  print *, 'master thread: ', a, b  ! master threadprivate

  !$omp parallel
  print *, 'thread ', omp_get_thread_num(), ': ', a, b
  !$omp end parallel
end subroutine dsa_threadprivate

! In a parallel construct, if no default clause is present,
! these variables (not predetermined or explicit determined) are shared.
subroutine dsa_bad_sum()
  integer :: a = 0
  !$omp parallel do
  do i = 1, 10 ! i is private
     do j = 1, 10 ! j is shared
        a = a + j ! a is shared
     enddo
  enddo
  print *, a.eq.550
  ! "private(j) reduction(+:a)" on the PARALLEL DO directive will make it right
end subroutine dsa_bad_sum

! The loop iteration variable(s) in the associated do-loop(s) of a do,
! parallel do, taskloop, or distribute construct is (are) private.
subroutine dsa_loop_iv()
  i = 0
  !$omp parallel do
  do i = 1, 10 ! loop induction variable is private
  enddo
  print *,i.eq.0
end subroutine dsa_loop_iv

! The loop iteration variable in the associated do-loop of a simd construct
! with just one associated do-loop is linear with a linear-step that is the
! increment of the associated do-loop.
! The loop iteration variables in the associated do-loops of a simd construct
! with multiple associated do-loops are lastprivate.
subroutine dsa_simd_iv()
  i = 1
  !$omp simd
  do i = 1, 5 ! i is linear
  end do
  print *,i

  !$omp simd
  do i = 1, 10 ! i is lastprivate
     do j = 1, 20 ! j is lastprivate
     end do
  end do
  print *,i,j
  ! however, result varies from vendor to vendor
end subroutine dsa_simd_iv

! Implied-do indices and forall indices are private.
subroutine dsa_forall()
  REAL, DIMENSION(10, 10) :: A
  !$omp parallel
  !$omp single
  FORALL (i=1:10) A(i, i) = 1 ! forall index I is private
  !$omp end single
  !$omp end parallel
  print *,sum(a)
end subroutine dsa_forall

! Cray pointees have the same the data-sharing attribute as the storage
! with which their Cray pointers are associated.
subroutine dsa_cray()
  integer result(5)
  double precision dum(2)
  double precision :: d
  pointer(ptr, d)

  !$omp parallel shared(ptr)
  ptr = loc(dum) ! pointee d should have the same DSA with associated pointer
  dum(1) = 2.0
  !$omp end parallel
  print *, d
end subroutine dsa_cray

subroutine dsa_assumed_size_helper(A, N)
  real A(1:N,*)
  !$omp parallel do
  do i = 1, N
     do j = 1, N
        A(j,i) = j ! assumed-size array is shared
     end do
  enddo
  print *,sum(A(:,1))
end subroutine dsa_assumed_size_helper

! Assumed-size arrays are shared.
subroutine dsa_assumed_size()
  real A(10,10)
  A = 0.0
  call dsa_assumed_size_helper(A, 10)
end subroutine dsa_assumed_size

! An associate name preserves the association with the selector established
! at the ASSOCIATE statement.
subroutine dsa_associate()
  use omp_lib
  !$omp parallel private(i)
  i = omp_get_thread_num()
  associate(id => i) ! id should also be private, otherwise error
    print *, id ! print private i value
  end associate
  !$omp end parallel
end subroutine dsa_associate

! In a task generating construct, if no default clause is present, a variable
! for which the data-sharing attribute is not determined by the rules above
! is firstprivate.
subroutine dsa_explicit_task()
  a = 1
  !$omp task shared(a)
  !$omp task
  a = a + 1 ! a should be firstprivate
  !$omp end task
  !$omp end task
  print *,a.eq.1.0
end subroutine dsa_explicit_task

program mm
  call dsa_threadprivate
  call dsa_bad_sum
  call dsa_loop_iv
  call dsa_simd_iv
  call dsa_forall
  call dsa_cray
  call dsa_assumed_size
  call dsa_associate
  call dsa_explicit_task
end program mm
