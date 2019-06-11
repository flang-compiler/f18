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

! RUN: ${F18} -fopenmp -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! OPTIONS: -fopenmp

! Check OpenMP clause validity for the following directives:
!
!    PARALLEL
!    ...

  N = 1024
  !$omp parallel
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: 'SCHEDULE(STATIC)' not allowed in OMP PARALLEL
  !$omp parallel schedule(static)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: 'COLLAPSE(2)' not allowed in OMP PARALLEL
  !$omp parallel collapse(2)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  a = 1.0
  !$omp parallel firstprivate(a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: 'LASTPRIVATE(A)' not allowed in OMP PARALLEL
  !ERROR: 'NUM_TASKS(4)' not allowed in OMP PARALLEL
  !ERROR: 'INBRANCH' not allowed in OMP PARALLEL
  !$omp parallel lastprivate(a) NUM_TASKS(4) inbranch
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel
end
