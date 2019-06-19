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

! Check OpenMP 2.17 Nesting of Regions

  N = 1024
  !$omp do
  do i = 1, N
     !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region.
     !$omp do
     do i = 1, N
        a = 3.14
     enddo
  enddo
end
