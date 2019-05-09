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

! The parse of this source produces messages that make it difficult 
! to figure out what the problem is.
!
! This program compiles and executes without error on GNU Fortran
!
! RUN: ${F18} -fparse-only %s 2>&1 | ${FileCheck} %s

PROGRAM parsequirk01
  IMPLICIT NONE
  INTEGER :: n
  INTEGER, DIMENSION(5) :: table
  n = 1

  DO CONCURRENT (table(n) = 1:n)
    PRINT *, "Hello"
  END DO
END PROGRAM parsequirk01
