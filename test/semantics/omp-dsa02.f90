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

! OpenMP Data-Sharing Attributes examples (expecting errors)

! The loop iteration variable(s) in the associated do-loop(s) of a do,
! parallel do, taskloop, or distribute construct may be listed in a
! private or lastprivate clause.
subroutine dsa_iv()
  !WRONG: IV(s) may be listed on PRIVATE or LASTPRIVATE clauses
  !$omp parallel do firstprivate(i)
  do i = 1, 10
  enddo
end subroutine dsa_iv

subroutine dsa_default_none()
  a = 1
  !$omp parallel do default(none) shared(c)
  do i = 1, 10 ! i is predetermined as private
     !WRONG: with default(none), a must appear in a DSA clause
     c = a
  enddo
end subroutine dsa_default_none

! A threadprivate variable must not appear in any clause except
! the copyin, copyprivate, schedule, num_threads, thread_limit,
! and if clauses.
subroutine threadprivate_illegal_clause()
  integer,save :: a
  !$omp threadprivate(a)

  !WRONG: a is threadprivate and can only appear on certain clauses
  !$omp parallel shared(a)
  !$omp end parallel
end subroutine threadprivate_illegal_clause

! A variable that is part of another variable (as an array or structure
! element) cannot appear in a threadprivate clause.
subroutine threadprivate_element()
  type helper
     integer :: a
  end type helper
  type(helper) hp
  !WRONG: no array or structure element
  !$omp threadprivate(hp%a)
end subroutine threadprivate_element

! If a threadprivate directive specifying a common block name
! appears in one program unit, then such a directive must also
! appear in every other program unit that contains a COMMON statement
! specifying the same name. It must appear after the last such COMMON
! statement in the program unit.
subroutine threadprivate_common()
  common /c/a,b
  integer :: a, b
  !$omp threadprivate(/c/)
contains
  subroutine helper()
    !WRONG: /c/ is not visible for threadprivate here
    !$omp parallel copyin(/c/)
    !$omp end parallel
  end subroutine helper
end subroutine threadprivate_common

! TODO
! If a threadprivate variable or a threadprivate common block is
! declared with the BIND attribute, the corresponding C entities
! must also be specified in a threadprivate directive in the C program.

! A variable that appears in a threadprivate directive must be
! declared in the scope of a module or have the SAVE attribute,
! either explicitly or implicitly.
subroutine threadprivate_no_save()
  !WRONG: need to save it!
  !$omp threadprivate(a)
end subroutine threadprivate_no_save

! A variable can only appear in a threadprivate directive in the
! scope in which it is declared. It must not be an element of a
! common block or appear in an EQUIVALENCE statement.
subroutine threadprivate_wrong_scope1()
  common /c/a,b
  integer :: a,b
  !$omp threadprivate(a)
end subroutine threadprivate_wrong_scope1

subroutine threadprivate_wrong_scope2()
  integer,save :: a
  real,save :: b
  EQUIVALENCE(a,b)
  !$omp threadprivate(a)
end subroutine threadprivate_wrong_scope2

program mm
  call dsa_iv
  call dsa_default_none
  call threadprivate_illegal_clause
  call threadprivate_element
  call threadprivate_common
  call threadprivate_no_save
  call threadprivate_wrong_scope1
  call threadprivate_wrong_scope2
end program mm
