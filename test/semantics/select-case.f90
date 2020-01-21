!===--- select-case.f90 - Test select case constraints -------------------===
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===

! C1145 - case-expr shall be of type character, integer, or logical
program selectCaseProg
implicit none
   ! local variable declaration
   character :: grade1 = 'B'
   integer :: grade2 = 3
   logical :: grade3 = .false.
   character (len = 10) :: name = 'test'
   real :: grade4 = 2.0
   logical, parameter :: grade5 = .false.

   select case (grade1)
      case ('A')
      print*, "Excellent!"
      case ('B')
      case ('C')
         print*, "Well done"
      case default
         print*, "Invalid grade"
   end select

   select case (grade2)
      case (1)
      print*, "Excellent!"
      case (2)
      case (3)
         print*, "Well done"
      case default
         print*, "Invalid grade"
   end select

   select case (grade3)
      case (.true.)
       name = 'true'
      case (.false.)
        name = 'false'
   end select
   print*, "Your grade is ", name

   select case (name)
      case default
         print*, "Invalid grade"
      case ('now')
      case ('test')
         print*, "Well done"
   end select

   !ERROR: SELECT CASE expression must be of type CHARACTER, INTERGER, OR LOGICAL
   select case (grade4)
      case (1.0)
      print*, "Excellent!"
      case (2.0)
      case (3.0)
         print*, "Well done"
      case default
         print*, "Invalid grade"
   end select

   select case (grade3)
      case default
         print*, "Invalid grade"
      case (.true.)
         print*, "Well done"
      !ERROR: Not more than one of the selectors of case statements may be default
      case default
         print*, "Invalid grade"
   end select

   select case (grade3)
      case default
         print*, "Invalid grade"
      case (.true.)
      !ERROR: SELECT CASE value must be of type LOGICAL
      case (3)
         print*, "Well done"
   end select

   select case (grade3)
      case default
         print*, "Invalid grade"
      !ERROR: SELECT CASE expression of type LOGICAL must not have range of case value
      case (.true. :)
   end select

   select case (grade2)
      case default
         print*, "Invalid grade"
      case (2 :)
         print*, "Well done"
      !ERROR: SELECT CASE value must be of type INTEGER
      case (.true. :)
   end select

   select case (grade3)
      case (.true.)
       name = 'true'
      case (.false.)
        name = 'false'
      !ERROR: SELECT CASE statement value must not match more than one case-value-range
      case (.true.)
       name = 'true'
      !ERROR: SELECT CASE statement value must not match more than one case-value-range
      case (grade5)
       name = 'true'
   end select
   print*, "Your grade is ", name

   select case (grade2)
      case (100:)
         print*, "Excellent!"
      case (:30)
         print*, "Excellent!"
      case (40)
         print*, "Excellent!"
      case (90)
         print*, "Excellent!"
      case (91:99)
         print*, "Very good!"
      !ERROR: SELECT CASE statement value must not match more than one case-value-range
      case (81:90)
         print*, "Very good!"
      !ERROR: SELECT CASE statement value must not match more than one case-value-range
      case (:80)
         print*, "Well done!"
      case default
         print*, "Invalid marks"
   end select

  select case (grade2)
     !ERROR: SELECT CASE value must be of type INTEGER
     case (:'Z')
         print*, "Excellent!"
     case default
         print*, "Invalid grade"
   end select

  select case (grade1)
     !ERROR: SELECT CASE value must be of type CHARACTER
     case (:1)
         print*, "Excellent!"
     case default
         print*, "Invalid grade"
   end select

  select case (name)
     case ('hello')
         print*, "Excellent!"
     case ('hey')
         print*, "Good"
     case ('hi':'ho')
         print*, "Well done"
     !ERROR: SELECT CASE statement value must not match more than one case-value-range
     case ('hj')
         print*, "Well done now"
     case default
         print*, "Invalid grade"
   end select

end program selectCaseProg
