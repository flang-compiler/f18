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

   !ERROR: SELECT CASE expression must be of type character, integer, or logical
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
      !ERROR: SELECT CASE value type must be same as SELECT CASE expression type
      case (3) 
         print*, "Well done" 
   end select

   select case (grade3)
      case default
         print*, "Invalid grade"
      !ERROR: SELECT CASE expression of type logical must not have case value range using colon
      case (.true. :)
   end select

   select case (grade2)
      case default
         print*, "Invalid grade" 
      case (2 :)
         print*, "Well done"
      !ERROR: SELECT CASE value type must be same as SELECT CASE expression type
      case (.true. :)
   end select

end program selectCaseProg
