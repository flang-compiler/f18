! Test SELECT CASE Constraints: C1145, C1146, C1147, C1148, C1149
program selectCaseProg
implicit none
   ! local variable declaration
   character :: grade1 = 'B'
   integer :: grade2 = 3
   logical :: grade3 = .false.
   character (len = 10) :: name = 'test'
   real :: grade4 = 2.0
   logical, parameter :: grade5 = .false.
   CHARACTER(KIND=1), PARAMETER    :: ze = 'a', at='b'
   CHARACTER(KIND=2), PARAMETER    :: se = 'c'
   CHARACTER(KIND=4), PARAMETER    :: mh='d'

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

   !ERROR: SELECT CASE expression must be of type CHARACTER, INTEGER, OR LOGICAL
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
      !ERROR: Not more than one of the selectors of SELECT CASE statement may be DEFAULT
      case default
         print*, "Invalid grade"
   end select

   select case (grade3)
      case default
         print*, "Invalid grade"
      case (.true.)
      !ERROR: CASE value must be of type LOGICAL
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
      !ERROR: CASE value must be of type INTEGER
      case (.true. :)
   end select

   select case (grade3)
      case (.true.)
       name = 'true'
      case (.false.)
        name = 'false'
      !ERROR: CASE value true matches a previous CASE statement
      case (.true.)
       name = 'true'
      !ERROR: CASE value false matches a previous CASE statement
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
      !ERROR: CASE value 81:90 matches a previous CASE statement
      case (81:90)
         print*, "Very good!"
      !ERROR: CASE value :80 matches a previous CASE statement
      !ERROR: CASE value :80 matches a previous CASE statement
      case (:80)
         print*, "Well done!"
      case default
         print*, "Invalid marks"
   end select

  select case (grade2)
     !ERROR: CASE value must be of type INTEGER
     case (:'Z')
         print*, "Excellent!"
     case default
         print*, "Invalid grade"
   end select

  select case (grade1)
     !ERROR: CASE value must be of type CHARACTER
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
     !ERROR: CASE value "hj" matches a previous CASE statement
     case ('hj')
         print*, "Well done now"
     case default
         print*, "Invalid grade"
   end select

  select case (ze)
     case (at)
         print*, "Excellent!"
     !ERROR: CASE value kind is 4, it must be 1
     case (mh)
         print*, "Well done now"
     !ERROR: CASE value kind is 2, it must be 1
     case (se)
         print*, "Good"
     case default
         print*, "Invalid grade"
   end select

end program selectCaseProg
