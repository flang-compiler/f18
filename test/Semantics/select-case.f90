! Test SELECT CASE Constraints: C1145, C1146, C1147, C1148, C1149
program selectCaseProg
implicit none
   ! local variable declaration
   character :: grade1 = 'B'
   integer :: grade2 = 3
   logical :: grade3 = .false.
   real :: grade4 = 2.0
   character (len = 10) :: name = 'test'
   logical, parameter :: grade5 = .false.
   CHARACTER(KIND=1), PARAMETER    :: UTF8_var1 = 'a', UTF8_var2='b'
   CHARACTER(KIND=2), PARAMETER    :: UTF16_var = 'c'
   CHARACTER(KIND=4), PARAMETER    :: UTF32_var ='d'
   type scores
     integer :: val
   end type
   type (scores) :: score = scores(25)
   type (scores), parameter :: score_val = scores(50)

  ! Valid Cases
  ! ***************************************************
   select case (grade1)
      case ('A')
      print *, "Excellent!"
      case ('B')
      case ('C')
         print *, "Well done"
      case default
         print *, "Invalid grade"
   end select

   select case (grade2)
      case (1)
      print *, "Excellent!"
      case (2)
      case (3)
         print *, "Well done"
      case default
         print *, "Invalid grade"
   end select

   select case (grade3)
      case (.true.)
       name = 'true'
      case (.false.)
        name = 'false'
   end select
   print *, "Your grade is ", name

   select case (name)
      case default
         print *, "Invalid grade"
      case ('now')
      case ('test')
         print *, "Well done"
   end select
  ! ***************************************************

  ! C1145
  ! ***************************************************
  !ERROR: SELECT CASE expression must be of type CHARACTER, INTEGER, OR LOGICAL
  select case (grade4)
     case (1.0)
     print *, "Excellent!"
     case (2.0)
     case (3.0)
        print *, "Well done"
     case default
        print *, "Invalid grade"
  end select

  !ERROR: SELECT CASE expression must be of type CHARACTER, INTEGER, OR LOGICAL
  select case (score)
     case (score_val)
       print *, "Half century"
     case (scores(100))
       print *, "Century"
  end select
  ! ***************************************************

  ! C1146
  ! ***************************************************
  select case (grade3)
     case default
        print *, "Invalid grade"
     case (.true.)
        print *, "Well done"
     !ERROR: Not more than one of the selectors of SELECT CASE statement may be DEFAULT
     case default
        print *, "Invalid grade"
  end select
  ! ***************************************************

  ! C1147
  ! ***************************************************
  select case (grade2)
     !ERROR: CASE value must be of type INTEGER
     case (:'Z')
         print *, "Excellent!"
     case default
         print *, "Invalid grade"
   end select

  select case (grade1)
     !ERROR: CASE value must be of type CHARACTER
     case (:1)
         print *, "Excellent!"
     case default
         print *, "Invalid grade"
   end select

  select case (grade3)
     case default
        print *, "Invalid grade"
     case (.true.)
     !ERROR: CASE value must be of type LOGICAL
     case (3)
        print *, "Well done"
  end select

  select case (grade2)
     case default
        print *, "Invalid grade"
     case (2 :)
        print *, "Well done"
     !ERROR: CASE value must be of type INTEGER
     case (.true. :)
  end select

  select case (UTF8_var1)
     case (UTF8_var2)
         print *, "Excellent!"
     !ERROR: Character kind type of case construct (=1) mismatches with the kind type of case value (=4)
     case (UTF32_var)
         print *, "Well done now"
     !ERROR: Character kind type of case construct (=1) mismatches with the kind type of case value (=2)
     case (UTF16_var)
         print *, "Good"
     case default
         print *, "Invalid grade"
   end select
  ! ***************************************************

  ! C1148
  ! ***************************************************
  select case (grade3)
     case default
        print *, "Invalid grade"
     !ERROR: SELECT CASE expression of type LOGICAL must not have range of case value
     case (.true. :)
  end select
  ! ***************************************************

  ! C1149
  ! ***************************************************
  select case (grade3)
    case (.true.)
      name = 'true'
    case (.false.)
       name = 'false'
     !ERROR: CASE value .TRUE. matches a previous CASE statement
     case (.true.)
      name = 'true'
     !ERROR: CASE value .FALSE. matches a previous CASE statement
     case (grade5)
      name = 'true'
  end select
  print *, "Your grade is ", name

  select case (grade2)
     case (100:)
        print *, "Excellent!"
     case (:30)
        print *, "Excellent!"
     case (40)
        print *, "Excellent!"
     case (90)
        print *, "Excellent!"
     case (91:99)
        print *, "Very good!"
     !ERROR: CASE value 81:90 matches a previous CASE statement
     case (81:90)
        print *, "Very good!"
     !ERROR: CASE value :80 matches a previous CASE statement
     !ERROR: CASE value :80 matches a previous CASE statement
     case (:80)
        print *, "Well done!"
     case default
        print *, "Invalid marks"
  end select

  select case (name)
     case ('hello')
         print *, "Excellent!"
     case ('hey')
         print *, "Good"
     case ('hi':'ho')
         print *, "Well done"
     !ERROR: CASE value "hj" matches a previous CASE statement
     case ('hj')
         print *, "Well done now"
     case default
         print *, "Invalid grade"
   end select
  ! ***************************************************

end program selectCaseProg
