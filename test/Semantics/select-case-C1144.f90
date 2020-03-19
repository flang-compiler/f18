! RUN: %S/test_errors.sh %s %flang %t
! Test SELECT CASE Constraint: C1144
program selectCaseProg
implicit none
   ! local variable declaration
   character :: grade = 'B'
  ! Valid Case
  ! ***************************************************
  case1: select case (grade)
      case ('A') 
      print *, "Excellent!" 
      case ('B')
      case ('C') 
         print *, "Well done" 
      case default
         print *, "Invalid grade" 
   end select case1
  ! ***************************************************

  ! C1144
  ! ***************************************************
    case1: select case (grade)
      case ('A') 
      print*, "Excellent!" 
      case ('B')
      case ('C') 
         print*, "Well done" 
      case default
         print*, "Invalid grade" 
    !ERROR: SELECT CASE construct name mismatch
    end select case2
  
    case2: select case (grade)
      case ('A')
      print*, "Excellent!"
      case ('B')
      case ('C')
         print*, "Well done"
      case default
         print*, "Invalid grade"
    !ERROR: SELECT CASE construct name required but missing
    end select

    select case (grade)
      case ('A')
      print*, "Excellent!"
      case ('B')
      case ('C')
         print*, "Well done"
      case default
         print*, "Invalid grade"
    !ERROR: SELECT CASE construct name unexpected
    end select case2
  ! ***************************************************

end program selectCaseProg
