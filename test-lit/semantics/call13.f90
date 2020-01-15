! Test 15.4.2.2 constraints and restrictions for calls to implicit
! interfaces

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s(assumedRank, coarray, class, classStar, typeStar)
  type :: t
  end type

  real :: assumedRank(..), coarray[*]
  class(t) :: class
  class(*) :: classStar
  type(*) :: typeStar

  type :: pdt(len)
    integer, len :: len
  end type
  type(pdt(1)) :: pdtx

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid specification expression: reference to impure function 'implicit01'
  real :: array(implicit01())  ! 15.4.2.2(2)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Keyword 'keyword=' may not appear in a reference to a procedure with an implicit interface
  call implicit10(1, 2, keyword=3)  ! 15.4.2.2(1)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed rank argument requires an explicit interface
  call implicit11(assumedRank)  ! 15.4.2.2(3)(c)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Coarray argument requires an explicit interface
  call implicit12(coarray)  ! 15.4.2.2(3)(d)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Parameterized derived type argument requires an explicit interface
  call implicit13(pdtx)  ! 15.4.2.2(3)(e)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Polymorphic argument requires an explicit interface
  call implicit14(class)  ! 15.4.2.2(3)(f)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Polymorphic argument requires an explicit interface
  call implicit15(classStar)  ! 15.4.2.2(3)(f)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed type argument requires an explicit interface
  call implicit16(typeStar)  ! 15.4.2.2(3)(f)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: TYPE(*) dummy argument may only be used as an actual argument
  if (typeStar) then
  endif
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: TYPE(*) dummy argument may only be used as an actual argument
  classStar = typeStar  ! C710
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: TYPE(*) dummy argument may only be used as an actual argument
  typeStar = classStar  ! C710
end subroutine

