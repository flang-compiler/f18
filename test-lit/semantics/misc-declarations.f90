! Miscellaneous constraint and requirement checking on declarations:
! - 8.5.6.2 & 8.5.6.3 constraints on coarrays
! - 8.5.19 constraints on the VOLATILE attribute

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ALLOCATABLE coarray must have a deferred coshape
  real, allocatable :: mustBeDeferred[*]  ! C827
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Non-ALLOCATABLE coarray must have an explicit coshape
  real :: mustBeExplicit[:]  ! C828
  type :: hasCoarray
    real :: coarray[*]
  end type
  real :: coarray[*]
  type(hasCoarray) :: coarrayComponent
 contains
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may not apply to an INTENT(IN) argument
  subroutine C866(x)
    intent(in) :: x
    volatile :: x
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may apply only to a variable
    volatile :: notData
    external :: notData
  end subroutine
  subroutine C867
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may not apply to a coarray accessed by USE or host association
    volatile :: coarray
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
    volatile :: coarrayComponent
  end subroutine
  subroutine C868(coarray,coarrayComponent)
    real, volatile :: coarray[*]
    type(hasCoarray) :: coarrayComponent
    block
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may not apply to a coarray accessed by USE or host association
      volatile :: coarray
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
      volatile :: coarrayComponent
    end block
  end subroutine
end module
