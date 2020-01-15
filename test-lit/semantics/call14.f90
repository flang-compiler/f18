! Test 8.5.18 constraints on the VALUE attribute

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  type :: hasCoarray
    real :: coarray[*]
  end type
 contains
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may apply only to a dummy data object
  subroutine C863(notData,assumedSize,coarray,coarrayComponent)
    external :: notData
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may apply only to a dummy argument
    real, value :: notADummy
    value :: notData
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to an assumed-size array
    real, value :: assumedSize(10,*)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to a coarray
    real, value :: coarray[*]
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to a type with a coarray ultimate component
    type(hasCoarray), value :: coarrayComponent
  end subroutine
  subroutine C864(allocatable, inout, out, pointer, volatile)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to an ALLOCATABLE
    real, value, allocatable :: allocatable
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to an INTENT(IN OUT) argument
    real, value, intent(in out) :: inout
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to an INTENT(OUT) argument
    real, value, intent(out) :: out
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to a POINTER
    real, value, pointer :: pointer
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to a VOLATILE
    real, value, volatile :: volatile
  end subroutine
  subroutine C865(optional) bind(c)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: VALUE attribute may not apply to an OPTIONAL in a BIND(C) procedure
    real, value, optional :: optional
  end subroutine
end module
