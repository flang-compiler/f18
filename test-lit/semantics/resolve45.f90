! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


function f1(x, y)
  integer x
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to dummy argument 'x'
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to dummy argument 'y'
  save x,y
  integer y
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to function result 'f1'
  save f1
end

function f2(x, y)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to function result 'f2'
  real, save :: f2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to dummy argument 'x'
  complex, save :: x
  allocatable :: y
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to dummy argument 'y'
  integer, save :: y
end

subroutine s3(x)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute may not be applied to dummy argument 'x'
  procedure(integer), pointer, save :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Procedure 'y' with SAVE attribute must also have POINTER attribute
  procedure(integer), save :: y
end

subroutine s4
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Explicit SAVE of 'z' is redundant due to global SAVE statement
  save z
  save
  procedure(integer), pointer :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Explicit SAVE of 'x' is redundant due to global SAVE statement
  save :: x
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Explicit SAVE of 'y' is redundant due to global SAVE statement
  integer, save :: y
end

subroutine s5
  implicit none
  integer x
  block
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'x'
    save x
  end block
end

subroutine s6
  save x
  save y
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute was already specified on 'y'
  integer, save :: y
  integer, save :: z
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: SAVE attribute was already specified on 'x'
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE attribute was already specified on 'z'
  save x,z
end

subroutine s7
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'x' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /x/
end
