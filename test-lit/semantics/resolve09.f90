! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


integer :: y
procedure() :: a
procedure(real) :: b
call a  ! OK - can be function or subroutine
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 'a' like a function
c = a()
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'b' like a subroutine
call b
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'y' like a subroutine
call y
call x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 'x' like a function
z = x()
end

subroutine s
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'f' like a subroutine
  call f
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 's' like a function
  i = s()
contains
  function f()
  end
end

subroutine s2
  ! subroutine vs. function is determined by use
  external :: a, b
  call a()
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 'a' like a function
  x = a()
  x = b()
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'b' like a subroutine
  call b()
end

subroutine s3
  ! subroutine vs. function is determined by use, even in internal subprograms
  external :: a
  procedure() :: b
contains
  subroutine s3a()
    x = a()
    call b()
  end
  subroutine s3b()
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'a' like a subroutine
    call a()
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 'b' like a function
    x = b()
  end
end

module m
  ! subroutine vs. function is determined at end of specification part
  external :: a
  procedure() :: b
contains
  subroutine s()
    call a()
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call subroutine 'b' like a function
    x = b()
  end
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s4
  block
    import, none
    integer :: i
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Use of 'm' as a procedure conflicts with its declaration
    i = m()
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Use of 'm' as a procedure conflicts with its declaration
    call m()
  end block
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s5
  block
    import, none
    integer :: i
    i = foo()
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Cannot call function 'foo' like a subroutine
    call foo()
  end block
end

subroutine s6
  call a6()
end
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'a6' was previously called as a subroutine
function a6()
  a6 = 0.0
end

subroutine s7
  x = a7()
end
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'a7' was previously called as a function
subroutine a7()
end

!OK: use of a8 and b8 is consistent
subroutine s8
  call a8()
  x = b8()
end
subroutine a8()
end
function b8()
  b8 = 0.0
end
