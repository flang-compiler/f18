! Confirm enforcement of constraints and restrictions in 15.6.2.1

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


non_recursive function f01(n) result(res)
  integer, value :: n
  integer :: res
  if (n <= 0) then
    res = n
  else
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: NON_RECURSIVE procedure 'f01' cannot call itself
    res = n * f01(n-1) ! 15.6.2.1(3)
  end if
end function

non_recursive function f02(n) result(res)
  integer, value :: n
  integer :: res
  if (n <= 0) then
    res = n
  else
    res = nested()
  end if
 contains
  integer function nested
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: NON_RECURSIVE procedure 'f02' cannot call itself
    nested = n * f02(n-1) ! 15.6.2.1(3)
  end function nested
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be RECURSIVE
recursive character(*) function f03(n) ! C723
  integer, value :: n
  f03 = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be RECURSIVE
recursive function f04(n) result(res) ! C723
  integer, value :: n
  character(*) :: res
  res = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot return an array
character(*) function f05()
  dimension :: f05(1) ! C723
  f05(1) = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot return an array
function f06()
  character(*) :: f06(1) ! C723
  f06(1) = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot return a POINTER
character(*) function f07()
  pointer :: f07 ! C723
  character, target :: a = ' '
  f07 => a
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot return a POINTER
function f08()
  character(*), pointer :: f08 ! C723
  character, target :: a = ' '
  f08 => a
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be PURE
pure character(*) function f09() ! C723
  f09 = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be PURE
pure function f10()
  character(*) :: f10 ! C723
  f10 = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be ELEMENTAL
elemental character(*) function f11(n) ! C723
  integer, value :: n
  f11 = ''
end function

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: An assumed-length CHARACTER(*) function cannot be ELEMENTAL
elemental function f12(n)
  character(*) :: f12 ! C723
  integer, value :: n
  f12 = ''
end function

function f13(n) result(res)
  integer, value :: n
  character(*) :: res
  if (n <= 0) then
    res = ''
  else
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-length CHARACTER(*) function 'f13' cannot call itself
    res = f13(n-1) ! 15.6.2.1(3)
  end if
end function

function f14(n) result(res)
  integer, value :: n
  character(*) :: res
  if (n <= 0) then
    res = ''
  else
    res = nested()
  end if
 contains
  character(1) function nested
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-length CHARACTER(*) function 'f14' cannot call itself
    nested = f14(n-1) ! 15.6.2.1(3)
  end function nested
end function
