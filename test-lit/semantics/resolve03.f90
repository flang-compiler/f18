!RUN: %test_error %s %flang

implicit none
integer :: x
!ERROR: No explicit type declared for 'y'
y = x
end
