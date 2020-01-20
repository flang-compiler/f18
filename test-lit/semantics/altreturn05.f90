! Test extension: RETURN from main program

!RUN: %test_error %s %flang

return !ok
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
return 0
end
