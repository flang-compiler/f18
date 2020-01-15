! Test extension: RETURN from main program

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR %s


return !ok
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
return 0
end
