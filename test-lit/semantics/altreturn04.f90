! Functions cannot use alt return

!RUN: %test_error %s %flang

REAL FUNCTION altreturn01(X)
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
