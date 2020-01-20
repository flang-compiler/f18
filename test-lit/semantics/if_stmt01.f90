! Simple check that if statements are ok.

!RUN: %test_error %s %flang

IF (A > 0.0) A = LOG (A)
END
