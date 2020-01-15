! Simple check that if statements are ok.

! RUN: %flang -fdebug-resolve-names -fparse-only %s 2>&1


IF (A > 0.0) A = LOG (A)
END
