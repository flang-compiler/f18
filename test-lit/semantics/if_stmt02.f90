! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IF statement is not allowed in IF statement
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
