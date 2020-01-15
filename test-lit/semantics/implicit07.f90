! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


implicit none(external)
external x
call x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'y' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
call y
block
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'z' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  call z
end block
end
