! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


implicit none
integer :: x
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: No explicit type declared for 'y'
y = x
end
