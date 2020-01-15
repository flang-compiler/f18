! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


integer :: g(10)
f(i) = i + 1  ! statement function
g(i) = i + 2  ! mis-parsed array assignment
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'h' has not been declared as an array
h(i) = i + 3
end
