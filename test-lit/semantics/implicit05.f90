! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'a' does not follow 'b' alphabetically
  implicit integer(b-a)
end
