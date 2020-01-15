! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  implicit integer(a-c)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: More than one implicit type specified for 'c'
  implicit real(c-g)
end

subroutine s2
  implicit integer(a-c)
  implicit real(8)(d)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: More than one implicit type specified for 'a'
  implicit integer(f), real(a)
end
