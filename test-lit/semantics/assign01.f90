! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


integer :: a1(10), a2(10)
logical :: m1(10), m2(5,5)
m1 = .true.
m2 = .false.
a1 = [((i),i=1,10)]
where (m1)
  a2 = 1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: mask of ELSEWHERE statement is not conformable with the prior mask(s) in its WHERE construct
elsewhere (m2)
  a2 = 2
elsewhere
  a2 = 3
end where
end
