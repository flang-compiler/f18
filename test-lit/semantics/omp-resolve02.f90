!OPTIONS: -fopenmp

! Test the effect to name resolution from illegal clause

! RUN: not %flang -fdebug-resolve-names -fparse-only -fopenmp %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  !a = 1.0
  b = 2
  !$omp parallel private(a) shared(b)
  a = 3.
  b = 4
  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: LASTPRIVATE clause is not allowed on the PARALLEL directive
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'a' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel private(a) shared(b) lastprivate(a)
  a = 5.
  b = 6
  !$omp end parallel
  !$omp end parallel
  print *,a, b
end
