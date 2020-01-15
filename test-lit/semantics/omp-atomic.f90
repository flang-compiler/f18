! OPTIONS: -fopenmp

! Check OpenMP 2.13.6 atomic Construct

! RUN: %flang -fdebug-resolve-names -fparse-only -fopenmp %s 2>&1


  a = 1.0
  !$omp parallel num_threads(4)
  !$omp atomic seq_cst, read
  b = a

  !$omp atomic seq_cst write
  a = b
  !$omp end atomic

  !$omp atomic capture seq_cst
  b = a
  a = a + 1
  !$omp end atomic

  !$omp atomic
  a = a + 1
  !$omp end parallel
end
