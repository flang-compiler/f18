
! Check for semantic errors in ALLOCATE statements

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine C943_C944(src, src2)
! C943
! No alloc-opt shall appear more than once in a given alloc-opt-list.
  character(50) msg
  integer stat, stat2
  real src(2:4), src2(2:4)
  real mld(2:4), mld2(2:4)
  real, allocatable :: x1(:), x2(:), x3(:), x4(:), x5(:), x6(:), x7(:), x8(:), x9(:)
  real, allocatable :: y1(:), y2(:), y3(:), y4(:)
  real, pointer :: p1, p2

  !Nominal cases, no error expected
  allocate(x1, source=src)
  allocate(x2, mold=mld)
  allocate(x3(2:4), stat=stat)
  allocate(x4(2:4), stat=stat, errmsg=msg)
  allocate(x5(2:4), source=src, stat=stat, errmsg=msg)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: STAT may not be duplicated in a ALLOCATE statement
  allocate(x6, stat=stat, source=src, stat=stat2)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SOURCE may not be duplicated in a ALLOCATE statement
  allocate(x7, source=src, stat=stat, source=src2)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: MOLD may not be duplicated in a ALLOCATE statement
  allocate(x8, mold=mld, stat=stat, mold=mld)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ERRMSG may not be duplicated in a ALLOCATE statement
  allocate(x9, mold=mld, errmsg=msg, stat=stat, errmsg= msg)

! C944
! At most one of source-expr and type-spec must appear.

  !Nominal cases already tested in C943 and type-spec tests (e.g C934)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: At most one of source-expr and type-spec may appear in a ALLOCATE statement
  allocate(real:: y1, source=src)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: At most one of source-expr and type-spec may appear in a ALLOCATE statement
  allocate(real:: y2, mold=mld)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: At most one of source-expr and type-spec may appear in a ALLOCATE statement
  allocate(y3, source=src, stat=stat, errmsg=msg, mold=mld)
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: At most one of source-expr and type-spec may appear in a ALLOCATE statement
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: At most one of source-expr and type-spec may appear in a ALLOCATE statement
  allocate(real:: y4, source=src, stat=stat, errmsg=msg, mold=mld)
end subroutine
