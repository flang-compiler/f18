! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1
  !OK: interface followed by type with same name
  interface t
  end interface
  type t
  end type
  type(t) :: x
  x = t()
end subroutine

subroutine s2
  !OK: type followed by interface with same name
  type t
  end type
  interface t
  end interface
  type(t) :: x
  x = t()
end subroutine

subroutine s3
  type t
  end type
  interface t
  end interface
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't' is already declared in this scoping unit
  type t
  end type
  type(t) :: x
  x = t()
end subroutine
