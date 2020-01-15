! Test SELECT TYPE errors: C1157

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1()
  type :: t
  end type
  procedure(f) :: ff
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Selector is not a named variable: 'associate-name =>' is required
  select type(ff())
    class is(t)
    class default
  end select
contains
  function f()
    class(t), pointer :: f
    f => null()
  end function
end subroutine
