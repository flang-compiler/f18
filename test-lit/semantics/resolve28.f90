! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s
  type t
  end type
  interface
    subroutine s1
      import, none
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT,NONE must be the only IMPORT statement in a scope
      import, all
    end subroutine
    subroutine s2
      import :: t
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT,NONE must be the only IMPORT statement in a scope
      import, none
    end subroutine
    subroutine s3
      import, all
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT,ALL must be the only IMPORT statement in a scope
      import :: t
    end subroutine
    subroutine s4
      import :: t
      !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT,ALL must be the only IMPORT statement in a scope
      import, all
    end subroutine
  end interface
end

module m
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT is not allowed in a module scoping unit
  import, none
end

submodule(m) sub1
  import, all !OK
end

submodule(m) sub2
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT,NONE is not allowed in a submodule scoping unit
  import, none
end

function f
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT is not allowed in an external subprogram scoping unit
  import, all
end

subroutine sub2()
  block
    import, all !OK
  end block
end

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: IMPORT is not allowed in a main program scoping unit
import
end
