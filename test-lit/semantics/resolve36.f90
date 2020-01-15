! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  integer i
end module

submodule(m1) s1
contains
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' was not declared a separate module procedure
  module subroutine i
  end
end submodule

module m2
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Declaration of 'i' conflicts with its use as module procedure
  integer i
contains
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 't' is already declared in this scoping unit
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'i' was not declared a separate module procedure
  module subroutine i
  end
end module

! Separate module procedure defined in same module as declared
module m3
  interface
    module subroutine sub
    end subroutine
  end interface
contains
  module procedure sub
  end procedure
end module

! Separate module procedure defined in a submodule
module m4
  interface
    module subroutine a
    end subroutine
    module subroutine b
    end subroutine
  end interface
end module
submodule(m4) s4a
contains
  module procedure a
  end procedure
end submodule
submodule(m4:s4a) s4b
contains
  module procedure b
  end procedure
end
