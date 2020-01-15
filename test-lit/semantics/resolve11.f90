! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  public i
  integer, private :: j
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The accessibility of 'i' has already been specified as PUBLIC
  private i
  !The accessibility of 'j' has already been specified as PRIVATE
  private j
end

module m2
  interface operator(.foo.)
    module procedure ifoo
  end interface
  public :: operator(.foo.)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The accessibility of operator '.foo.' has already been specified as PUBLIC
  private :: operator(.foo.)
  interface operator(+)
    module procedure ifoo
  end interface
  public :: operator(+)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The accessibility of 'operator(+)' has already been specified as PUBLIC
  private :: operator(+) , ifoo
contains
  integer function ifoo(x, y)
    logical, intent(in) :: x, y
  end
end module

module m3
  type t
  end type
  private :: operator(.lt.)
  interface operator(<)
    logical function lt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The accessibility of 'operator(<)' has already been specified as PRIVATE
  public :: operator(<)
  interface operator(.gt.)
    logical function gt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  public :: operator(>)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The accessibility of 'operator(.gt.)' has already been specified as PUBLIC
  private :: operator(.gt.)
end
