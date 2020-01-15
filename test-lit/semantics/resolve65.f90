! Test restrictions on what subprograms can be used for defined assignment.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  implicit none
  type :: t
  contains
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Defined assignment procedure 'binding' must be a subroutine
    generic :: assignment(=) => binding
    procedure :: binding => assign_t1
    procedure :: assign_t
    procedure :: assign_t2
    procedure :: assign_t3
    !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: Defined assignment subroutine 'assign_t2' must have two dummy arguments
    !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: In defined assignment subroutine 'assign_t3', second dummy argument 'y' must have INTENT(IN) or VALUE attribute
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: In defined assignment subroutine 'assign_t4', first dummy argument 'x' must have INTENT(OUT) or INTENT(INOUT)
    generic :: assignment(=) => assign_t, assign_t2, assign_t3, assign_t4
    procedure :: assign_t4
  end type
  type :: t2
  contains
    procedure, nopass :: assign_t
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Defined assignment procedure 'assign_t' may not have NOPASS attribute
    generic :: assignment(=) => assign_t
  end type
contains
  subroutine assign_t(x, y)
    class(t), intent(out) :: x
    type(t), intent(in) :: y
  end
  logical function assign_t1(x, y)
    class(t), intent(out) :: x
    type(t), intent(in) :: y
  end
  subroutine assign_t2(x)
    class(t), intent(out) :: x
  end
  subroutine assign_t3(x, y)
    class(t), intent(out) :: x
    real :: y
  end
  subroutine assign_t4(x, y)
    class(t) :: x
      integer, intent(in) :: y
  end
end

module m2
  type :: t
  end type
  interface assignment(=)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: In defined assignment subroutine 's1', dummy argument 'y' may not be OPTIONAL
    subroutine s1(x, y)
      import t
      type(t), intent(out) :: x
      real, optional, intent(in) :: y
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: In defined assignment subroutine 's2', dummy argument 'y' must be a data object
    subroutine s2(x, y)
      import t
      type(t), intent(out) :: x
      intent(in) :: y
      interface
        subroutine y()
        end
      end interface
    end
  end interface
end

! Detect defined assignment that conflicts with intrinsic assignment
module m5
  type :: t
  end type
  interface assignment(=)
    ! OK - lhs is derived type
    subroutine assign_tt(x, y)
      import t
      type(t), intent(out) :: x
      type(t), intent(in) :: y
    end
    !OK - incompatible types
    subroutine assign_il(x, y)
      integer, intent(out) :: x
      logical, intent(in) :: y
    end
    !OK - different ranks
    subroutine assign_23(x, y)
      integer, intent(out) :: x(:,:)
      integer, intent(in) :: y(:,:,:)
    end
    !OK - scalar = array
    subroutine assign_01(x, y)
      integer, intent(out) :: x
      integer, intent(in) :: y(:)
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Defined assignment subroutine 'assign_10' conflicts with intrinsic assignment
    subroutine assign_10(x, y)
      integer, intent(out) :: x(:)
      integer, intent(in) :: y
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Defined assignment subroutine 'assign_ir' conflicts with intrinsic assignment
    subroutine assign_ir(x, y)
      integer, intent(out) :: x
      real, intent(in) :: y
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Defined assignment subroutine 'assign_ii' conflicts with intrinsic assignment
    subroutine assign_ii(x, y)
      integer(2), intent(out) :: x
      integer(1), intent(in) :: y
    end
  end interface
end
