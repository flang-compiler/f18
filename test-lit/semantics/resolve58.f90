! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine s1(x, y)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Array pointer 'x' must have deferred shape or assumed rank
  real, pointer :: x(1:)  ! C832
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Allocatable array 'y' must have deferred shape or assumed rank
  real, dimension(1:,1:), allocatable :: y  ! C832
end

subroutine s2(a, b, c)
  real :: a(:,1:)
  real :: b(10,*)
  real :: c(..)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Array pointer 'd' must have deferred shape or assumed rank
  real, pointer :: d(:,1:)  ! C832
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Allocatable array 'e' must have deferred shape or assumed rank
  real, allocatable :: e(10,*)  ! C832
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-rank array 'f' must be a dummy argument
  real, pointer :: f(..)  ! C837
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-shape array 'g' must be a dummy argument
  real :: g(:,1:)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-size array 'h' must be a dummy argument
  real :: h(10,*)  ! C833
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Assumed-rank array 'i' must be a dummy argument
  real :: i(..)  ! C837
end

subroutine s3(a, b)
  real :: a(*)
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Dummy array argument 'b' may not have implied shape
  real :: b(*,*)  ! C836
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Implied-shape array 'c' must be a named constant
  real :: c(*)  ! C836
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Named constant 'd' array must have explicit or implied shape
  integer, parameter :: d(:) = [1, 2, 3]
end

subroutine s4()
  type :: t
    integer, allocatable :: a(:)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Component array 'b' without ALLOCATABLE or POINTER attribute must have explicit shape
    integer :: b(:)  ! C749
    real, dimension(1:10) :: c
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Array pointer component 'd' must have deferred shape
    real, pointer, dimension(1:10) :: d  ! C745
  end type
end

function f()
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Array 'f' without ALLOCATABLE or POINTER attribute must have explicit shape
  real, dimension(:) :: f  ! C832
end

subroutine s5()
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Allocatable array 'a' must have deferred shape or assumed rank
  integer :: a(10), b(:)
  allocatable :: a
  allocatable :: b
end subroutine
