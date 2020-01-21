! Simple character assignment tests
subroutine assign1(s1, s2)
  character(*, 1) :: s1, s2
  s1 = s2
end subroutine
subroutine assign2(s1, s2)
  character(*, 2) :: s1, s2
  s1 = s2
end subroutine
subroutine assign4(s1, s2)
  character(*, 4) :: s1, s2
  s1 = s2
end subroutine

! Test substring assignment
subroutine assign_substring1(s1, s2, lb, ub)
  character(*, 1) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine
subroutine assign_substring2(s1, s2, lb, ub)
  character(*, 2) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine
subroutine assign_substring4(s1, s2, lb, ub)
  character(*, 4) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine

! Test overlapping assignments
! RHS depends on LHS in a way that if no temp is made to evaluate
! RHS, these tests should fail.
subroutine assign_overlap1(s1, s2, lb)
  character(*, 1) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine
subroutine assign_overlap2(s1, s2, lb)
  character(*, 2) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine
subroutine assign_overlap4(s1, s2, lb)
  character(*, 4) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine

! Test when length is given in specification expressions
subroutine assign_spec_expr_len1(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 1) :: s1
  character(l2, 1) :: s2
  s1 = s2
end subroutine
subroutine assign_spec_expr_len2(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 2) :: s1
  character(l2, 2) :: s2
  s1 = s2
end subroutine
subroutine assign_spec_expr_len4(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 4) :: s1
  character(l2, 4) :: s2
  s1 = s2
end subroutine
