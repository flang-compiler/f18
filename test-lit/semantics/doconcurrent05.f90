! C1167 -- An exit-stmt shall not appear within a DO CONCURRENT construct if 
! it belongs to that construct or an outer construct.

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


subroutine do_concurrent_test1(n)
  implicit none
  integer :: n
  integer :: j,k
  mydoc: do concurrent(j=1:n)
  mydo:    do k=1,n
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
             if (k==5) exit mydoc
             if (j==10) exit mydo
           end do mydo
         end do mydoc
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(n)
  implicit none
  integer :: j,k,n
  mydoc: do concurrent(j=1:n)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
           if (k==5) exit
         end do mydoc
end subroutine do_concurrent_test2

subroutine do_concurrent_test3(n)
  implicit none
  integer :: j,k,n
  mytest3: if (n>0) then
  mydoc:    do concurrent(j=1:n)
              do k=1,n
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
                if (j==10) exit mytest3
              end do
            end do mydoc
          end if mytest3
end subroutine do_concurrent_test3

subroutine do_concurrent_test4(n)
  implicit none
  integer :: j,k,n
  mytest4: if (n>0) then
  mydoc:    do concurrent(j=1:n)
              do concurrent(k=1:n)
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
                if (k==5) exit
!ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: EXIT must not leave a DO CONCURRENT statement
                if (j==10) exit mytest4
              end do
            end do mydoc
          end if mytest4
end subroutine do_concurrent_test4
