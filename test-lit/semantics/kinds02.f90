! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: INTEGER(KIND=0) is not a supported type
integer(kind=0) :: j0
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: INTEGER(KIND=-1) is not a supported type
integer(kind=-1) :: jm1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: INTEGER(KIND=3) is not a supported type
integer(kind=3) :: j3
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: INTEGER(KIND=32) is not a supported type
integer(kind=32) :: j32
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=0) is not a supported type
real(kind=0) :: a0
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=-1) is not a supported type
real(kind=-1) :: am1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=1) is not a supported type
real(kind=1) :: a1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=7) is not a supported type
real(kind=7) :: a7
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REAL(KIND=32) is not a supported type
real(kind=32) :: a32
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX(KIND=0) is not a supported type
complex(kind=0) :: z0
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX(KIND=-1) is not a supported type
complex(kind=-1) :: zm1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX(KIND=1) is not a supported type
complex(kind=1) :: z1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX(KIND=7) is not a supported type
complex(kind=7) :: z7
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX(KIND=32) is not a supported type
complex(kind=32) :: z32
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX*1 is not a supported type
complex*1 :: zs1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX*2 is not a supported type
complex*2 :: zs2
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: COMPLEX*64 is not a supported type
complex*64 :: zs64
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: LOGICAL(KIND=0) is not a supported type
logical(kind=0) :: l0
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: LOGICAL(KIND=-1) is not a supported type
logical(kind=-1) :: lm1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: LOGICAL(KIND=3) is not a supported type
logical(kind=3) :: l3
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: LOGICAL(KIND=16) is not a supported type
logical(kind=16) :: l16
end program
