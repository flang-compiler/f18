! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  integer :: unit10 = 10
  integer :: unit11 = 11

  integer(kind=1) :: stat1
  integer(kind=8) :: stat8

  character(len=55) :: msg

  close(unit10)
  close(unit=unit11, err=9, iomsg=msg, iostat=stat1)
  close(12, status='Keep')

  close(iostat=stat8, 11) ! nonstandard

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: CLOSE statement must have a UNIT number specifier
  close(iostat=stat1)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate UNIT specifier
  close(13, unit=14, err=9)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate ERR specifier
  close(err=9, unit=15, err=9, iostat=stat8)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid STATUS value 'kept'
  close(status='kept', unit=16)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid STATUS value 'old'
  close(status='old', unit=17)

9 continue
end
