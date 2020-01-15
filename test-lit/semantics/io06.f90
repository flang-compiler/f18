! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  character(kind=1,len=100) msg1
  character(kind=2,len=200) msg2
  integer(1) stat1
  integer(2) stat2
  integer(8) stat8

  open(10)

  backspace(10)
  backspace(10, iomsg=msg1, iostat=stat1, err=9)

  endfile(unit=10)
  endfile(iostat=stat2, err=9, unit=10, iomsg=msg1)

  rewind(10)
  rewind(iomsg=msg1, iostat=stat2, err=9, unit=10)

  flush(10)
  flush(iomsg=msg1, unit=10, iostat=stat8, err=9)

  wait(10)
  wait(99, id=id1, end=9, eor=9, err=9, iostat=stat1, iomsg=msg1)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate UNIT specifier
  backspace(10, unit=11)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate IOSTAT specifier
  endfile(iostat=stat2, err=9, unit=10, iostat=stat8, iomsg=msg1)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: REWIND statement must have a UNIT number specifier
  rewind(iostat=stat2)

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Duplicate ERR specifier
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate ERR specifier
  flush(err=9, unit=10, &
        err=9, &
        err=9)

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Duplicate ID specifier
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: WAIT statement must have a UNIT number specifier
  wait(id=id2, eor=9, id=id3)

9 continue
end
