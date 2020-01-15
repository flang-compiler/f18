! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  character(kind=1,len=50) internal_file
  character(kind=1,len=100) msg
  character(20) sign
  integer*1 stat1, id1
  integer*2 stat2
  integer*4 stat4
  integer*8 stat8
  integer :: iunit = 10
  integer, parameter :: junit = 11
  integer, pointer :: a(:)

  namelist /nnn/ nn1, nn2

  sign = 'suppress'

  open(10)

  write(*)
  write(*, *)
  write(*)
  write(*, *)
  write(unit=*) 'Ok'
  write(unit=iunit)
  write(unit=junit)
  write(unit=iunit, *)
  write(unit=junit, *)
  write(10)
  write(unit=10) 'Ok'
  write(*, nnn)
  write(10, nnn)
  write(internal_file)
  write(internal_file, *)
  write(internal_file, fmt=*)
  write(internal_file, fmt=1) 'Ok'
  write(internal_file, nnn)
  write(internal_file, nml=nnn)
  write(unit=internal_file, *)
  write(fmt=*, unit=internal_file)
  write(10, advance='yes', fmt=1) 'Ok'
  write(10, *, delim='quote', sign='plus') jj
  write(10, '(A)', advance='no', asynchronous='yes', decimal='comma', &
      err=9, id=id, iomsg=msg, iostat=stat2, round='processor_defined', &
      sign=sign) 'Ok'

  print*
  print*, 'Ok'

  allocate(a(2), stat=stat2)
  allocate(a(8), stat=stat8)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Duplicate UNIT specifier
  write(internal_file, unit=*)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: WRITE statement must have a UNIT specifier
  write(nml=nnn)

  !ERROR: [[@LINE+4]]:{{[0-9]+}}:{{.*}}error: WRITE statement must not have a BLANK specifier
  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: WRITE statement must not have a END specifier
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: WRITE statement must not have a EOR specifier
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: WRITE statement must not have a PAD specifier
  write(*, eor=9, blank='zero', end=9, pad='no')

  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: If NML appears, REC must not appear
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: If NML appears, FMT must not appear
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If NML appears, a data list must not appear
  write(10, nnn, rec=40, fmt=1) 'Ok'

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If UNIT=* appears, POS must not appear
  write(*, pos=n, nml=nnn)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If UNIT=* appears, REC must not appear
  write(*, rec=n)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If UNIT=internal-file appears, POS must not appear
  write(internal_file, err=9, pos=n, nml=nnn)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If UNIT=internal-file appears, REC must not appear
  write(internal_file, rec=n, err=9)

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If UNIT=* appears, REC must not appear
  write(*, rec=13) 'Ok'

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ADVANCE appears, UNIT=internal-file must not appear
  write(internal_file, advance='yes', fmt=1) 'Ok'

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ADVANCE appears, an explicit format must also appear
  write(10, advance='yes') 'Ok'

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid ASYNCHRONOUS value 'non'
  write(*, asynchronous='non')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  write(*, asynchronous='yes')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  write(internal_file, asynchronous='yes')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ID appears, ASYNCHRONOUS='YES' must also appear
  write(10, *, id=id) "Ok"

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If ID appears, ASYNCHRONOUS='YES' must also appear
  write(10, *, id=id, asynchronous='no') "Ok"

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: If POS appears, REC must not appear
  write(10, pos=13, rec=13) 'Ok'

  !ERROR: [[@LINE+4]]:{{[0-9]+}}:{{.*}}error: If DECIMAL appears, FMT or NML must also appear
  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: If ROUND appears, FMT or NML must also appear
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: If SIGN appears, FMT or NML must also appear
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid DECIMAL value 'Komma'
  write(10, decimal='Komma', sign='plus', round='down') jj

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: If DELIM appears, FMT=* or NML must also appear
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Invalid DELIM value 'Nix'
  write(delim='Nix', fmt='(A)', unit=10) 'Ok'

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: ID kind (1) is smaller than default INTEGER kind (4)
  write(id=id1, unit=10, asynchronous='Yes') 'Ok'

  write(*, '(X)')

1 format (A)
9 continue
end
