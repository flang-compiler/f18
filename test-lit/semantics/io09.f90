! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: String edit descriptor in READ format expression
  read(*,'("abc")')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: String edit descriptor in READ format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  read(*,'("abc)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'H' edit descriptor in READ format expression
  read(*,'(3Habc)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'H' edit descriptor in READ format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  read(*,'(5Habc)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'I' edit descriptor 'w' value must be positive
  read(*,'(I0)')
end
