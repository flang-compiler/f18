!OPTIONS: -Mstandard

! RUN: %flang -fdebug-resolve-names -fparse-only -Mstandard %s 2>&1 | FileCheck --check-prefixes=WARNING %s


  write(*, '(B0)')
  write(*, '(B3)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Expected 'B' edit descriptor 'w' value
  write(*, '(B)')

  !WARNING: [[@LINE+2]]:{{[0-9]+}}:{{.*}} Expected 'EN' edit descriptor 'w' value
  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Non-standard '$' edit descriptor
  write(*, '(EN,$)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Expected 'G' edit descriptor 'w' value
  write(*, '(3G)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Non-standard '\' edit descriptor
  write(*,'(A, \)') 'Hello'

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} 'X' edit descriptor must have a positive position value
  write(*, '(X)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Legacy 'H' edit descriptor
  write(*, '(3Habc)')

  !WARNING: [[@LINE+3]]:{{[0-9]+}}:{{.*}} 'X' edit descriptor must have a positive position value
  !WARNING: [[@LINE+2]]:{{[0-9]+}}:{{.*}} Expected ',' or ')' in format expression
  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} 'X' edit descriptor must have a positive position value
  write(*,'(XX)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Expected ',' or ')' in format expression
  write(*,'(RZEN8.2)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Expected ',' or ')' in format expression
  write(*,'(3P7I2)')

  !WARNING: [[@LINE+1]]:{{[0-9]+}}:{{.*}} Expected ',' or ')' in format expression
  write(*,'(5X i3)')
end
