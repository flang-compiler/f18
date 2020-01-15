! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  write(*,*)
  write(*,'()')
  write(*,'(A)')
  write(*,'(2X:2X)')
  write(*,'(2X/2X)')
  write(*,'(3/2X)')
  write(*,'(3PF5.2)')
  write(*,'(+3PF5.2)')
  write(*,'(-3PF5.2)')
  write(*,'(000p,10p,0p)')
  write(*,'(3P7D5.2)')
  write(*,'(3P,7F5.2)')
  write(*,'(2X,(i3))')
  write(*,'(5X,*(2X,I2))')
  write(*,'(5X,*(2X,DT))')
  write(*,'(*(DT))')
  write(*,'(*(DT"value"))')
  write(*,'(*(DT(+1,0,-1)))')
  write(*,'(*(DT"value"(+1,000,-1)))')
  write(*,'(*(DT(0)))')
  write(*,'(S,(RZ),2E10.3)')
  write(*,'(7I2)')
  write(*,'(07I02)')
  write(*,'(07I02.01)')
  write(*,'(07I02.02)')
  write(*,'(I0)')
  write(*,'(G4.2)')
  write(*,'(G0.8)')
  write(*,'(T3)')
  write(*,'("abc")')
  write(*,'("""abc""")')
  write(*,'("a""""bc", 2x)')
  write(*,'(3Habc)')
  write(*,'(3Habc, 2X, 3X)')
  write(*,'(987654321098765432X)')
  write(*,'($)')
  write(*,'(\)')
  write(*,'(RZ,RU,RP,RN,RD,RC,SS,SP,S,3G15.3e2)')

  ! C1302 warnings; no errors
  write(*,'(3P7I2)')
  write(*,'(5X i3)')
  write(*,'(XEN)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Empty format expression
  write(*,"")

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Empty format expression
  write(*,"" // '' // "")

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Format expression must have an initial '('
  write(*,'I3')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '+' in format expression
  write(*,'(+7I2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '-' in format expression
  write(*,'(-7I2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'P' edit descriptor must have a scale factor
  write(*,'(P7F5.2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'P' edit descriptor must have a scale factor
  write(*,'(P7F' // '5.2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected integer constant
  write(*,'(X,3,3L4)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' before ')' in format expression
  write(*,'(X,i3,)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' in format expression
  write(*,'(X,i3,,)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' in format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' before ')' in format expression
  write(*,'(X,i3,,,)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' before ')' in format expression
  write(*,'(X,(i3,))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '*' in format expression
  write(*,'(*)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(*(DT(+1,0,=1)))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,+))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected integer constant in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,*))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected ',' or ')' in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,2*))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected ',' or ')' in 'DT' edit descriptor v-list
  write(*,'(DT(1,0,2*,+,?))')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Expected integer constant in 'DT' edit descriptor v-list
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(DT(1,0,*)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Expected ',' or ')' in 'DT' edit descriptor v-list
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(DT(1,0,2*,+,?)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unexpected '?' in format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' in format expression
  write(*,'(?,*(DT(+1,,1)))')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Repeat specifier before unlimited format item list
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unlimited format item list must contain a data edit descriptor
   write(*,'(5X,3*(2(X)))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Nested unlimited format item list
  write(*,'(D12.2,(*(F10.2)))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unlimited format item list must contain a data edit descriptor
  write(*,'(5X,*(2(X)))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Character in format after unlimited format item list
  write(*,'(*(Z5),*(2F20.3))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Character in format after unlimited format item list
  write(*,'(*(B5),*(2(I5)))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Character in format after unlimited format item list
  write(*,'(*(I5), D12.7)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'I' edit descriptor 'm' value is greater than 'w' value
  write(*,'(07I02.0 3)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'Z' edit descriptor 'm' value is greater than 'w' value
  write(*,'(07Z02.4)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'I' edit descriptor repeat specifier must be positive
  write(*,'(0I2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: List repeat specifier must be positive
  write(*,'(0(I2))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: List repeat specifier must be positive
  write(*,'(000(I2))')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: List repeat specifier must be positive
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'I' edit descriptor repeat specifier must be positive
  write(*,'(0(0I2))')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Kind parameter '_' character in format expression
  write(*,'(5_4X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '+' in format expression
  write(*,'(I+3)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '-' in format expression
  write(*,'(I-3)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '-' in format expression
  write(*,'(I-3, X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'X' edit descriptor must have a positive position value
  write(*,'(0X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'Y' in format expression
  write(*,'(XY)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'Y' in format expression
  write(*,'(XYM)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'M' in format expression
  write(*,'(MXY)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unexpected 'R' in format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'R' in format expression
  write(*,"(RR, RV)")

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unexpected '-' in format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'Y' in format expression
  write(*,'(I-3, XY)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'A' edit descriptor 'w' value must be positive
  write(*,'(A0)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'L' edit descriptor 'w' value must be positive
  write(*,'(L0)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected 'G' edit descriptor '.d' value
  write(*,'(G4)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'e' in 'G0' edit descriptor
  write(*,'(G0.8e)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected 'e' in 'G0' edit descriptor
  write(*,'(G0.8e2)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Kind parameter '_' character in format expression
  write(*,'(I5_4)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Kind parameter '_' character in format expression
  write(*,'(5_4P)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'T' edit descriptor must have a positive position value
  write(*,'(T0)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'T' edit descriptor must have a positive position value
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(T0')

  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: 'TL' edit descriptor must have a positive position value
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'T' edit descriptor must have a positive position value
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected 'EN' edit descriptor 'd' value after '.'
  write(*,'(TL0,T0,EN12.)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Expected 'EX' edit descriptor 'e' value after 'E'
  write(*,'(EX12.3e2, EX12.3e)')

  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: 'TL' edit descriptor must have a positive position value
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: 'T' edit descriptor must have a positive position value
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(TL00,T000')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(-')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(I3+')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(I3,-')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected integer constant
  write(*,'(3)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' before ')' in format expression
  write(*,'(3,)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' in format expression
  write(*,'(,3)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected ',' before ')' in format expression
  write(*,'(,)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(X')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(XX') ! C1302 warning is not an error

  !ERROR: [[@LINE+3]]:{{[0-9]+}}:{{.*}}error: Unexpected '@' in format expression
  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unexpected '#' in format expression
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '&' in format expression
  write(*,'(@@, #  ,&&& &&, ignore error 4)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Repeat specifier before 'TR' edit descriptor
  write(*,'(3TR0)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'TR' edit descriptor must have a positive position value
  write(*,'(TR0)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Kind parameter '_' character in format expression
  write(*,'(3_4X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Kind parameter '_' character in format expression
  write(*,'(1_"abc")')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unterminated string
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'("abc)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '_' in format expression
  write(*,'("abc"_1)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unexpected '@' in format expression
  write(*,'(3Habc, 3@, X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(4Habc)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unterminated 'H' edit descriptor
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(5Habc)')

  !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Unterminated 'H' edit descriptor
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Unterminated format expression
  write(*,'(50Habc)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Integer overflow in format expression
  write(*,'(9876543210987654321X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Integer overflow in format expression
  write(*,'(98765432109876543210X)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Integer overflow in format expression
  write(*,'(I98765432109876543210)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Integer overflow in format expression
  write(*,'(45I20.98765432109876543210, 45I20)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Integer overflow in format expression
  write(*,'(45' // '  I20.9876543' // '2109876543210, 45I20)')

  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Repeat specifier before '$' edit descriptor
  write(*,'(7$)')
end
