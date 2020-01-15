! C1108  --  Save statement in a BLOCK construct shall not conatin a
!            saved-entity-list that does not specify a common-block-name

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


program  main
  integer x, y, z
  real r, s, t
  common /argmnt2/ r, s, t
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'argmnt1' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /argmnt1/
  block
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: SAVE statement in BLOCK construct may not contain a common block name 'argmnt2'
    save /argmnt2/
  end block
end program
