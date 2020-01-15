! Testing 7.6 enum

! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


  ! OK
  enum, bind(C)
    enumerator :: red, green
    enumerator blue, pink
    enumerator yellow
    enumerator :: purple = 2
  end enum

  integer(yellow) anint4

  enum, bind(C)
    enumerator :: square, cicrle
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'square' is already declared in this scoping unit
    enumerator square
  end enum

  dimension :: apple(4)
  real :: peach

  enum, bind(C)
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'apple' is already declared in this scoping unit
    enumerator :: apple
    enumerator :: pear
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'peach' is already declared in this scoping unit
    enumerator :: peach
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'red' is already declared in this scoping unit
    enumerator :: red
  end enum

  enum, bind(C)
    !ERROR: [[@LINE+2]]:{{[0-9]+}}:{{.*}}error: Enumerator value could not be computed from the given expression
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Must be a constant value
    enumerator :: wrong = 0/0
  end enum

end
