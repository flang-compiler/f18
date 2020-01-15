! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  interface
    subroutine sub0
    end
    !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: A PROCEDURE statement is only allowed in a generic interface block
    procedure :: sub1, sub2
  end interface
contains
  subroutine sub1
  end
  subroutine sub2
  end
end
