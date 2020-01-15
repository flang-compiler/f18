! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
end

subroutine sub
end

use m1
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Error reading module file for module 'm2'
use m2
!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: 'sub' is not a module
use sub
end
