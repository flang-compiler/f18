! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  public
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: The default accessibility of this module has already been declared
  private
end

subroutine s
  !ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: PUBLIC statement may only appear in the specification part of a module
  public
end
