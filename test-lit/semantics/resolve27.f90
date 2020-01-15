! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m
  interface
    module subroutine s()
    end subroutine
  end interface
end

submodule(m) s1
end

submodule(m) s2
end

submodule(m:s1) s3
  integer x
end

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Module 'm' already has a submodule named 's3'
submodule(m:s2) s3
  integer y
end
