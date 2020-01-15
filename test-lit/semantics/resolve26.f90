! RUN: not %flang -fdebug-resolve-names -fparse-only %s 2>&1 | FileCheck --check-prefixes=ERROR --implicit-check-not error: %s


module m1
  interface
    module subroutine s()
    end subroutine
  end interface
end

module m2
  interface
    module subroutine s()
    end subroutine
  end interface
end

submodule(m1) s1
end

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Error reading module file for submodule 's1' of module 'm2'
submodule(m2:s1) s2
end

!ERROR: [[@LINE+1]]:{{[0-9]+}}:{{.*}}error: Error reading module file for module 'm3'
submodule(m3:s1) s3
end
