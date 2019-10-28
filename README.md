<!--
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
-->

# FIR

Working branch for FIR development.

## Monorepo

This is quite similar to the old way, but with a few subtle differences.

1. Get the stuff.

```
  git clone git@github.com:flang-compiler/f18-llvm-project.git
  git clone git@github.com:flang-compiler/f18-mlir.git
  git clone git@github.com:schweitzpgi/f18.git 
```

2. Get "on" the right branches.

```
  (cd f18-llvm-project ; git checkout f18)
  (cd f18-mlir ; git checkout f18)
  (cd f18 ; git checkout f18)
```
             
3. Setup the LLVM space for in-tree builds.
   
``` 
  (cd f18-llvm-project/llvm/projects ; ln -s ../../../f18-mlir mlir)
  (cd f18-llvm-project ; ln -s ../f18 flang)
```

4. Create a build space for cmake and make/ninja

```
  mkdir build
  cd build
  cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS=flang -DCMAKE_CXX_STANDARD=17 <other-arguments>
```

5. Build everything

One can, for example, do this with make as follows.

```
  make <make-arguments>
```

Or, of course, use their favorite build tool (such as ninja).
