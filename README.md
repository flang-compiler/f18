<!--
Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
-->

# FIR

Working branch for FIR development.


1. Get the stuff.

```
  git clone http://llvm.org/git/llvm.git
  git clone git@github.com:schweitzpgi/mlir.git
  git clone git@github.com:schweitzpgi/f18.git 
```

2. Get "on" the right branches.

```
  (cd llvm; git checkout master)
  (cd mlir; git checkout f18)
  (cd f18; git checkout f18)
```
             
3. Setup the LLVM space for in-tree builds.
   
``` 
  cd llvm/projects ; ln -s ../../mlir .
  cd llvm/tools ; ln -s ../../f18 flang
```

4. Create a build space for cmake and make/ninja

```
  mkdir build; cd build; cmake /path/to/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=X86 ...
```


