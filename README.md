<!--===- README.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# FIR

Working branch for FIR development.

## Monorepo now contains MLIR

### In-tree build

This is quite similar to the old way, but with a few subtle differences.

1. Get the stuff.

```
  git clone git@github.com:flang-compiler/f18-llvm-project.git
  git clone git@github.com:schweitzpgi/f18.git 
```

2. Get "on" the right branches.

```
  (cd f18-llvm-project ; git checkout mono)
  (cd f18 ; git checkout mono)
```
             
3. Setup the LLVM space for in-tree builds.
   
``` 
  (cd f18-llvm-project ; ln -s ../f18 flang)
```

4. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS="flang;mlir" -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DLLVM_INSTALL_UTILS=On <other-arguments>
```

5. Build everything

One can, for example, do this with make as follows.

```
  make <make-arguments>
```

Or, of course, use their favorite build tool (such as ninja).

### Out-of-tree build

1. Get the stuff is the same as above. Get the code from the same repos.

2. Get on the right branches. Again, same as above.

3. SKIP step 3 above. We're not going to build `flang` yet.

4. Create a build space for cmake and make (or ninja)

```
  mkdir build
  cd build
  export CC=<my-favorite-C-compiler>
  export CXX=<my-favorite-C++-compiler>
  cmake -GNinja ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DLLVM_INSTALL_UTILS=On -DCMAKE_INSTALL_PREFIX=<install-llvm-here> <other-arguments>
```

5. Build and install

```
  ninja
  ninja install
```

6. Add the new installation to your PATH

```
  PATH=<install-llvm-here>/bin:$PATH
```

7. Create a build space for another round of cmake and make (or ninja)

```
  mkdir build-flang
  cd build-flang
  cmake -GNinja ../f18 -DLLVM_DIR=<install-llvm-here> -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_CXX_STANDARD=17 -DLLVM_BUILD_TOOLS=On -DCMAKE_INSTALL_PREFIX=<install-flang-here> <other-arguments>
```
Note: if you plan on running lit regression tests, you should either:
- Use `-DLLVM_DIR=<build-llvm-here>` instead of `-DLLVM_DIR=<install-llvm-here>`
- Or, keep `-DLLVM_DIR=<install-llvm-here>` but add `-DLLVM_EXTERNAL_LIT=<path to llvm-lit>`.
A valid `llvm-lit` path is `<build-llvm-here>/bin/llvm-lit`.
Note that LLVM must also have been built with `-DLLVM_INSTALL_UTILS=On` so that tools required by tests like `FileCheck` are available in `<install-llvm-here>`.

8. Build and install

```
  ninja
  ninja install
```

### Running regression tests

Inside `build` for in-tree builds or inside `build-flang` for out-of-tree builds:

```
  ninja check-flang
```

Special CMake instructions given above are required while building out-of-tree so that lit regression tests can be run.

### Problems

Despite best efforts, there may be situations where the above repos will
get out of sync, and the build will fail.
