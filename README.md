<!--===- README.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang

Flang is a ground-up implementation of a Fortran front end written in modern
C++.

## Getting Started

Read more about flang in the [documentation directory](documentation).
Start with the [compiler overview](documentation/Overview.md).

To better understand Fortran as a language
and the specific grammar accepted by flang,
read [Fortran For C Programmers](documentation/FortranForCProgrammers.md)
and
flang's specifications of the [Fortran grammar](documentation/f2018-grammar.txt)
and
the [OpenMP grammar](documentation/OpenMP-4.5-grammar.txt).

Treatment of language extensions is covered
in [this document](documentation/Extensions.md).

To understand the compilers handling of intrinsics,
see the [discussion of intrinsics](documentation/Intrinsics.md).

To understand how a flang program communicates with libraries at runtime,
see the discussion of [runtime descriptors](documentation/RuntimeDescriptor.md).

If you're interested in contributing to the compiler,
read the [style guide](documentation/C++style.md)
and
also review [how flang uses modern C++ features](documentation/C++17.md).

## Building Flang

### Supported C++ compilers

Flang is written in C++17.

The code has been compiled and tested with
GCC versions 7.2.0, 7.3.0, 8.1.0, and 8.2.0.

The code has been compiled and tested with
clang version 7.0 and 8.0
using either GNU's libstdc++ or LLVM's libc++.

### LLVM dependency

The instructions to build LLVM can be found at
https://llvm.org/docs/GettingStarted.html. If you are building flang as part
of LLVM, follow those instructions and add flang to `LLVM_ENABLE_PROJECTS`.

We highly recommend using the same compiler to compile both llvm and flang.

The flang CMakeList.txt file uses
the variable `LLVM_DIR` to find the installed LLVM components
and
the variable `MLIR_DIR` to find the installed MLIR components.

To get the correct LLVM and MLIR libraries included in your flang build,
define LLVM_DIR and MLIR_DIR on the cmake command line.
```
LLVM=<LLVM_BUILD_DIR>/lib/cmake/llvm \
MLIR=<LLVM_BUILD_DIR>/lib/cmake/mlir \
cmake -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR ...
```
where `LLVM_BUILD_DIR` is
the top-level directory where LLVM was built.

### Building f18 with GCC
=======
### Building flang with GCC
>>>>>>> Adjust README for upstreaming to llvm.

By default,
cmake will search for g++ on your PATH.
The g++ version must be one of the supported versions
in order to build flang.

Or,
cmake will use the variable CXX to find the C++ compiler.
CXX should include the full path to the compiler
or a name that will be found on your PATH,
e.g. g++-8.3, assuming g++-8.3 is on your PATH.
```
export CXX=g++-8.3
```
or
```
CXX=/opt/gcc-8.3/bin/g++-8.3 cmake ...
```

### Building flang with clang

To build flang with clang,
cmake needs to know how to find clang++
and the GCC library and tools that were used to build clang++.

CXX should include the full path to clang++
or clang++ should be found on your PATH.
```
export CXX=clang++

### Installation Directory

To specify a custom install location,
add
`-DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>`
to the cmake command
where `<INSTALL_PREFIX>`
is the path where flang should be installed.

### Build Types

To create a debug build,
add
`-DCMAKE_BUILD_TYPE=Debug`
to the cmake command.
Debug builds execute slowly.

To create a release build,
add
`-DCMAKE_BUILD_TYPE=Release`
to the cmake command.
Release builds execute quickly.

### Build Flang
```
cd ~/flang/build
cmake -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR ~/flang/src
make
```
### How to Run the Regression Tests

To run all tests:
```
cd ~/flang/build
cmake -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR ~/flang/src
make test check-all
```

To run individual regression tests llvm-lit needs to know the lit
configuration for flang. The parameters in charge of this are:
flang_site_config and flang_config. And they can be set as shown bellow:
```
<path-to-llvm-lit>/llvm-lit \
 --param flang_site_config=<path-to-flang-build>/test-lit/lit.site.cfg.py \
 --param flang_config=<path-to-flang-build>/test-lit/lit.cfg.py \
  <path-to-fortran-test>
```

# How to Generate FIR Documentation

If f18 was built with `-DLINK_WITH_FIR=On` (`On` by default), it is possible to
generate FIR language documentation by running `make flang-doc`. This will
create `docs/Dialect/FIRLangRef.md` in f18 build directory.
