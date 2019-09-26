The Flang driver and frontend option parsing
============================================

Status
======

As of September 2019, this work is draft.

* Changes in libclangDriver to add --driver-mode=flang:
  * RFC: http://lists.llvm.org/pipermail/cfe-dev/2019-June/062669.html.
  * (In progress) Clang Revision: https://reviews.llvm.org/D63607
  * (To be scheduled): Small changes to make pieces of libclangDriver
    and libclangFrontend less clang-specific.

* Introduction of new main() entrypoint bin/flang:
  This is the part which chooses which binaries to invoke for the
  frontend and linking.
  * (In progress) at https://github.com/flang-compiler/f18/pull/759

* Introduction of new fc1_main() entrypoint `bin/flang -fc1`
  * (Done) Describing the vision and implementation in this document.
  * (In progress) A straw-man implementation created by copying
    libFrontend from clang and removing all of the C++ specific parts.

Next steps:

* Make the straw-man implementation more capable and move it out of the clang
  namespace in a big renaming.

Summary
=======

The flang driver is implemented in terms of the clang driver, with the aim of
reusing code and being structured similarly so that familiarity of one driver
translates to familiarity with the other.

The frontend is implemented in a similar manner as clang, with some code
sharing. However, it is able to reuse less of `libclangFrontend` than
`libclangDriver` because there is a large amount of C-specific code there.

As of September 2019, it is a work in progress; the intent is to add initial
scaffolding, even if it is incomplete, so that it can be fleshed out and
improved.

Overarching vision
==================

The intent is to get to a point where anyone can contribute support for
individual driver options as early as possible. This requires putting a fair
amount of scaffolding in place.

Functionality can be reused from clang where it makes to do so. There are lots
of cases where it cannot be directly reused or extended because it is too
specific to C/C++-style languages. Sometimes, generalizing to support both
languages in a single construct would make things worse than the alternative of
having a "fork" of that construct.

The desired end state is to have a software architecture as extensible and
powerful as clang is. For example, clang has many different kinds of
FrontendAction that can be invoked on inputs, making use of different pieces of
the compiler pipeline.

Flang's frontend can be made by taking a verbatim copy of clang and deleting the
C-specific bits, as if putting together a statue by whittling away a block of
marble, this is described in "The Compiler Frontend" below.

`using namespace clang` should be avoided so that it is clear where clang
functionality is being used, and there should ideally be only a small amount of
`clang::` chatter in the final state.

Overview
========

Invoking the compiler is a multi-stage process. Consider what happens
when running `flang foobar.f90`:

1. The driver decides which processes to invoke to achieve the compile and link.

2. The compiler frontend, `flang -fc1` is invoked, producing an object file.

3. The linker is invoked to produce a binary.

(1) The Driver
==============

Compiler drivers are surprisingly complicated. Clang implements "a gcc-like
driver" in lib/Driver
[DriverInternals](https://clang.llvm.org/docs/DriverInternals.html), in 41,000
lines of code. Much of this logic can be reused.

## Short term

Fortunately `libclangDriver` was designed with reuse in mind, and so can be
linked to and used directly, and extended so that it can fulfill the needs of
flang. It only depends on `libclangBasic`, so that will be depended on as well.

Key concerns:

* Having a `--driver-mode=flang` which causes the driver to invoke flang rather
  than reuse its existing fall-back-to-gcc behaviour.

  **Status**: `--driver-mode=flang` is implemented in
  https://reviews.llvm.org/D6360.

* flang & clang should be able to compile a mix of C and Fortran inputs by
  invoking appropriate frontends. This is already handled by libclangDriver.
* flang --help should only list fortran-related options.
* Right now `TextDiagnosticPrinter` and related claseses live in
  `libclangFrontend`, and is required to instantiate a driver, which means
  copying it or depending on `libclangFrontend`. It could move in clang to
  `libclangBasic`, where the other diagnostic infrastructure lives.
* Many other small details.

## Longer-term

* (Re-)Instating helpful features that clang has. For example: the ability to
  load plugins.

* For the time being, `clang foo.f90` has the behaviour of invoking a fallback -
  GCC (gfortran) - so that it may be used in place of gcc in build scripts. Once
  f18 has landed in the LLVM project and it can do code generation, the
  fallback-to-gcc/gfortran behaviour could be removed from clang.

* If there is a community appetite, it is envisaged that libclangDriver could
  gradually become less clang-specific, and perhaps become a libLLVMDriver or
  alike.

(2) The Compiler Frontend
=========================

The compiler frontend will look similar to clang. As a prototype, it is being
made by taking a verbatim copy of clang and deleting the C-specific bits.

In the short term there will be two new libraries introduced:

* `libFortranFrontend`
  * `CompilerInstance.cpp`
  * `CompilerInvocation.cpp`
  * `FrontendAction.cpp`

* `libFortranFrontendTool`
  * `ExecuteCompilerInvocation.cpp`

In the fortran namespace, some classes will mirror what clang has:

* CompilerInstance
  Represents the compiler application. This holds compiler
  state which lives for the lifetime of the application. Clang has fields
  holding the CompilerInvocation, DiagnosticsEngine, TargetInfo, FileManager,
  SourceManager, etc.

* CompilerInvocation
  Represents the parsed form of all of the arguments used to
  invoke the compiler. It encapsulates numerous classes which represent options
  affecting different pieces of the compiler. For example, clang has
  LangOptions, TargetOptions, FileSystemOptions, CodeGenOptions, etc.

* FrontendInputFile
  Represents an input.

* FrontendAction represents an action to execute on an input. There will be
  classes representing each of the different compiler phases that flang can stop
  at.

There is much more to explore in this space, and this document will evolve to
reflect that.

## Short term

There is a pull request at
[#762](https://github.com/flang-compiler/f18/pull/762) which showcases the
approach. It includes a commit which takes a full copy of everything from clang,
followed by a commit which removes a large amount of functionality. This
approach means that the diff showing lines removed is useful: It can be used for
finding candidate features to implement in flang and serves as a reference for
implementing various features.

## Medium term

The immediate goal is to get the new frontend to the point where it can replace
the existing shell scripts, and work for the `-E` and `-fsyntax-only` cases, for
example.

This copied-from-clang structure needs wiring up to the existing f18 machinery.
Initially, this may result in quite a bit of glue code and some duplication
between the f18 and clang machinery.

Open issues
===========

* Options & where they live:
  Intent:
  * Introduce an FC1Options.td in the f18 repository which contains options
    which are known to the frontend.
  * In the clang driver, introduce a FlangOptions.td and include it into
    Options.td. Individual options will be marked as being applicable to flang
    or not.
  * Some options may have the same spelling between Fortran and C:
    Hypothetically, `-std=` might take on different values depending on the
    langauge being compiled. This will require careful thought on its own,
    especially handling the case where both C and Fortran are being compiled in
    a single driver invocation, with a different language standard specified for
    each, for example.

* Diagnostics.
  There is a lot of diagnostics machinery in clang. Unfortunately a lot of it is
  tied to the preprocessor and the AST.

* File system access, "source manager".
  * Clang has various useful things in libclangBasic which are close to being
    language agnostic, such as the FileManager and SourceManager. Right now they
    are marked as friend classes of clang::ASTReader/ASTWriter, but they don't
    introduce a dependency on the AST. There are things to do with macro
    expansion, but it seems plausible that they could be wired up in the context
    of f18.
