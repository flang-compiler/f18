## A Bijective Internal Name Mangling Proposal

FIR has a flat namespace.  No two objects may have the same name.  This
necessitates that we deploy some sort of name mangling scheme to unique
symbols from the front-end into FIR.

We also want to be able to reverse-mangle these names back to rederive
their Fortran names.

Fortran is case insensitive, which allows the compiler to convert the
user's identifiers to all lower case.  Such a universal conversion implies
that all upper case letters are available for use in mangling.

### Prefix `_Q`

All mangled names have the prefix sequence `_Q` to indicate the name has
been mangled.  (Q is chosen because it is a
[low frequency letter](http://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html)
in English.)

### Scope Building

Symbols can be scoped by the module, submodule, or procedure that contains
that symbol.  After the `_Q` sigil, names are constructed from outermost to
innermost scope as

   * Module name prefixed with `M`
   * Submodule name prefixed with `S`
   * Procedure name prefixed with `F`

Given:
```
    submodule (mod:s1mod) s2mod
      ...
      subroutine sub
        ...
      contains
        function fun
```

The mangled name of `fun` becomes:
```
    _QMmodSs1modSs2modFsubPfun
```

### Common blocks

   * A common block name will be prefixed with `B`

### Module scope global data

   * A global data entity is prefixed with `E`
   * A global entity that is constant will be prefixed with `EC`

### Procedures

   * A procedure is prefixed with `P`

Given:
```
    subroutine sub
```
The mangled name of `sub` becomes:
```
    _QPsub
```

### Derived types and related

   * A derived type is prefixed with `T`
   * If a derived type has KIND parameters, they are listed in a consistent
     canonical order where each takes the form `Ki` and where _i_ is the
     compile-time constant value. (All type parameters are integer.)  If _i_
     is a negative value, the prefix `KN` will be used and _i_ will reflect
     the magnitude of the value.

Given:
```
    module mymodule
      type mytype
        integer :: member
      end type
      ...
```
The mangled name of `mytype` becomes:
```
    _QMmymoduleTmytype
```

Given:
```
    type yourtype(k1,k2)
      integer, kind :: k1, k2
      real :: mem1
      complex :: mem2
    end type
```

The mangled name of `yourtype` where `k1=4` and `k2=-6` (at compile-time):
```
    _QTyourtypeK4KN6
```

   * A derived type dispatch table is prefixed with `D`.  The dispatch table
     for `type t` would be `_QDTt`
   * A type descriptor instance is prefixed with `C`.  Intrinsic types can
     be encoded with their names and kinds.  The type descriptor for the
     type `yourtype` above would be `_QCTyourtypeK4KN6`.  The type
     descriptor for `REAL(4)` would be `_QCrealK4`.

### Compiler generated names

Compiler generated names do not have to be mapped back to Fortran.  These
names will be prefixed with `_QQ` and followed by a unique compiler
generated identifier.
