# FIR Language Reference

This document describes the FIR dialect, an extension of the MLIR extensible
IR. FIR (Fortran IR) is a higher-level compiler intermediate representation for
Fortran compilation units used by the Flang compiler.

Some familiarity with [MLIR](https://github.com/tensorflow/mlir/blob/master/g3doc/LangRef.md) and [LLVM IR](https://llvm.org/docs/LangRef.html) is encouraged.  The [LLVM tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) may help as well.

# Type System

This section defines the _FIR type system_, which is related to but certainly
not the same as Fortran's concept and definition of a type. FIR is a strongly
typed language. FIR types subsume Fortran types and attributes that, when taken
together, describe the operational properties applicable to a Fortran entity.

Standard dialect MLIR types are available in FIR. FIR adds its own types as
well, and this section documents the FIR dialect type system.

In the most general case, the reified FIR type of a Fortran entity may not be
known at compile-time. Specifically, the size, rank, shape, index ranges,
strides, aggregate layout, data member sizes, and parameters of that type
instance may all be deferred until runtime.

In FIR, it is necessary to be able to abstract away much of the Fortran syntax
and runtime specialization of an entity's properties, while maintaining the
advantages of a strong type system. For one optimization, it may suffice to
know that a Fortran entity has a rank. For another, it may be desirable (or
required) for the operand entities to have a more precise reified FIR type for
efficiency and performance.

For example, for a Fortran variable with a declared type of `TYPE(*)`, it is
possible to express this variable in FIR as the most general sort of entity
with an unlimited polymorphic type. However, the compiler will be able to
produce better performing (and smaller) code if it can prove and rewrite this
value as having a FIR type of `i32`, which might be kept in a machine register,
for example.

The point here is that the FIR type system needs to allow transformations on a
Fortran entity that express both an unlimited polymorphic type and, say like in
the last example, a reified type as a 32-bit signed integer depending on the
context.

#### Notation

* _abc-list_ is a possibly empty comma separated list of _abc_.
* _abc-xlist_ is a non-empty 'x'-character separated list of _abc._
* _abc-type_ is a type.

## Fortran Intrinsic Types

<pre><code><b>!fir.int&lt;</b><em>kind</em><b>&gt;<br>
!fir.real&lt;</b><em>kind</em><b>&gt;<br>
!fir.complex&lt;</b><em>kind</em><b>&gt;<br>
!fir.logical&lt;</b><em>kind</em><b>&gt;<br>
!fir.character&lt;</b><em>kind</em><b>&gt;</b><br>
&nbsp;&nbsp;&nbsp; where <em>kind</em> := <em>integer-constant</em><br>
</code></pre>

Some of these types may be directly rewritten to standard dialect types. The
intrinsic types are meant to correspond one-to-one with the front-end's
intrinsic types. What a particular kind-value means in terms of bit-width is
not relevant to these FIR types. However, the semantics of kind-values are
absolutely relevant when these types are converted to a lower-level
dialect. For example, assuming the f18 front-end definition then `!fir.int<2>`
can be trivially rewritten to `i16`.

## Fortran Derived Types

<pre><code><b>!fir.type&lt;</b><em>derived-type-name [</em> <b>(</b><em>len-param-list</em><b>)</b> <em>] [</em> <b>{</b><em>field-id-list</em><b>}</b> <em>]</em> <b>&gt;</b><br>
&nbsp;&nbsp;&nbsp; where <em>len-param</em> := <em>len-param-name</em> <b>:</b> <em>integer-type</em><br>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>field-id</em> := <em>field-name</em> <b>:</b> <em>type</em><br>
</code></pre>

This is the type of a Fortran derived type. Each derived type is given a unique
name. The data parts of the derived type are elaborated in the field list. The
degenerate form is allowed to be able to construct recursive references. As
these elaborated record types can become rather verbose, the intention will be
to exploit MLIR's type alias feature to create shorter nicknames.

## Modeled Types

These types are not Fortran types. These modeled types capture the semantic
properties of attributes on Fortran data entities by converting those
attributes into the constraints of FIR's static type system.

### Function Type

<pre><code><b>(</b><em>argument-type-list</em><b>) -&gt; </b><em>result-type</em><br>
</code></pre>

This is the type of a Fortran procedure (function or subroutine) and is the
same as the standard dialect. The void type is simply `()`.

### Sequence Type

<pre><code><b>!fir.array&lt;</b><em>extent-xlist</em> <b>:</b> <em>element-type</em><b>&gt;</b><br>
&nbsp;&nbsp;&nbsp; where <em>extent</em> := <em>integer-constant |</em> <b>?</b> <em>|</em> <b>*</b><br>
</code></pre>

This is used to declare that the value is an array.  It may additionally
capture the shape of the array. It maps to a Fortran column-major layout. An
array with explicit shape can be represented with an xlist of rank length of
integers. An unknown extent in one of the dimensions can be specified with the
`?` character. The `*` character is used (by itself) when the shape of the
array is not known.

Examples:

```mlir
    !fir.array<10x10:i32>	; array of rank 2 and shape [10, 10]
    !fir.array<5x?:f32>	; array of rank 2 and shape [5, unknown]
    !fir.array<*:f64>	; array of unknown rank and shape
```

### Pointer-like Types

<pre><code><b>!fir.ref&lt;</b><em>ref-to-type</em><b>&gt;</b><br>
</code></pre>

This is the type of a memory reference. This is needed in a number of
situations. For one example, dummy arguments (variables) in Fortran can refer
to the actual argument passed from the calling procedure. Assignments to the
dummy variable will change the value of the actual variable. This semantics can
be reified by passing a memory reference to the actual variable to the called
procedure. _ref-to-type_ cannot be `!fir.ref`.

<pre><code><b>!fir.ptr&lt;</b><em>ptr-to-type</em><b>&gt;</b><br>
</code></pre>

This is also a memory reference type, but it is limited to Fortran's POINTER
attribute. A variable with a POINTER attribute has a runtime defined reference
value. This introduces aliasing (which may be important to the optimizer), and
having a separate type allows for some refinement in alias
analysis. _ptr-to-type_ cannot be `!fir.ptr`, `!fir.heap`, or `!fir.ref`.

<pre><code><b>!fir.heap&lt;</b><em>heap-to-type</em><b>&gt;</b><br>
</code></pre>

This is the third memory reference type, and this one is limited to Fortran's
ALLOCATABLE attribute. The allocation of ALLOCATABLE variables will return a
reference of `!fir.heap` type. _heap-to-type_ cannot be `!fir.ptr`,
`!fir.heap`, or `!fir.ref`.

By having different pointer-like types, the language constraint prohibiting
first-class pointers-to-pointers can be trivially enforced. For example, C852
prohibits having both a POINTER and ALLOCATABLE attribute on entity. This can
be enforced in the type system by disallowing the construction of
`!fir.ptr<!fir.heap<T>>` and `!fir.heap<!fir.ptr<T>>` types.

### Descriptor Types

<pre><code><b>!fir.box&lt;</b><em>of-type</em><b>&gt;</b><br>
</code></pre>

The type of the most general object in Fortran. A boxed value can be a scalar
or an array. A boxed value has a memory reference (to the value's data), a
reified type value, optional type parameters, and optional array dimension
information (if it's an array box). A value of box type can return dynamic
values such as the rank of the object, the size of an element, or the size of
the object.

<pre><code><b>!fir.boxchar&lt;</b><em>kind</em><b>&gt;</b><br>
</code></pre>

A Fortran CHARACTER type can be a pair of values, the buffer of characters
along with a runtime LEN value. This is the abstract type of the pair that
describes a CHARACTER value.

<pre><code><b>!fir.boxproc&lt;</b><em>function-type</em><b>&gt;</b><br>
</code></pre>

A Fortran procedure POINTER may be to an internal procedure. An internal
procedure may require a runtime host instance value. This is the abstract type
of a procedure pointer, including a procedure pointer with a host instance
value.

### Other Types

<pre><code><b>!fir.dims&lt;</b><em>rank</em><b>&gt;</b><br>
</code></pre>

This is the type of a vector of array dimension triples. A boxed array object
takes a vector of dimension triples to properly instantiate the box value.

<pre><code><b>!fir.field</b><br>
</code></pre>

This is the type of a part reference via a field name. A field value can be
constructed to provide an abstract value to refer to members of a value of type
`!fir.type`. In the most general case, the layout of a parametric derived type
may not be known until runtime and the offset of a particular field must be
computed.

<pre><code><b>!fir.tdesc&lt;</b><em>of-type</em><b>&gt;</b><br>
</code></pre>

A Fortran type has a meta-type known as a type descriptor. This is the type of
these meta-types. A boxed value carries a `!fir.tdesc` value to identify the
type of that instance.


# Modules, Functions, and Basic Blocks

For organizing a Fortran compilation unit, FIR uses the MLIR infrastructure for
modules, functions, regions, and basic blocks.  These concepts are explained in
the MLIR Language Reference.


# Operations

This section defines the _FIR operations_.  MLIR operations are a generalized
abstraction meant to allow a dialect, like FIR, to add its own constructs with
their own semantics.  FIR operations are meant to capture the operations and
structures from the Fortran language for presentation to optimization passes.

Some FIR operations capture execution semantics and are intended to be placed
in Blocks. Other operations are Module level abstractions and intended to be
referenced by name.

## Executable Operations

### SSA Memory Related Ops

#### `fir.alloca`

Syntax:	<code><b>fir.alloca</b> <em>T [</em> <b>,</b> <em>size-list ]</em> <b>: !fir.ref&lt;</b><em>T</em><b>&gt;</b></code>

Allocate uninitialized space on the stack for a variable of type _T_.  If
allocating an array of _T_, then a size-list of ssa-values sufficient rank must
be provided to compute the array's shape.

Example:

```mlir
    %11 = fir.alloca i32 : !fir.ref<i32>
    %12 = fir.alloca !fir.array<8:i64> : !fir.ref<!fir.array<8:i64>>
    %13 = fir.alloca f32, %5 : !fir.ref<f32>
```

Note that in the case of `%13`, a contiguous block of memory is allocated and
its size is some runtime multiple of a 32-bit REAL value. Furthermore, the
operation is undefined if the ssa-value `%5` is nonpositive.


#### `fir.load`

Syntax:	<code><b>fir.load</b> <em>memory-reference</em> <b>:</b> <em>reference-type</em></code>

Loads a value from a memory reference. A memory reference has type
`!fir.ref<T>`, `!fir.heap<T>`, or `!fir.ptr<T>`.

Example:

```mlir
    %14 = fir.alloca i32 : !fir.ref<i32>
    %15 = fir.load %14 : !fir.ref<i32>
```

#### `fir.store`

Syntax:	<code><b>fir.store</b> <em>ssa-value</em> <b>to</b> <em>memory-reference</em> <b>:</b> <em>reference-type</em></code>

Store a value to a memory reference.

Example:

```mlir
    %16 = fir.call @foo() : f64
    %17 = fir.call @bar() : !fir.ptr<f64>
    fir.store %16 to %17 : !fir.ptr<f64>
```

The above store changes the value to which the pointer is pointing and not
the pointer itself.


#### `fir.undefined`

Syntax:	<code><b>fir.undefined</b> <em>T</em></code>

An undefined value. This is a constant that can be used to represent an
undefined _ssa-value_ of any type except
<code>!fir.ref&lt;<em>U</em>&gt;</code>.

Example:

```mlir
    %18 = fir.undefined !fir.array<10:!fir.type<T{...}>>
```

### Heap Memory Ops

#### `fir.allocmem`

Syntax:	<code><b>fir.allocmem</b> <em>T [</em> <b>,</b> <em>size-list ]</em> <b>: !fir.heap&lt;</b><em>T</em><b>&gt;</b></code>

Allocate contiguous memory on the heap. It is expected that a properly
constructed FIR program properly pairs `fir.allocmem` and `fir.freemem`
operations.


Example:

```mlir
    %20 = fir.allocmem !fir.type<Z(p:i32){field:i32}> : !fir.heap<!fir.type<Z(p:i32){field:i32}>>
```

#### `fir.freemem`


Syntax:	<code><b>fir.freemem</b> <em>heap-value</em> <b>: !fir.heap&lt;</b><em>T</em><b>&gt;</b></code>

Deallocate a previously allocated block of memory returned from
`fir.allocmem`.

Example:

```mlir
    %21 = fir.allocmem !fir.type<Z(p:i32){field:i32}> : !fir.heap<!fir.type<"Z"(p:i32){field:i32}>>
    ...
    fir.freemem %21 : !fir.heap<!fir.type<"Z"(p:i32){field:i32}>>
```


### Terminators

#### `fir.select`


Syntax:
<pre><code><b>fir.select</b> <em>selector</em> <b>:</b> <em>selector-type</em> <b>[</b> <em>value-target-list</em> <b>]</b><br>
&nbsp;&nbsp;&nbsp; where <em>value-target</em> := <em>select-const</em> <b>,</b> <em>block block-arg-list</em><br>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>select-const</em> := <em>integer-const |</em> <b>unit</b><br>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>block-arg-list</em> := <em>[</em> <b>(</b> <em>value-type-list</em> <b>)</b> <em>]</em>
</code></pre>

A terminator for a simple switch like control flow.

Example:

    fir.select %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), -3,^bb3(%arg2,%2:i32,i32), 4,^bb4(%1:i32), unit,^bb5 ]


#### `fir.select_case`

Syntax:
<pre><code><b>fir.select_case</b> <em>selector</em> <b>:</b> <em>selector-type</em> <b>[</b> <em>case-target-list</em> <b>]</b><br>
&nbsp;&nbsp;&nbsp; where <em>case-target-list</em> := <em>case-attr</em> <b>,</b> <em>case-attr-values</em> <b>,</b> <em>block block-arg-list</em><br>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>case-attr</em> := <b>unit</b> <em>|</em> <b>#fir.point</b> <em>|</em> <b>#fir.interval</b> <em>|</em> <b>#fir.lower</b> <em>|</em> <b>#fir.upper</b><br>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>case-attr-values</em> := <em>[ ssa-value [</em> <b>,</b> <em>ssa-value ] ]</em>
</code></pre>

A terminator for the SELECT CASE construct.

Example:

    fir.select_case %arg : i32 [#fir.point, %0, ^bb1(%0:i32), #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1:i32,i32,i32,i32), #fir.interval, %2, %3, ^bb3(%2,%arg2:i32,i32), #fir.upper, %arg, ^bb4(%1:i32), unit, ^bb5]

#### `fir.select_rank`

Syntax:	<code><b>fir.select_rank</b> <em>selector</em> <b>:</b> <em>selector-type</em> <b>[</b> <em>value-target-list</em> <b>]</b></code>

A terminator for the SELECT RANK construct.

Example:

    fir.select_rank %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), 3,^bb3(%arg2,%2:i32,i32), -1,^bb4(%1:i32), unit,^bb5 ]


#### `fir.select_type`

Syntax:
<pre><code><b>fir.select_type</b> <em>selector</em> <b>[</b> <em>type-target-list</em> <b>]</b></code><br>
&nbsp;&nbsp;&nbsp; where <em>type-target-list</em> := <em>type-attr</em> <b>,</b> <em>block block-arg-list</em>
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>case-attr</em> := <b>unit</b> <em>|</em> <b>#fir.instance&lt;</b><em>type</em><b>&gt;</b> <em>|</em> <b>#fir.subsumed&lt;</b><em>type</em><b>&gt;</b>
</code></pre>

A terminator for the SELECT TYPE construct.

Example:

    fir.select_type %arg : !fir.box<()> [ #fir.instance<!fir.type<type1{f1:i32,f2:i64,f3:i1}>>,^bb1(%0:i32), #fir.instance<!fir.type<type2{...}>>,^bb2(%2:i32), #fir.subsumed<!fir.type<type3{...}>>,^bb3(%2:i32), #fir.instance<!fir.type<type4{...}>>,^bb4(%1:i32), unit,^bb5 ]

#### `fir.unreachable`

Syntax:	`fir.unreachable`

A terminator that should never be reached by the executing program.  This
terminator is translated to LLVM's `unreachable` instruction.


### Ops for Boxed Values


#### Packing Boxed Values


#### `fir.embox`


Syntax:
<pre><code><b>fir.embox</b> <em>mem-ref [</em> <b>,</b> <em>access-map ]</em> <b>: (</b> <em>arg-type-list</em> <b>) -&gt; !fir.box&lt;</b><em>T</em><b>&gt;</b><br>
&nbsp;&nbsp;&nbsp; where <em>access-map</em> := <em>dims | affine-map</em>
</code></pre>

Creation of a boxed value.  A boxed value is a memory reference value that
is wrapped with a Fortran descriptor.  References to scalars, arrays,
pointers, allocatables, etc. can be boxed.

Example:

```mlir
    %34 = fir.dims(%c1, %c10, %c1) : (i32, i32, i32) -> !fir.dims<1>
    %35 = fir.call @foo() : () -> !fir.ref<!fir.array<10:i32>>
    %36 = fir.embox %35, %34 : (!fir.ref<!fir.array<10:i32>>, !fir.dims<1>) -> !fir.box<!fir.array<10:i32>>
```

#### `fir.emboxchar`

Syntax:	<code><b>fir.emboxchar</b> <em>buffer-ref</em> <b>,</b> <em>len-value</em> <b>: (</b> <em>arg-type-list</em> <b>) -&gt; !fir.boxchar&lt;</b><em>kind</em><b>&gt;</b></code>

Creation of a boxed CHARACTER pair.  A variable of type CHARACTER has a
dependent LEN type parameter that is the size of the buffer holding the
CHARACTER value.

Example:

```mlir
    %c20 = constant 20 : i32
    %37 = fir.call @foo() : !fir.ref<!fir.character<1>>
    %38 = fir.emboxchar %37, %c20 : !fir.boxchar<1>
```

#### `fir.emboxproc`


Syntax:	<code><b>fir.emboxproc</b> <em>callee [</em> <b>,</b> <em>context ]</em> <b>: (</b> <em>arg-type-list</em> <b>) -&gt; !fir.boxproc&lt;(</b><em>T</em><b>) -&gt;</b> <em>U</em><b>&gt;</b></code>

Creation of a boxed procedure reference.

Example:

```mlir
    %39 = fir.emboxproc @proc_xyz
```

#### Unpacking Boxed Values


#### `fir.unbox`


Syntax:	<code><b>fir.unbox</b> <em>box-value</em> <b>: (</b> <em>arg-type</em> <b>) -&gt; (</b>!fir.ref&lt;<em>Tx</em><b>&gt;, i</b><em>v</em><b>, i</b><em>w</em><b>, !fir.tdesc&lt;</b><em>Tx</em><b>&gt;, i</b><em>y</em><b>, !fir.dims&lt;</b><em>z</em><b>&gt;)</b></code>

Unbox a boxed value into a result of multiple values from the box's
component data.  The values are, minimally, a reference to the data of the
entity, the byte-size of one element, the rank, the type descriptor, a set
of flags (packed in an integer, and an array of dimension information (of
size rank).

Example:

```mlir
    %40 = fir.call @foo() : !fir.box<!fir.type<"T"{field:i32}>>
    %41 = fir.unbox %40 : (!fir.box<!fir.type<"T"{field:i32}>>) -> (!fir.ref<!fir.type<"T"{field:i32}>>, i32, i32, !fir.tdesc<!fir.type<"T"{field:i32}>>, i32, !fir.dims<4>)
```

Note: the exact type and content of the returned multiple value is still to
be determined and may change.

#### `fir.unboxchar`

Syntax:	<code><b>fir.unboxchar</b> <em>boxchar-value</em> <b>: (</b> <em>boxchar-type</em> <b>) -&gt; (</b> <em>reference</em> <b>,</b> <em>len</em> <b>)</b></code>

Unbox a boxed CHARACTER pair.

Example:

```mlir
    %45 = fir.call @foo() : !fir.boxchar<1>
    %46 = fir.unboxchar %45 : (!fir.boxchar<1>) -> (!fir.ref<!fir.character<1>>, i32)
```

#### `fir.unboxproc`


Syntax:	<code><b>fir.unboxproc</b> <em>boxproc-value</em> <b>: (</b> <em>boxproc-type</em> <b>) -> (</b> <em>callee</em> <b>,</b> <em>context</em> <b>)</b></code>

Unbox a boxed procedure reference.


Example:

```mlir
    %47 = fir.call @foo() : () -> !fir.boxproc<() -> i32>
    %48 = fir.unboxproc %47 : (!fir.ref<() -> i32>, !fir.ref<(f32, i32)>)
```

#### Queries on Boxed Values

#### `fir.box_addr`


Syntax:	<code><b>fir.box_addr</b> <em>boxable</em> <b>: (</b> <em>box-type</em> <b>) -&gt; !fir.ref&lt;</b><em>T</em><b>&gt;</b></code>


Return the referenced entity from the boxed value. The boxable value must
have a type of <code>!fir.box&lt;<em>T</em>&gt;</code>,
<code>!fir.boxchar&lt;<em>C</em>&gt;</code>, or
<code>!fir.boxproc&lt;<em>FT</em>&gt;</code>.


Example:

```mlir
    %51 = fir.box_addr %boxvec : (!fir.box<!fir.array<?:f64>>) -> !fir.ref<!fir.array<?:f64>>
```

#### `fir.box_dims`


Syntax:	<code><b>fir.box_dims</b> <em>box-value</em> <b>,</b> <em>dim</em> <b>: (</b> <em>box-type</em> <b>) -&gt; (i</b><em>n</em><b>, i</b><em>n</em><b>, i</b><em>n</em><b>)</b></code>


Return the dimension vector for the boxed value, _box-value_, at dimension,
_dim_. The returned value is the triple of lower bound, extent, and stride,
respectively.  If _dim_ is larger than the rank of the boxed value, this
operation has undefined behavior.


Example:

```mlir
    %c1 = constant 1 : i32
    %52 = fir.box_dims %40, %c1 : (!fir.box<!fir.array<*:f64>>, i32) -> (i32, i32, i32)
```

#### `fir.box_elesize`


Syntax:	<code><b>fir.box_elesize</b> <em>box-value</em> <b>: (</b> <em>box-type</em> <b>) -&gt; i</b><em>n</em></code>


Return the size of an element for the boxed value. The returned value may
not be constant and only known at runtime.


Example:

```mlir
    %53 = fir.box_elesize %40 : (!fir.box<!fir.array<*:f64>>, i32) -> i32
```

#### `fir.box_isalloc`


Syntax:	<code><b>fir.box_isalloc</b> <em>box-value</em> <b>: (</b> <em>box-type</em> <b>) -&gt; i1</b></code>


Return true if the boxed value is an ALLOCATABLE. This will return true if
the originating _box-value_ was from a `fir.embox` with a _mem-ref_ value
that had the type <code>!fir.ref&lt;!fir.heap&lt;<em>T</em>&gt;&gt;</code>.


Example:

```mlir
    %54 = fir.box_isalloc %40 : (!fir.box<!fir.array<*:f64>>, i32) -> i1
```

#### `fir.box_isarray`

Syntax:	<code><b>fir.box_isarray</b> <em>box-value</em> <b>:  (</b> <em>box-type</em> <b>) -&gt; i1</b></code>


Return true if the boxed value has a rank greater than 0. This will return
true if the originating _box-value_ was from a `fir.embox` with a _mem-ref_
value that had the type <code>!fir.ref<!fir.array<<em>T</em>>></code> and a
<em>dims</em> argument.


Example:

```mlir
    %55 = fir.box_isarray %40 : (!fir.box<!fir.array<*:f64>>, i32) -> i1
```

#### `fir.box_isptr`


Syntax:	<code><b>fir.box_isptr</b> <em>box-value</em> <b>: (</b> <em>box-type</em> <b>) -&gt; i1</b></code>

Return true if the boxed value is a POINTER. This will return true if the
originating _box-value_ was from a `fir.embox` with a _mem-ref_ value that
had the type <code>!fir.ref&lt;!fir.ptr&lt;<em>T</em>&gt;&gt;</code>.

Example:

```mlir
    %56 = fir.box_isptr %40 : (!fir.box<!fir.array<*:f64>>, i32) -> i1
```

#### `fir.box_rank`

Syntax:	<code><b>fir.box_rank</b> <em>box-value</em> <b>: (</b> <em>box-type</em> <b>) -&gt; i</b><em>n</em></code>

Return the rank of the boxed value. The rank of a scalar is 0.

Example:

```mlir
    %57 = fir.box_rank %40 : (!fir.box<!fir.array<*:f64>>, i32) -> i32
```

#### `fir.box_tdesc`

Syntax:	<code><b>fir.box_tdesc</b> <em>box-value</em> <b>: (</b> <em>box-type</em> <b>) -&gt; !fir.tdesc&lt;</b> <em>ele-type</em> <b>&gt;</b></code>

Return the type descriptor of the boxed value.

Example:

```mlir
    %58 = fir.box_tdesc %40 : (!fir.box<!fir.array<*:f64>>) -> !fir.tdesc<!fir.array<*:f64>>
```

#### `fir.boxchar_len`

Syntax:	<code><b>fir.boxchar_len</b> <em>boxchar-value</em> <b>: (!fir.boxchar&lt;1&gt;) -&gt; i</b><em>n</em></code>

Return the LEN type parameter of a boxchar value.

Example:

```mlir
    %59 = fir.boxchar_len %45 : (!fir.boxchar<1>) -> i32
```

#### `fir.boxproc_host`


Syntax:	<code><b>fir.boxproc_host</b> <em>boxproc-value</em> <b>: (</b> <em>boxproc-type</em> <b>) -&gt; </b><em>host-context</em></code>


Return the host context of a boxproc value, if any.


Example:

```mlir
    %60 = fir.boxproc_host %47 : (!fir.boxproc<() -> none>) -> (() -> none, !fir.ref<(f32, i64)>)
```

The content and type of _host-context_ is to be determined.


### Ops for Derived Types and Arrays

#### `fir.coordinate_of`

Syntax:	<code><b>fir.coordinate_of</b> <em>box-or-ref-value</em> <b>,</b> <em>index-field-list</em> <b>: (</b> <em>reference-like-type</em> <b>) -&gt; !fir.ref&lt;</b><em>T</em><b>&gt;</b></code>

Compute the internal coordinate address starting from a boxed value or
unboxed memory reference. Returns a memory reference.

Example:

```mlir
    %57 = fir.call @foo() : () -> !fir.heap<!fir.array<?:f32>>
    %58 = fir.coordinate_of %57, %56 : (!fir.heap<!fir.array<?:f32>>, index) -> !fir.ref<f32>
```

#### `fir.extract_value`

Syntax:	<code><b>fir.extract_value</b> <em>entity</em> <b>,</b> <em>index-field-list</em> <b>: (</b> <em>entity-type</em> <b>,</b> <em>index-type</em> <b>) -&gt;</b> <em>subobject-type</em></code>

Extract a value from an entity with a type composed of arrays and/or
derived types. Returns the value from _entity_ with the type of the
specified component.

Example:

```mlir
    %59 = fir.field_index("field") : !fir.field
    %60 = fir.call @foo3() : () -> !fir.type<X{field:i32}>
    %61 = fir.extract_value %60, %59 : (!fir.type<X{field:i32}>, !fir.field) -> i32
```

#### `fir.field_index`

Syntax:	<code><b>fir.field_index ("</b><em>field-name</em><b>") : !fir.field</b></code>

Compute the field offset of a particular named field in a derived
type. Note: it is possible in Fortran to write code that can only determine
the exact offset of a particular field in a parameterized derived type at
runtime.

Example:

```mlir
    %62 = fir.field_index ("member_1") : !fir.field
```

#### `fir.gendims`

Syntax:	<code><b>fir.gendims</b> <em>triple-list</em> <b>: (</b> <em>type-list</em> <b>) -&gt; !fir.dims&lt;</b><em>R</em><b>&gt;</b></code>


Generate dimension information. This is needed to embox array entities.


Example:

```mlir
    %c1 = constant 1 : i32
    %c10 = constant 10 : i32
    %63 = fir.gendims %c1,%c10,%c1 : (i32,i32,i32) -> !fir.dims<1>
```

#### `fir.insert_value`

Syntax:	<code><b>fir.insert_value</b> <em>entity</em> <b>,</b> <em>value</em> <b>,</b> <em>index-field-list</em> <b>: (</b> <em>entity-type</em> <b>,</b> <em>value-type</em> <b>,</b> <em>index-field-type-list</em> <b>) -&gt;</b> <em>entity-type</em></code>

Insert a value into an entity with a type composed arrays and/or derived
types. Returns a new value of the same type as _entity_.

Example:

```mlir
    %64 = fir.field_index("field") : !fir.field
    %65 = fir.call @foo2() : () -> i32
    %66 = fir.call @foo3() : () -> !fir.type<X{field:i32}>
    %67 = fir.insert_value(%66, %65, %64) : (!fir.type<X{field:i32}>, i32, !fir.field) -> !fir.type<X{field:i32}>
```

The above is a possible translation of the following Fortran code sequence.

```Fortran
    temp1 = foo2()
    temp2 = foo3()
    temp2%field = temp1
```


#### `fir.len_param_index`

Syntax:	<code><b>fir.len_param_index ("</b><em>len-param-name</em><b>") : !fir.field</b></code>

Compute the LEN type parameter offset of a particular named parameter in a
derived type.

Example:

```mlir
    %62 = fir.len_param_index("param_1") : !fir.field
```

### Generalized Control Flow Ops


#### `fir.loop`

Syntax:
<pre><code><b>fir.loop</b> <em>ssa-id</em> <b>=</b> <em>lower-bound</em> <b>to</b> <em>upper-bound [</em> <b>step</b> <em>step-value ] [</em> <b>unordered</b> <em>]</em> <b>{</b><br>
&nbsp;&nbsp;&nbsp; <em>op-list</em><br>
<b>}</b>
</code></pre>

Generalized high-level looping construct. This operation is similar to
MLIR's affine.for but does not have the restriction that the loop be
affine.

Example:

```mlir
    %72 = fir.load %A : !fir.ref<!fir.type<"R"{fld:!fir.array<?:f32>}>>
    fir.loop %i = 1 to 10 unordered {
      %73 = fir.extract_element %72, %field, %i : (!fir.ref<!fir.type<"R"{fld:!fir.array<?:f32>}>>, !fir.field, i32) -> f32
      %74 = fir.call @compute(%73) : (f32) -> i32
      %75 = fir.coordinate_of %B, %74 : (!fir.ref<!fir.array<?:f32>>, i32) -> !fir.ref<f32>
      fir.store %73 to %75 : !fir.ref<f32>
    }
```

The above fir.loop is a possible translation for the following Fortran DO
CONCURRENT loop.


```Fortran
    DO CONCURRENT (i = 1:10) LOCAL(x)
       x = A%fld(i)
       B(compute(x)) = x
    END DO
```


#### `fir.where`

Syntax:
<pre><code><b>fir.where</b> <em>condition</em> <b>{</b><br>
&nbsp;&nbsp;&nbsp; <em>op-list</em><br>
<b>}</b> <em>[</em> <b>otherwise {</b><br>
&nbsp;&nbsp;&nbsp; <em>op-list</em><br>
<b>}</b> <em>]</em>
</code></pre>

To conditionally execute operations (typically) within the body of a
`fir.loop` operation. This operation is similar to `affine.if`, but it is
generalized and not restricted to affine loop nests.

Example:

```mlir
    %78 = fir.icall %75(%74) : !fir.ref<!T>
    fir.where %56 {
      fir.store %76 to %78 : !fir.ref<!T>
    } otherwise {
      fir.store %77 to %78 : !fir.ref<!T>
    }
```

#### `fir.call`

Syntax:	<code><b>fir.call</b> <em>callee</em> <b>(</b> <em>arg-list</em> <b>) :</b> <em>func-type</em></code>

Call the specified function.

Example:

```mlir
    %90 = fir.call @function(%arg1, %arg2) : (!fir.ref<f32>, !fir.ref<f32>) -> f32
```

#### `fir.icall`

Syntax:	<code><b>fir.icall</b> <em>callee</em> <b>(</b> <em>arg-list</em> <b>) :</b> <em>func-type</em></code>

Call the specified function reference.

Example:

```mlir
    %89 = fir.icall %funcref(%arg0) : (!fir.ref<f32>) -> f32
```

#### `fir.dispatch`

Syntax:	<code><b>fir.dispatch</b> <em>method-id</em> <b>(</b> <em>arg-list</em> <b>) :</b> <em>func-type</em></code>


Perform a dynamic dispatch on the method name via the dispatch table
associated with the first argument.


Example:

```mlir
    %91 = fir.dispatch "methodA"(%89, %90) : (!fir.box<!fir.type<T>>, !fir.ref<f32>) -> i32
```

### Complex Ops

The standard dialect does not have primitive operations for complex types.
We've added these primitives in the FIR dialect.

#### `fir.addc`
#### `fir.subc`
#### `fir.mulc`
#### `fir.divc`

### Other Ops

#### `fir.address_of`

Syntax:	<code><b>fir.address_of (@</b><em>symbol</em><b>) :</b> <em>T</em></code>

Converts a symbol to an SSA-value.

Example:

```mlir
    %func = fir.address_of(@func) : !fir.ref<(!fir.ref<i32>) -> ()>
```

#### `fir.convert`

Syntax:	<code><b>fir.convert</b> <em>ssa-value</em> <b> : (</b> <em>T</em> <b>) -&gt;</b> <em>U</em></code>


Generalized type conversion. Convert the _ssa-value_ from type _T_ to type
_U_.  Conversions between some types may not be defined.  When _T_ and _U_
are the same type, this instruction is a NOP.


Example:

```mlir
    %92 = fir.call @foo() : () -> i64
    %93 = fir.convert %92 : (i64) -> i32
```

The above conversion truncates a 64-bit integer value to 32-bits.


#### `fir.gentypedesc`

Syntax:	<code><b>fir.gentypedesc</b> <em>T</em> <b>: !fir.tdesc&lt;</b><em>T</em><b>&gt;</b></code>

Generate a type descriptor for the type _T_.  This may be useful for
generating type discriminating code. A type descriptor is an opaque
singleton constant value in FIR. (It is assumed to be COMDAT.)


Example:

```mlir
    !T = type !fir.type<T{...}>
    %97 = fir.gentypedesc !T : !fir.tdesc<!T>
```

#### `fir.no_reassoc`

Syntax:	<code><b>fir.no_reassoc</b> <em>ssa-value</em> <b>:</b> <em>T</em></code>

Primitive operation meant to intrusively prevent operator reassociation.
The operation is otherwise a nop and the value returned is the same as the
argument.


Example:

```mlir
    %98 = mulf %96,%97 : f32
    %99 = fir.no_reassoc %98 : f32
    %100 = addf %99,%95 : f32
```

The presence of this operation prevents any local optimizations. In the
above example, this would prevent replacing the multiply and add with an
FMA operation.

## Module Abstractions

#### `fir.global`

Syntax:
<pre><code><b>fir.global @</b><em>global-name [</em> <b>constant</b> <em>]</em> <b>:</b> <em>type</em> <b>{</b><br>
&nbsp;&nbsp;&nbsp; <em>initializer-list</em><br>
<b>}</b>
</code></pre>

A global variable or constant with initial values.

Example:

```mlir
    fir.global @_QV_Mquark_Vvarble : !VarType {
      constant 1 : i32
      constant @some_func : (i32) -> !fir.logical<1>
    }
```

The example creates a global variable (writable) named
`@_QV_Mquark_Vvarble` with some initial values. The initializer should
conform to the variable's type.

#### `fir.global_entry`

Syntax:	<code><b>fir.global_entry</b> <em>field-id</em> <b>,</b> <em>constant</em></code>

A global entry is a mapping in a global variable that binds a field-id to a
constant value.  This allows one to specify the values composed in a
product type and simultaneously defer layout decisions.

Example:

    To do.

#### `fir.dispatch_table`


Syntax:
<pre><code><b>fir.dispatch_table @</b><em>table-name</em> <b>{</b><br>
&nbsp;&nbsp;&nbsp; <em>dt-entry-list</em><br>
<b>}</b>
</code></pre>

A dispatch lookup table used implicitly by a fir.dispatch operation.

Example:

    See below.


#### `fir.dt_entry`

Syntax:	<code><b>fir.dt_entry</b> "<em>method-id</em>" <b>,</b> <em>callee</em></code>


A dispatch table entry is a mapping in a dispatch table that binds a
method-id to a callee-reference.


Example:

```mlir
    fir.dispatch_table @_QDTMquuzTfoo {
      fir.dt_entry "method1", @_QFNMquuzTfooPmethod1AfooR
      fir.dt_entry "method2", @_QFNMquuzTfooPmethod2AfooII
    }
```

