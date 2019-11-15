# Array Expression Lowering
This document describes how Fortran array expressions will be lowered from their `Fortran::evaluate::Expr` tree representation to a sequence of FIR operations.
Some higher-level design regarding array expression evaluation is presented in [Array Expression Evaluation Design Document](documentation/ArrayComposition.md).
The objectives of array expression lowering to FIR are to produce a sequence of FIR operations that:
1.  Implements Fortran Semantics and f18 semantic extensions
2.  Exposes enough structure so that array optimizations can later be performed on FIR
3.  Either implements, or makes it possible to implement as FIR transformations the expressions rewriting exposed in [Array Expression Evaluation Design Document](documentation/ArrayComposition.md).
## Discussions about the objectives
### Implementing Fortran and f18 extension semantics 
Rules to evaluate expressions are listed in Fortran 2018 section 10. For what matters to lowering to FIR, one of the most important part is section 10.1.4 regarding expression evaluation and 10.1.7 regarding evaluation of operands.
One of the take-away is that Fortran standard gives a lot of freedom to the compiler regarding the order of evaluations of operands and whether they are evaluated or not. The compiler is also allowed to perform mathematically equivalent rewriting of expressions (see 10.1.5.2.4 point 3.).
A few rules to restrain this freedom during expression lowering to FIR and keep the roles of different compilation phases clear are listed below.
#### Preserving numerical properties
##### Rule The lowering of expression to FIR will not perform any expression rewriting while lowering that could lead to different numerical results than the one specified by the semantic context.
##### Rationale
Numerical stability is important for many Fortran coders and some others valuate performance more. Options will be required to control numerical stability requirements during compilation. Starting to do a lowering that may change numerical properties will lead to augmenting the number of possible paths in expression lowering to respect these numerical options. Instead, these transformations should be perform directly on `evaluate::Expr` or later on the FIR operations.
##### Example
`x + y + z` is not numerically equivalent to `z + x + y`, although it is mathematically.  Lowering to FIR shall not modify the order of the additions in such case.
##### Plans to enforce the rule while lowering to FIR

This mainly translate in preserving the `evaluate::Expr` tree structure in the FIR operation sequence structure. The operands order should be respected as well as the dependencies. In case the lowering of an expression does not introduce the need to store some sub expressions, the dependency tree of the SSA value holding the result value should match the `evaluate::Expr` tree.
Concretely, because there is not a one to one match between FIR operations and all Fortran operations and intrinsic procedures. Some rewriting must be done while lowering expressions.
For instance REAL MAX intrinsics might be lowered to `fir.fcmp` comparison and a MLIR `std.select` operation. In such case, the lowering should ensure that choices and combinations of FIR operations match with the floating-point behavior that is guaranteed for `MAX` given the context (that may include where it is in a program and some compile time options).
Numerical requirements will need to be accessible in such cases so that lowering can select the right operations and/or propagate the requirements as flags on the operations.

#### Scalar operands evaluation in elemental expressions
##### Rule
Scalar operands in elemental expressions must be evaluated only once.
##### Rational
10.1.10 point 2. says a scalar in an elemental expression must be treated as an array with _all elements equal_ to the scalar.
This means that if the compiler cannot prove that re-evaluating such operands will yield the same result, it must be evaluated only once while evaluating the elemental expression. In other cases, a lowering that re-evaluate the scalar sub-expression for every element would be standard compliant. 
However, in general, skipping useless evaluation should be a good thing regarding performance. So, rather than to do an analysis regarding whether or scalar operand must not be re-evaluated, it is easier and performance friendly to request that they should be evaluated only once.
##### Example
In `array + foo(x)` where `foo(x)` is scalar, foo will only be called once.
##### Plans to enforce the rule while lowering to FIR
The evaluation of scalar operands will be inserted before the loop evaluating the elemental expression. In the example above, the generated FIR should look like:
```
%1 = fir.call foo %x
fir.loop %i 1 to n {
  %2 = fir.extract_value %array %i 
  %3 = fir.fadd  %2 %1
  %4 = fir.coordinate_of %result %i
  fir.store %3 to %4
} 
```
#### Short-circuiting
##### Rule
Expression lowering to FIR must insert short-circuiting of operand evaluation that are guaranteed to not be performed in contexts defined in by f18 semantic extensions. Other short-circuiting opportunities should be left to later optimization phases.
##### Rational
If it needs to be guaranteed something will not be evaluated if unneeded, then that is semantic related, and it is unsafe to assume later FIR transformation and processing will respect this if short-circuiting was not explicitly inserted while lowering the expression.
However, in general, the decision of whether to short-circuit an evaluation or not should be left to the optimizers based on the cost of the useless operation compared to the cost of the potential branch. Therefore, expression lowering should not add short-circuits that are not required because these cannot always be reversed in FIR (unless FIR transformations take the liberty to violate semantics which they probably should not).
Note that it is unclear how well FIR transformation will be able to introduce short-circuits that are rather obvious looking at the `evaluate::Expr` tree, so this rule might get revisited.
So far, the required short-circuits are:
- `b.OR.SE`, `b.AND.SE` -> do not evaluate `SE` when `b` is false and `SE` has side effects.
- `MERGE(SE1, SE2, b)` -> when `b` is a scalar, either `SE1` or `SE2` evaluation has side effects.
Side effects should be understood in the broad sense (array accesses can cause segfaults, floating point operations can raise signals).
##### Example
In `b.AND.foo()` the call to `foo` shall not be performed if `b` is false.
##### Plans to enforce the rule while lowering to FIR
It is possible to use `fir.where` to implement a simple if/else logic without having to build a control flow graph using `br_cond` and basic blocks. Although both of these implementation are equivalent, `fir.where` gives more structural guarantees than basic block logic and is expected to interact in better way with optimizations. It is also more readable.
However, with a `fir.where`, the result of such expression will need to be materialized (with basic blocks logic, one could have taken advantages of phi nodes scalar values, but FIR/LLVM transformation should anyway get rid of scalar temporary in such cases).
The fir for the example above will look like:
```
%1 = load %b
fir.where %1 {
// b true case  
  %1 = fir.call foo
   fir.store %1 tmp 
} otherwise {
  %2 = constant 0
  fir.store %2 tmp
}
```
### Exposing enough structure so that array optimizations can be performed on FIR
Because there are no FIR array optimizations yet, it is not entirely defined what this requirement implies yet. However, one of the objectives of MLIR is to be able to represent array layouts and access patterns as affine expressions. These affine expressions can then be composed allowing generalized loop and data transformations.
Of course, this only make sense for array expressions that have such affine properties, mainly elemental and transformational operations. There is not much one can do to transform opaque function calls involving arrays into some affine operation.
#### FIR operations to express array operations
In FIR, affine expressions can be implemented using the `fir.loop`, `fir.where`, `fir.extract_value` and `fir.coordinate_of` operations in combinations with the `fir.array<>` type and the `fir.box<>` type that can hold affine maps encoding linear access patterns (see [FIR  Language reference](documentation/FIRLangRef.md)).
#### Views over array in FIR
In FIR, one can avoid copying array sections by creating `fir.box` with that holds the memory reference to the base variable as well as the subscript describing the re-mapping.
For instance, `A(10:20:2, :)` where `A` is `REAL, DIMENSION(20, 10)` can be written in FIR as:
```
//%a is the memory reference to A storage, it is of type !fir.ref<fir.array<20:5:fir.real<4>>>
%0 = fir.gendims 2*m+1, 4*m, 2
%1 = fir.gendims 1, n, 1
%asection = fir.embox %a, %0, %1
// %asection has type !fir.box<fir.ref<fir.array<5:10:fir.real<4 >>>>
```
Then the element (i, j) of `A(10:20:2, :)`  can be accessed with:
```
// fir.extract_value takes care of applying the index mapping of the array section over A.
%asectionij = fir.extract_value %asection, %i, %j
// % asectionij has type !fir.real<4>
```  
No copy into a temporary was needed to implement the example above. `fir.box` allows one to represent array and manipulate subsections in FIR without materializing them. It can be seen as if we had created a Fortran pointer over `A` (except such object would have had the type `%fir.box<fir.ptr< fir.array<5:10:fir.real<4 >>>`, notice the `fir.ptr` instead of `fir.ref`).

One notable difference between using `fir.box` and materializing a section inside a temporary is that its value its not “fixed”, the evaluation of the section is delayed until it is needed. This introduce the risk that `A` variable is modified in-between the array emboxing and its usage. In the context of expression evaluation, the only thing that could modify another variable appearing inside the expression evaluation are function call (potentially hidden in user defined operators).
Fortran Standard prohibit this by stating in section 10.1.4 point 2
> [..] Except in those cases: the evaluation of a function reference shall neither affect nor be affected by the evaluation of any other entity within the statement

The cases mentioned do not apply inside an expression. However, it is in general unsafe to propagate such box outside of an expression evaluation. This means that if `A(10:20:2, :)` if a top level expression, it is probably safer to materialize into a temporary storage and leave later optimizations determine if it is safe to keep skip the materialization.
This means for the Fortan code:
```
B = A(1:10, :)
PRINT *m, B
```
The bridge to FIR will not replace refences to `B` by the `fir.box` encoding ` A(1:10, :)` because it cannot ensure `A` was not modified in-between without an analysis of A potential definition in-between that does not belong to the bridge towards FIR.

An alternative, that requires FIR language modifications, would be to allow `fir.box` to embox array SSA values (`fir.array<T>`) instead of only memory references (like `fir.ref<fir.array<T>>` ).  The value-based aspect would remove such need of performing re-definition analysis to propagate boxes. 
#### Lowering transformational intrinsic procedures to `fir.box`
`fir.box` can also be used to implement some transformational intrinsic because it can describe more than array subsection in case it is used with an mlir affine map instead of susbscripts.
For instance `transpose(A)` where `A` is `REAL, DIMENSION(20, 10)` can be written in FIR as:
```
#transpose = (i, j)[] -> (j, i)
%atrans= fir.embox %a  [#transpose]
```
Element `(i, j)` form `transpose(A)` would then simply be accessed with `fir.extract_value %atrans, %i, %j`.

Affine maps can be combined and should allow to implement many of the ideas from the [Array Expression Evaluation Design Document](documentation/ArrayComposition.md) in FIR.

TODO: write the affine maps for all the transformational mentioned in last document.

More can be read about MLIR affine maps in the [Affine Dialect Language Reference](https://github.com/tensorflow/mlir/blob/master/g3doc/Dialects/Affine.md)
#### Expression storage in FIR

A `fir.loop` is more or less a regular counted loop and it is not returning an ssa value that would be the result of the array expression associated to the loop. Hence, it can only have effects on the program through memory accesses. `fir.store` operations in the `fir.loop` must store the array result elements into some memory reference.
This implies that every `fir.loop` computing an array expression must be given a memory reference to some storage can store the array result. The required size of such storage may know at compile time or may be dynamic.
Hence, expression lowering will have to handle storage for expressions and may have to introduce dynamic memory allocation and free operations (`fir.allocmem` and `fir.freemem`).
To simplify the design, we will assume the entry point of array expression lowering is provided with a memory-reference that can store the result. Whether this reference is a temporary and whether it is dynamic should not affect the lowering.
However, expression lowering may have to introduce storages to materialize sub-expressions. In such case, expression lowering will be responsible to create the storage and to release them if necessary. 
To simplify, expression lowering will not try to do any kind of clever temporary re-use or pooling. It is expected later transformations should do so based on the dependencies and lifetime of the created temporaries.
Note that this means that in the case of assignment statement `y = x + transpose(x)`, expression lowering is not involved in the decision of using a temporary to evaluate the right hand side before storing into `y` (which would be required if `y` was a pointer aliasing `x`). The assignment statement lowering is free to make such analysis and to provide the right memory storage to expression lowering or to always provide a temporary and hope the useless copy will be optimized out.
Expression lowering should be written independently of the kind of statement in which it is being lowered.
### Relation to [Array Expression Evaluation Design Document](documentation/ArrayComposition.md).
Array Expression Evaluation Design Document](documentation/ArrayComposition.md) presents an evaluation of array expression that intends to minimize the use of materialization of operands as into temporary storage.
An alternative is to implement this document design when lowering expressions (Functional Approach), the other is to still materialize every sub-expressions during expression lowering (Data Approach) and to leave later FIR transformation understand the generated structure and re-group the loops.
Both alternatives are presented more in detail in the next sections. 
## Proposed Lowering to FIR
To illustrate the difference, the following Fortran example will be used:
```
  integer, parameter :: n = 100
  integer, parameter :: m = 50
  real :: R(m, n), B(m, n)
  real :: A(4*m, n), C(n, m)
  real :: x
  real, external :: foo
  R = A(2*m+1:4*m:2,:) * B + foo(x) * TRANSPOSE(C)
```
It is complex enough so that differences can appear more clearly. It also allows to illustrate some of the points made above.
In the related FIR code, `%a`, `%b`, `%x` and `%c` are the MLIR SSA values for the storage of `A`, `B`, `x` and `C` Fortran variables. They are respectively of types `fir.ref<fir.array<4*n:m:fir.real<4>>>`, `fir.ref<fir.array<n:m:fir.real<4>>>`, `fir.ref< fir.real<4>>` and ` fir.ref<fir.array<n:m:fir.real<4>>>`. 
`n` and `m` of compile time constant for the Fortran parameters `n` and `m`.
Expression lowering has been provided with an SSA value `%res` to a memory reference to store the result (it can be a `fir.ref` or a `fir.box that may not be contiguous, it does not matter). It has the correct type and shape to store the expression. 

### Proposed Array Expression Lowering – Functional Approach
The proposed approach is to fully implement [Array Expression Evaluation Design Document](documentation/ArrayComposition.md) while lowering and to try as much as possible to group elemental operations in a single `fir.loop` whose block implement the expression as a function.
The following FIR would be generated for the above example:
```
// No need for temporary storage for this expression
// Create a box over A to implement A(2*m+1:4*m:2,:)
%0 = fir.gendims 2*m+1, 4*m, 2
%1 = fir.gendims 1, n, 1
%asection = fir.embox %a, %0, %1

// Represent TRANSPOSE(C) mapping over C
#transpose = (i, j)[] -> (j, i)
%ct= fir.embox %c [#transpose]

// Compute foo(x)
%foox = fir.call %x

// Finally compute the whole elemental expression in a single loop nest
fir.loop %i = 1 to m {
  fir.loop %j = 1 to n {
    // Compute A(2*m+1:4*m:2,:) * B at the scalar level
    %2 = fir.extract_value %asection %i, %j
    %3 = fir.extract_value %b %i, %j
    %4= fir.fmul %2 %3
    // Compute foo(x) * transpose(C) at the scalar level
    %5 = fir.extract_value %ct %i, %j
    %6= fir.fmul %foox %5
   // Compute the top-level expression node at the scalar level
    %7= fir.fadd %4 %5
   // Store result for element in the given storage fir the whole expression
    %8= fir.coordinate_of %res %i %j
    fir.store %7 to %8
  }
}
```

The advantage of this approach is that the lowering to FIR has not pessimized the array expression from a memory footprint point of view and it has not introduced dependent loops that are trivially mergeable looking at Fortran rules.
The drawback is that it requires a slightly more complex expression lowering approach where one must determine the maximum sub-tree that can be implemented in a loop and then first generate the dependencies of this subtree (which are its scalar branches and branches towards array sub-expressions that are not linear). It then gets the ssa nodes that where generated for this dependencies and generate the `fir.loop`.
This could be implemented as two passes over the expression tree, with the drawback these each of these passes would have strong expectation regarding what the other passes would do. It could also be done in a single pass provided additional state is kept keeping track of where FIR operations should be generated outside and inside the loop. This does not like a state that is too hard to maintain.

When it is determined a sub-expression requires short-circuiting, it is treated as if it were not linear and is computed outside of the `fir.loop` (see example in last section). An alternative could be to try to raise short-circuits at the top level to avoid introducing dependent loops at the cost of duplicating the whole expression evaluation to cover all scenarios (this is also describe in more detail at the end of this document).
### Alternative – Data Approach 
The data-oriented approach materializes every array subexpression (but subsections and transformational that use `fir.box<>`). The FIR generated for the example 
```
// Allocate temporary storages on the stack
%tmp1 = fir.alloca fir.array<n:m:fir.real<4>> 
%tmp2 = fir.alloca fir.array<n:m:fir.real<4 >>

// Create a box over A to implement A(2*m+1:4*m:2,:)
%0 = fir.gendims 2*m+1, 4*m, 2
%1 = fir.gendims 1, n, 1
%asection = fir.embox %a, %0, %1

// Compute A(2*m+1:4*m:2,:) * B into %tmp1
fir.loop %i = 1 to m {
  fir.loop %j = 1 to n {
    %3 = fir.extract_value %b %i, %j
    %4= fir.fmul %2 %3
    %5= fir.coordinate_of %tmp1 %i %j
    fir.store %4 to %5
  }
}

// Compute foo(x)
%6 = fir.call %x

// Represent TRANSPOSE (C) mapping over C
#transpose = (i, j)[] -> (j, i)
%ct= fir.embox %c [#transpose]

// Compute foo(x) * TRANSPOSE(C) into %tmp2
fir.loop %i = 1 to m {
  fir.loop %j = 1 to n {
    %7 = fir.extract_value %ct %i, %j
    %8= fir.fmul %6 %7
    %9= fir.coordinate_of %tmp2 %i %j
    fir.store %8 to %9
  }
}

fir.loop %i = 1 to m {
  fir.loop %j = 1 to n {
    %10 = fir.extract_value %tmp1 %i, %j
    %11 = fir.extract_value %tmp2 %i, %j
    %12= fir.fadd %10 %11
    %13= fir.coordinate_of %res %i %j
    fir.store %12 to %13
  }
}
```
Its main advantage is that it is easy to generate in single bottom-up expression tree visit (generating FIR for an expression node at the post visit). No particular states need to be maintained regarding where the FIR code should be emitted (the `mlir::OpBuilder` is sufficient for that).
Its main drawback is that a lot of temporaries may be created, and it is not entirely sure later optimizations will be able to get rid of them and to merge the loops. Even if it is, it may consume “optimization time and resource” to undo something that could have been done right the first time with only a little more work.

This could be implemented as a single pass over the expression tree where the lowering for each expression node first asks for the lowering of its operands and expect to get a memory reference that materializes it (or an SSA value if the operand is scalar). In case the node being processed is an elemental operation, a `fir.loop` is generated to implement it. If a node is an array section or a call to a transformational, a box is created over the data reference or the result of the transformational argument expression evaluation.

Short-circuits are generated also generated outside of any fir-loop which already is the case of other operands so nothing on top of generating the `fir.where` needs to be done.

## Further Features that Expression Lowering
#### Raising short-circuits to the top of expression evaluation
It is not clear if expression lowering should raise the short-circuits as high as possible inside expression evaluation to diminish the needs of temporary materialization due to short-circuits:
For instance, `A + MERGE(foo(), B, b)` can also be seen as `MERGE(A+foo(), A+B, b)` (`foo` is still guaranteed to not be executed in the second case and the rewrite is numerically equivalent). Because the short-circuit is at the top-level in the second form, the memory reference for the result can directly be used in the `fir.where` providing the short-circuit logic whereas the first form requires a temporary materialization and to loop twice.
With the proposed lowering and assuming `A` and `B` are `REAL, DIMENESION(1:n)` arrays, `A + MERGE(foo(), B, cdt)` translates to: 
```
%tmp = fir.alloca fir.real<4> %n
fir.where % cdt {
  %1 = fir.call foo
  fir.loop %i = 1 to %n {
    %2 = fir.coordinate_of %tmp %i
    fir.store %1 to %2
  }
} otherwise {
  fir.loop %i = 1 to %n {
    %3 = fir.exctract_value %b %i
    %4 = fir.coordinate_of %tmp %i
    fir.store %3 to %4
  }
}

fir.loop %i = 1 to %n {
  %5 = fir.exctract_value %a %i
  %6 = fir.exctract_value %tmp %i
  %7 = fir.addf %5 %6
  %8 = fir.coordinate_of %res %i
  fir.store %7 to %8
}
```
Whereas `MERGE(A + foo(), A+B, cdt)` translates to: 
```
%tmp = fir.alloca fir.real<4> %n
fir.where % cdt {
  %1 = fir.call foo
  fir.loop %i = 1 to %n {
    %2 = fir.exctract_value %a %i
    %3 = fir.addf %2 %1
    %4 = fir.coordinate_of %res %i
    fir.store %3 to %4
  }
} otherwise {
  fir.loop %i = 1 to %n {
    %5 = fir.exctract_value %a %i
    %6 = fir.exctract_value %b %i
    %7 = fir.addf %5 %6
    %8 = fir.coordinate_of %res %i
    fir.store %7 to %8
  }
}
```
The second form is probably better in many cases from a performance point of view but it is not clear if expression lowering should take care of introducing short-circuits in such way, or if it should go for the simpler first form and let later FIR optimization passes transform to the second form.
In the data-structure oriented approach, this does not make much of a difference since everything is materialized anyway.

