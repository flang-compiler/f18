The semantic pass will determine whether the input program is a legal Fortran
program.

If the program is not legal, the results of the semantic pass will be a list of
errors associated with the program.

If the program is legal, the semantic pass will produce an unambiguous parse
tree with additional information that is useful for the tools API and creation
of the DST.

What is required of semantics?
* Error checking
* A non-ambiguous parse tree
* Symbol tables with scope information
* Name & operator resolution

What do we want from semantics?
* Cache information about labels and references to labels
* Cache information derived from static expression evaluation

What don’t we want from semantics?
* Semantics will not display error messages directly.  Instead, error messages
and their associated source locations will be saved and returned to the caller.
* The parse tree will not be modified except to resolve ambiguity and resolve
names, operators, and labels.

Semantic checking does not need to preserve information that is easily
recomputed, such as pointers to enclosing structures.

The parse tree shall be immutable after resolution of names, operators, labels
and ambiguous sub-trees.  This means that the parse tree does not have direct
references to error messages, etc.

Much of the work that is to be performed by semantic analysis has been specified
in the Fortran standard with numbered constraints.  The structure of the code in
the semantic analyzer should correspond to the structure of the Fortran standard
as closely as possible so that one can refer to the Standard easily from the
code, and so that we can audit the code for missing checks.

The code that generates LLVM will be able to be implemented with assertions
rather than with user error message generation; in other words, semantic
analysis will detect and report all errors. Note that informational and warning
messages may be generated after semantic analysis.

Analyses and data structures that can be deferred to the deep structure should
be so, with exceptions for cases where completing an analysis is just a little
more complex than completing a correctness check (e.g. EQUIVALENCE overlays).


## Symbol resolution and scope assignment
The section describes the when scopes are created and how symbols are resolved.
It is a step-by-step process.  Each step is envisioned as a separate pass over
the tree.  The sub-bullets under each step will happen roughly in the order
specified.

There is a special predefined scope for intrinsics.  This scope is an ancestor
of all other scopes.

More detail is needed about this predefined scope. Who populates this special
intrinsic scope? Does it need to be constructed and populated for each
compilation unit? Maybe it could be a single distinct immutable scope from which
names can be associated, rather than an ancestor.

The following steps will be followed each program unit:

_N.B. Modules are not yet covered_

_N.B. We need to define the semantics of the LOC intrinsic_

#### Step 1. Process the top-level declaration, e.g. a subroutine
1. Create a new scope
1. Add the name of the program unit to the scope except for functions without
result clause
1. Add the result variable to the scope
1. Add the names of the dummy arguments to the scope

Implementation note:  When a program make an illegal forward reference, we
should emit at least a warning so that programs that are illegally assuming host
association for a name won’t be silently invalidated; preferably with a message
that references both instances.

#### Step 2.  Process the specification part
1. Set up implicit rules
1. Process imports, uses, and host association
1. Add the names of the internal and module procedures
1. Process declaration constructs in a single pass
1. Apply implicit rules to undefined locals, dummy arguments and the function
result
1. Create new scopes for derived type, structure, union

Host association logically happens at step 2; perhaps host association can
be deferred until the symbol is referenced?

At this point, all names in the specification part of the parse tree reference
a symbol.

We  can process declaration constructs in a single pass because:
- It is not legal to reference an internal procedure.
- It is not  legal to reference not-yet-defined parameters, constants, etc.
- It is not possible to inquire about a type parameter or array bound for an
object that is not yet defined
- So, no other forward definitions, so yes, we can do in a single pass

Do we ever need to apply implicit rules in the specification section?
1. `integer(kind = kind(x)) :: y ! does implicit rule apply to ‘x’`?
1. `integer, parameter :: z = rank(x) ! use implicit rule to get ‘0’`?

What if (1) and (2) are legal & x’s type is subsequently declared?

#### Step 3. Resolve statement functions vs array assignments
1. Rewrite and move array assignments to execution part
1. Why rewrite?  Because array assignment needs processing in Step 4
1. Statement functions need scopes for the dummy arguments

N.B. As soon as a statement function definition is determined to actually be a
misrecognized assignment to an array element, all of the statement definitions
that follow it in the same specification-part must also be converted into array
element assignments, even if that would lead to an error.

#### Step 4. Resolve symbols in the execution part
1. Look up the name
  - If it exists in a scope, update the name to reference the symbol
  - If it does not exist,
    * Apply the implicit rules
    * Add the name to the scope
    * Update the name to reference the new symbol
  - Introduce new scopes for
    * Select Type type guard statements
    * Select Rank case statements
    * Associate construct
    * Block construct
      - Block has a specification part
      - Blocks start Step 1..4 again
      - N.B. Implicits are applied to the host scope
    * Implied Do
    * Index names in Forall and Do Concurrent
    * Change Team
    * OpenMP and OpenACC constructs
    * ENTRY

References to derived types members are not resolved until semantics

No semantic checking or resolving of types (except for implicit declarations)
has happened yet.

#### Step 5. Perform Step 1..4 on each internal procedure
- Side effect is that each internal procedure gets a proper interface in the
parent scope
- We do this now because we need to know the return and argument types for
functions, e.g. `a = f(a, b, c) % x + 1`

#### Step 6. Tree Disambiguation

At this point, or during Step 3 (TBD), the tree can be rewritten to be
unambiguous.
- Structure vs operator a.b.c.d
- Array references vs function calls
- Statement functions vs array assignment (In Step 3)
- READ/WRITE stmts where the arguments do not have keywords
  - WRITE (6, X)  ….
  - That X might be a namelist group or an internal variable
  - Need to know the names of the namelist groups to disambiguate it
- Others….? TBD

Resolution of parse tree ambiguity (statement function definition, function vs.
  array)

#### Step 7. Do enough semantic processing to generate .mod files
- Fully resolve derived types
- Combine and check declarations of all entities within a given scope; resolve
their type, rank, shape, and other attributes.
- Constant evaluation is required at this point.

Why do Step 7 before the rest of semantic checking? The sooner we can generate
mod file the sooner we can read ‘em; you can test a lot of Fortran programs as
soon as you can read mod files.

#### Step 8. Semantic Rule Checking

An incomplete and unordered list of requirements for semantic analysis:

* EQUIVALENCE overlaying (checking at least)
* Intrinsic function generic->specific resolution, constraint checking, T/R/S.
* Compile-time evaluation of constant expressions, including intrinsic
functions.
* Resolution of generics and type-bound procedures.
* Identifying and recording uplevel references.
* Control flow constraint checking
* Labeled DO loop terminal statement expansion? (maybe not, can defer to CFG in
  DST).
* Construct association: distinguish pointer-like from allocatable-like
* OMP and OACC checking
* CUF constraint checking

## Utility Routines

### Diagnostic Output
TBD

## Semantic analysis of Expressions

### Basic scheme

Resolving the semantic of expressions requires to combine information from 
different sources:

* The current node provides a syntactic description of the operation
  (e.g. addition, multiplication, function call, ...).
* The children nodes provide the arguments of the operation.
* The parent nodes provide the context (e.g. a constant expression, 
  specification expression, actual argument requirering a reference, ...)

The order in which expressions will be processed will depend of the 
actual operation. As a rule of thumb, the semantic of an operation 
shall be resolved using the following steps
* For each child expression, resolve its execution context and perform 
  the semantic analysis (recursive)   
* Collect the types and other information computed during the semantic 
  analysis of the children expressions and figure out the semantic and 
  the resulting type of the current expression.
* Check that the computed semantic is coherent with the context (e.g. 
  apply restrictions on what is possible in constant expressions)
* Rewrite the parse-tree according to the semantic (where needed)

That simplified scheme is unfortunately not always applicable as 
described previously. For some operation, the basic scheme remain 
valid but the order of the various analysis may have to be adjusted. 

### Function calls and overloadable operations

Functions calls are one of the most common operations in Fortran 
expressions especially because most operators can be overloaded. 
Simply speaking, an operation such as a+b is nothing else than 
a disguised function call. This is true even if the operation 
resolves to one of the intrinsic cases for the '+' operation. 

In practice, the operations that do not fall in that category 
are quite limited:
* Reference to a symbol
* Mmember operator '%'
* Array subscripts
* Substrings 
* Parenthesis operator
* Array constructor operator (/ a,b,c... /)
* Implied do operator 
* IS THERE MORE ???? 

For function calls and other overloadable operations, the 
semantic analysis will proceed as follow:

#### Step 1 Resolution of the call target  

For an overloaded operator, that phase is straightforward since 
each operator is associated to a unique generic symbol. This 
is the same symbol that shall be 'extended' each time an 
INTERFACE OPERATOR construct is found. A simple lookup for 
in the current scope will provide that symbol which will be 
used as the call target.

For explicit calls, so expressions of the form 'target(args,...)' the target
is provided by the expression on the left of the opening parenthesis. 

The target of the FunctionReference is provided as a ProcedureDesignator that
can take multiple forms in the parse-tree.

The easiest case is a ProcedureDesignator containing a single Name that
should be resolved by the current scope to a callable symbol (i.e. a generic
or specific function or an entity with the EXTERNAL or PROCEDURE attribute).
If the symbol represents an non-callable entity with DIMENSION attribute then 
the function call needs to be transformed into an array subscript (see section 
below).

More complex cases are when the target is provided by a pointer or as
structure member. 

#### Step 2 Semantic analysis of all actual arguments

After that step, the type of each actual argument shall be known.  It shall
also be possible to differentiate actual arguments that are references (and so
can be used as Variables) and those that are variables with either the POINTER
or ALLOCATABLE attribute. 

#### Step 3 Resolution of the specific procedure symbol.

If the target is a generic procedure symbol, the types and names of all actual
arguments must be compared to the dummy arguments of all associated specific
procedure symbols. In case of success, the target becomes a specific procedure
symbol.

If the target was already a specific procedure symbol or a procedure pointer 
then the types of the actual arguments must be matched against the dummy arguments. 

#### Step 4

At that point, each actual argument should be associated to a dummy argument 
of the target procedure. So far, the matching was performed by only considering 
the type of the actual and dummy arguments.

Other attributes such as INTENT, POINTER, ALLOCATABLE, .... must now be
compared and, in some cases, the actual argument may have to be rewritten to
reflect the nature of the dummy argument (see the section "Disambiguation of
actual arguments" for more details).

#### Step 5

Assuming that all previous steps where successful, the result type of the call 
can be computed.


### Intrinsic function calls

Intrinsic functions can be classified in two groups: those that behave like 
regular functions and those that do not. 

The first group is easy to handle: The system scope shall provide an interface 
for each specific implementation of the function. For a some intrinsics that 
could represent a significant amount of specific symbols but the only technical 
difficult is to figure out a method to generate hundreds or thousands of 
function prototypes without having to write them manually.   

The second group of instrinsic function is more problematic. One of the
difficulties is that Fortran does not prevent users from overloading intrinsic
functions which means that whatever hand-written method to identify non-standard 
calls shall also consider the user-defined interfaces for the same name. 

It is not possible to define a generic method to handle all intrinsic 
functions calls.
 
### Disambiguation of Function Calls vs Array Subscripts

From a purely syntactic point of view, functions calls and array subscripts
can be indistinguishable in Fortran:

    x = abc(i,j)   ! Could be a call to function abc or an access to array abc 

Cases that should already be disambiguated by the compiler are:
* calls without argument (i.e. an array shall have at least one dimension):
    x = abc()
* calls with at least one named argument:
    x = abc(i,foo=j) 
* array subscripts with one or more indices being a range:
    x = abc(1:i,j)

Cases that cannot be disambiguated by the parser are assumed to be function calls.

Consequently the procedure to disambiguite a function call is: 

* resolve the type of its target (i.e. abc in the previous example). In most
  cases, that implies performing a semantic analysis on the target.
* if the argument list is empty or if any argument is this has to be a call 
  so abort the disambiguation.     
* if the target is a proper call target then this is a call so abort the 
  disambiguation. 
* if the target is a local entity without DIMENSION then try to give it 
  the EXTERNAL attribute. Abort the disambiguation   
* if the target does not have DIMENSION then emit an error message since 
  the current operation can be neither of a function call or an array subscript. 
* Finally, for an entity with the DIMENSION attribute, transform the 
  function call into a proper array subscript.
  
### Disambiguation of pointer and allocatable references

When dealing with pointers, the Fortran language inserts implicit dereferences
in all places where this is needed. Consequently the action of dereferencing a
pointer is not explicit in the parse-tree. The proposal is to introduce new
nodes references to the pointer from references to the pointed data.

The exact form of those two new nodes is still to unknown. That could take the
form of: 

* a new node PointerDeref to transform a pointer reference into a data
  reference to the pointed data.
* a new node PointerReference that would be used in replacement of
  DataReference for all references with a POINTER type.
  
Consider for instance the expression a%b%c%d. The parse-tree produced by the parser
initially looks as follow:
    
    | Expr -> Designator 
    | | DataReference -> StructureComponent
    | | | DataReference -> StructureComponent
    | | | | DataReference -> StructureComponent
    | | | | | DataReference -> Name = "a"
    | | | | | Name = "b"
    | | | | Name = "c"
    | | | Name = "d"

If we assume that c is a member with the POINTER attribute then the tree shall
become:

    | Expr -> Designator 
    | | DataReference -> StructureComponent
    | | + DataReference -> PointerDeref
    | | | + PointerReference -> StructureComponent
    | | | | + DataReference -> StructureComponent
    | | | | | + DataReference -> Name = "a"
    | | | | | + Name = "b"
    | | | | + Name = "c"
    | | + Name = "d"
    
Let's now consider a call that takes a pointer variable x as actual argument:

    call proc(x) 

If the dummy argument also has the POINTER attribute then the ActualArg tree 
shall be disambiguated as follow (the PONINTER is passed by reference):

 Variable -> Designator -> PointerReference -> Name 

If the dummy argument has the VALUE attribute then the ActualArg tree 
shall be disambiguated as follow:

   Expr -> Designator -> DataReference -> PointerDeref -> PointerReference -> Name 

If the dummy argument has neither the POINTER or the VALUE attribute then 
the ActualArg tree shall be disambiguated as follow:
 
   Variable -> Designator -> DataReference -> PointerDeref -> PointerReference -> Name


### Disambiguation of actual arguments

In the parse tree, ActualArg, so the representation of an actual arguments within a
function or subroutine call, is a wrapper for the following variants:
* Indirection<Expr>
* Indirection<Variable>
* Name
* ProcComponentRef
* AltReturnSpec
* PercentRef
* PercentVal

For the sake of simplificity, let's ignore the last 3 cases that represent
features that are either obsolete or coming from a language extension.

The cases Name and ProcComponentRef are a direct consequence of the parser 
rule for an actual argument. The Names is expected to be a procedure name 
but the parser will never actually generate that case (because a 
standalone name can also be matched as a Variable (or as an Expr). From a 
semantic point of view, matching a procedure name as a Variable is however 
incorrect. Also, for dummy arguments with the VALUE attribute, a Variable 
shall be represented by an Expr (i.e. data-reference vs value)

The proposal is:
* Modify ActualArg by replacing the Name and ProcComponentRef variant by a 
single ProcedureDesignator (or Indirection<ProcedureDesignator>).  
* For each ActualArg, perform a disambiguation between Expr, Variable and 
ProcedureDesignator

That disambiguation will have to be performed relatively late because it
requires detailed knowledge about the semantic of the call:
* arguments with a VALUE attribute must be Expr
* arguments with an INTENT(INOUT) or INTENT(OUT) attribute must be Variable
* And finally, arguments with an INTENT(IN) or no explicit intent can be 
  either Variable, Expr or ProcedureDesignator depending of the content of 
  the actual argument. 


### Constant expressions as actual arguments of intrinsic calls 

The Fortran specification clearly describes what is allowed or disallowed in
constant context. There are several rules to apply but none of them is problematic.

However, a few intrinsic functions require one of their arguments to be constant. A
typical example is the argument 'kind' found in most conversion intrinsics. It
has to be constant because its value is used to define the return type of the call.

Those 'kind' arguments are a bit problematic because they cannot be identified
before resolving the generic procedure symbol into a specific symbol and in
order to do that, semantic analysis must be applied to the arguments (to
resolve their type). This is a typical chicken-and-egg problem: The semantic
analysis of the 'kind' argument should be performed using the 'constant
expression' rules but we cannot know that the expression is constant before 
performing its semantic analysis. 

A possible solution to that problem could be to perform the semantic analysis 
twice. That could work but this is a risky approach since semantic analysis 
can have a lot of side effect (warnings, code transformations, ...) 

A better approach is probably to perform the semantic analysis of the argument
as if it was not in constant context (which we don't really know at the
begining) and later to attempt to evaluate its value at compile time. If the 
evaluation fails to return a value then an error message shall be emited. 

The resulting behavior shall be a superset of what should be legally valid 
according to a strict interpretation of the standard. 

### Actual arguments of operations 

Most of the Fortran operators can be overloaded which makes them behave as 
function calls. Consider for instance the following piece of code:

    interface operator(.NOT.)
      import
      logical function not_foo(a)
         type(Foo), intent(in) :: a
      end function not_foo
    end interface 

    type(foo) :: x 
    print *, not_foo(x) , .NOT. x 
  
The expressions 'not_foo(x)' and '.not. x' are semantically equivalent but 
they are described in inconsistent ways in the parse-tree.

The expression "not_foo(x)" is an explicit function call and since the dummy argument
corresponding to 'x' has INTENT(in), the ActualArg is a Variable. That
makes sense since 'x' is expected to be passed by reference.

Th expression ".NOT. x" is a unary operation so represented by a Expr::NOT
node in the parse tree. Consequently, "x" can only be represented by an Expr
node that itself contains a Designator that provides a reference to the Name "x".

From a semantic point of view, using an Expr to represent "x" is incorrect. The 
intent here is really to use a Variable. 

A possible solution to that problem could be to insert an ActualArg in the
parse-tree for each operation argument. This is not completely satisfactory
for two reasons: first ActualArg is directly related to a rule in the grammar
and second, ActualArg is a superset of the what we want to be able to represent.

A better approach could be to introduce a new node OperatorArg representing a
variant of an Expr and a Variable. The parser would systematically produce an
Expr (because this is how the Fortran grammar is written) and the semantic analysis  
would be in charge of transforming the argument into a Variable where needed. 

## Evaluation of Constant Expressions

Being able to evaluate constant expressions is an early requirement in a
Fortran compiler because most declaration statements require the evaluation of
at least one expression describing a kind, len or shape bound.

A proper implementation is not a trivial task because Fortran constant
expressions can contain array operations and are allowed to call a large 
number of intrinsic functions. 

The goal the first implementation will be to provide the constant evalation 
of scalar expressions that are commonly found in type declarations (in kind, 
len and shape expressions).

The priority will be given given to the following features: 
* real, integer, and logical literals.
* real, integer, and logical scalar parameters. 
* the common arithmetic operations: +, *, /, -
* the common comparisons operators for integers and real values. 
* the common logical operations: .NOT. .AND. .OR. etc
* the following intrisic functions (on scalar values only)
  - kind()
  - max() and min()
  - abs()
  - selected_int_kind() and selected_real_kind()






