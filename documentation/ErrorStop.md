<!--
Copyright (c) 2019, Arm Ltd.  All rights reserved.
-->

# Design proposal for ERROR STOP statement semantic analysis

## Current state

The overview of semantic analysis process is briefly described in the "Overview of Compiler Phases" document held in the F18 source code repository ([https://github.com/flang-compiler/f18/blob/master/documentation/Overview.md](https://github.com/flang-compiler/f18/blob/master/documentation/Overview.md)).

The process is orchestrated by the `Perform()` method in `Semantics` class as defined in `lib/semantics/semantics.cc` file:

~~~
bool Semantics::Perform() {
  return ValidateLabels(context_.messages(), program_) &&
      parser::CanonicalizeDo(program_) &&  // force line break
      ResolveNames(context_, program_) &&
      RewriteParseTree(context_, program_) &&
      StatementSemanticsPass1{context_}.Walk(program_) &&
      StatementSemanticsPass2{context_}.Walk(program_) &&
      ModFileWriter{context_}.WriteAll();
}

~~~

Following steps are performed:

1. Validate labels (see [Semantics: Resolving Labels and Construct Names](https://github.com/flang-compiler/f18/blob/master/documentation/LabelResolution.md))

2. Canonicalize DO statements

3. Resolve names (produce *tree of scopes* populated with symbols and types)

4. Rewrite Parse Tree

5. Statement semantics analysis pass 1 (see below)

6. Statement semantics analysis pass 2 (see below)

7. Run Mod File Writer

Pass 1 of the Statements Semantic Analysis currently analyzes experssions (semantic checks performed on all expressions):

~~~
using StatementSemanticsPass1 = SemanticsVisitor<ExprChecker>;
~~~

Pass 2 of the Statements Semantic Analysis currently analyzes assignments (semantic checks performed on all assignment statements) and other statements and language constructs (e.g. DO CONCURRENT); all those checkers are listed in alphabetic order:

~~~
using StatementSemanticsPass2 =
    SemanticsVisitor</* list of Pass2 checkers ...*/>;
~~~

Each of those checkers is defined as a class inheriting from `BaseChecker`. In order to implement their activities, they can override `Enter()` and/or `Leave()` methods that are no-op (i.e. `{}`) in the `BaseChecker` class. These methods are called for the parse tree nodes of interest, `Ender()` before the children, `Leave()` after.

* `ExprChecker` overrides `Entry()` method for `Expr` and `Variable` parser nodes.
* `AssignmentChecker` overrides `Entry()` method for `AssignmentStmt`, `PointerAssignmentStmt`, `WhereStmt`, `WhereConstruct`, `ForallStmt`, `ForallConstruct` parser nodes.
* `DoConcurrentChecker` overrides `Leave()` method for `DoConstruct` parser nodes.

Both of these passes are instances of `SemanticVisitor` on which `Walk()` method is applied, which in turn calls `parser::Walk()` method:

~~~
template<typename... C> class SemanticsVisitor : public virtual C... {
public:
  using C::Enter...;
  using C::Leave...;
  using BaseChecker::Enter;
  using BaseChecker::Leave;
  SemanticsVisitor(SemanticsContext &context)
    : C{context}..., context_{context} {}
  template<typename N> bool Pre(const N &node) {
    Enter(node);
    return true;
  }
  template<typename N> void Post(const N &node) { Leave(node); }
  bool Walk(const parser::Program &program) {
    parser::Walk(program, *this);
    return !context_.AnyFatalError();
  }

private:
  SemanticsContext &context_;
};
~~~

As we can see, whenever the visitor's `Pre()` method is called for any given node of the parse tree, the `Enter()` method of a given checker is executed for that node. Similarly, whenever the visitor's `Post()` method is called for any given node of the parse tree, the `Leave()` method of a given checker is executed for that node.

In order to prevent code duplications, useful functions (to be called from within the checkers) were extracted into `tools.cc` file and exposed to te `semantics` namespace. E.g. `ExprChecker` makes good use of `semantics::FindExternallyVisibleObject()` function.

## Proposed checker of ERROR STOP statements

`ERROR STOP` and also `STOP` are very simple statements for which the Standard specifies small number of constraints, most of them are functional rather than semantic.

1. The `StopChecker` class should be defined to inherit from `BaseChecker`. It should contain one private field, a reference to the current semantic context.

2. The `Enter()` method in this new `StopChecker` class should be overridden for `parser::StopStmt` type, defined to handle both `R1160` (`STOP` statement) and for `R1161` (`ERROR STOP` statement).

3. As the constraints are specified the same way for both kinds of `parser:StopStmt` nodes, we don't need to take `parser::StopStmt::Kind` value into consideration. The code in the overloaded `Enter()` method should remain straightforward.

4. The utility functions from `lib/semantics/tools.cc` are useful in semantic checkers, e.g. `semantics::ExprHasTypeCategory()`, yet this toolbox still needs to be extended, namely, `semantics::ExprTypeKindIsDefault()` has proven to be beneficial.

5. Since majority of the statements (including `STOP` and `ERROR STOP`, i.e. their `QUIET` clause) consists of nested expressions (and the checker for expressions is executed in the Pass 1), the `StopChecker` should be added to `StatementSemanticsPass2`'s list of the visitors.
