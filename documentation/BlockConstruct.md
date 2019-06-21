<!--
Copyright (c) 2019, Arm Ltd.  All rights reserved.
-->
# Design proposal for Block Construct statement semantic analysis
# The implementation of Block construct is based on the Section 11.1.4 of
# Fortran 2018 standards Document

The semantic checks of Block Construct in Section 11.1.4 are going to be
implemented in these new files:
 check-block.cc and check-block.h

The new files check-block.•   implements/have BlockChecker class.
The new class should contain (as others) one private field, a reference to the
current semantic context

The following sections contain:
A. Implementation proposal
and
B. Questions.

A. Implementation:
1. Implement C1107 with a similar approach to Do by using the method:
    void Post(const parser::ZZZStmt &).
Where ZZZ in ZZZStmt are the invalid statements on block specification part.
This method should be inside the BlockChecker class.

2. Implement C1108  as a method in BlockChecker class.
For this method, when in Block and if a SaveStmt is found the SavedEntity list
will have its type checked for Common Statement.
In case a  Common Statement is found  a message error will be printed.

3. Check C1109 will not be implemented inside BlockChecker.
C1109 is implemented in resolve-labels.cc as:
    void CheckName(const parser::BlockConstruct &blockConstruct)

B. Questions:
1. Is Point number 2 related to Semantic Checks?
Because I do not intend to implement Point 2 of section 11.1.4.
This point is related to Asynchrounous and Volatile statements inside Block.
I do not know if this should be implemented inside the BlockChecker class
as a checker.

2. Is Point number 3 related to  Semantic Checks?
I do no intend  to implement Point 3 of section 11.1.4.
This point is related to the evaluations of the specification expression inside
block construct.
I do not know if this should be implemented inside the BlockChecker class

3. Point number 4 in section 11.1.4 checks if a GOTO statement inside a BLOCK
construnct statment points to and end-block-statement within the block.
So should this check be handled inside the Block Construct statement or inside
GOTO statement.
Or should the Go To statements handle the check?
