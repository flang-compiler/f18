#ifndef FORTRAN_SEMA_DATA_H_
#define FORTRAN_SEMA_DATA_H_

#include <cassert> 
#include <iostream> 

//#include "StatementMap.h" 
//#include "expr-types.h" 


//
// 
// Declare here the members of the Semantic<T> that will 
// be attached to each parse-tree class T. The default is 
// an empty struct. All members added here shall be 
// copiable and should be provided with a default value.  
//
// Here are a few common fields 
//  
//  Scope *scope_provider  = the scope provided by a construct or statement
//  int stmt_index         = the index used in the StatementMap 
//
// Remark: Weither we want to annotate parse-tree nodes with 
// semantic information is still debatable. 
//

namespace Fortran::semantics {

class StatementMap;
class ExpressionType;
class ReferenceType;
class Scope;
class LabelTable;


#define DEFINE_SEMANTIC_DATA(Class) \
  inline Semantic<Fortran::parser::Class> & GetSema(const Fortran::parser::Class &node) { \
    assert(node.s) ;                  \
    return *(node.s) ;                \
  }  \
  template <> struct Semantic<Fortran::parser::Class> { \
    Semantic<Fortran::parser::Class>(Fortran::parser::Class *node) {} \
    enum {IS_DECLARED=1};
  
#define END_SEMANTIC_DATA \
  }

// Some fields that need to be defined for all statements
#define SEMANTIC_STMT_FIELDS \
   int stmt_index=0 

DEFINE_SEMANTIC_DATA(ProgramUnit)
  StatementMap *statement_map=0 ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(MainProgram)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SubroutineSubprogram)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FunctionSubprogram)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(Module)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(DerivedTypeDef)
  // WARNING: there is also a sm::DerivedTypeDef defined in types.h 
  Scope *scope_provider=0 ;
END_SEMANTIC_DATA;


DEFINE_SEMANTIC_DATA(AssignmentStmt)
  SEMANTIC_STMT_FIELDS; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(DataStmt)
  SEMANTIC_STMT_FIELDS; 
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FunctionStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SubroutineStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ModuleStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
  
DEFINE_SEMANTIC_DATA(EndModuleStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
  
DEFINE_SEMANTIC_DATA(StmtFunctionStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndFunctionStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndSubroutineStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TypeDeclarationStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(DerivedTypeStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndTypeStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(PrintStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(UseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ProgramStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndProgramStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ImplicitStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AccessStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AllocatableStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AsynchronousStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(BindStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CodimensionStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ContiguousStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ContainsStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(DimensionStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ExternalStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(IntentStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(IntrinsicStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(NamelistStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(OptionalStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(PointerStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ProtectedStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SaveStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TargetStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ValueStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(VolatileStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CommonStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EquivalenceStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(BasedPointerStmt) // extension
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(GenericStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ParameterStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EnumDef)
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EnumDefStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndEnumStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(InterfaceStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndInterfaceStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(IfThenStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ElseIfStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
  
DEFINE_SEMANTIC_DATA(ElseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
  
DEFINE_SEMANTIC_DATA(EndIfStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(IfStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SelectCaseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CaseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndSelectStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SelectRankStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SelectRankCaseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SelectTypeStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ProcedureDeclarationStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(StructureStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(StructureDef::EndStructureStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FormatStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EntryStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ImportStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AllocateStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(BackspaceStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CallStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CloseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ContinueStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(DeallocateStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndfileStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EventPostStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EventWaitStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CycleStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ExitStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FailImageStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FlushStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FormTeamStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(GotoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(InquireStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(LockStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(NullifyStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(OpenStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(PointerAssignmentStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ReadStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ReturnStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(RewindStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(StopStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SyncAllStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SyncImagesStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SyncMemoryStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(SyncTeamStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(UnlockStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(WaitStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(WhereStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(WriteStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ComputedGotoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ForallStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
 
DEFINE_SEMANTIC_DATA(ForallConstructStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndForallStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;
 
DEFINE_SEMANTIC_DATA(ArithmeticIfStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AssignStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AssignedGotoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(PauseStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(PrivateStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TypeBoundProcedureStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TypeBoundGenericStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FinalProcedureStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ComponentDefStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EnumeratorDefStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TypeGuardStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(NonLabelDoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(LabelDoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndDoStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(BlockStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndBlockStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(AssociateStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndAssociateStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ChangeTeamStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndChangeTeamStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CriticalStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(EndCriticalStmt)
  SEMANTIC_STMT_FIELDS;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(CharLiteralConstantSubstring)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(LiteralConstant)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(Designator)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(ArrayConstructor)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(StructureConstructor)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(TypeParamInquiry)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(FunctionReference)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

// Reminder:  Semantic<Expr::IntrinsicUnary> is used by
//            Expr::Parentheses. Expr::UnaryPlus, Expr::Negate,
//            amd Expr::NOT
DEFINE_SEMANTIC_DATA(Expr::IntrinsicUnary)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

// Reminder:  Semantic<Expr::IntrinsicBinary> is used by
//             Expr::Power, Expr::Multiply, Expr::Divide, Expr::Add, 
//             Expr::Subtract, Expr::Concat, Expr::LT, Expr::LE, 
//             Expr::EQ, Expr::NE, Expr::GE, Expr::GT, Expr::AND, 
//             Expr::EQV, Expr::EQV, Expr::NEQV, Expr::XOR
DEFINE_SEMANTIC_DATA(Expr::IntrinsicBinary)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(Expr::PercentLoc)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(Expr::DefinedUnary)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;

DEFINE_SEMANTIC_DATA(Expr::DefinedBinary)
  ExpressionType *expr_type ;
END_SEMANTIC_DATA;


#undef DEFINE_SEMANTIC_DATA
#undef END_SEMANTIC_DATA_SEMANTIC
#undef SEMANTIC_STMT_FIELDS


// Initialize the semantic information attached to a parser-tree node
//
// Ideally, the function should be called once to initialize the data structure 
// used to hold Semantic data in each node of the parse tree. 
// By default, calling that function twice on the same parse-tree node is incorrect
// but the 'strict' argument can be set to false to allow for the reuse of pre-existing 
// data. 
//
template <typename T>  Semantic<T> & InitSema(const T &node, bool strict=true) { 

  // Do not use the default implementation!
  // If the following assert fails, then a DECLARE_SEMANTIC_DATA is 
  // missing below
  assert(Semantic<T>::IS_DECLARED);

  if (node.s) {
    if (strict) {
      // TODO: emit proper message
      std::cerr << "Duplicate call of " << __PRETTY_FUNCTION__ << "\n" ;
      exit(1);
    } else {
      return *(node.s); 
    }
  }
  auto s = new Semantic<T>( const_cast<T*>(&node) ) ;
  const_cast<T&>(node).s = s; 
  return *s ; 
} 

#if 0
// A few parse-tree node classes are struct derived from a
// base class. For those classes, the Semantic<T> is already  
// provided by the base class. 
//
// For instance, GetSema() applied to a Expr::Add node will
// note return a Semantic<Expr::Add> but a Semantic<Expr::IntrinsicUnary>
// 
// This is handled by the macro DERIVED_GET_SEMA
//

#define DERIVED_GET_SEMA( FROM , TO )                    \
inline Semantic<Fortran::parser::FROM> &           \
GetSema(const Fortran::parser::TO &node) {               \
  assert(Semantic<Fortran::parser::FROM>::IS_DECLARED); \
  assert(node.s) ; \
  return *(node.s) ;\
}

DERIVED_GET_SEMA( Expr::IntrinsicUnary, Expr::Parentheses) 
DERIVED_GET_SEMA( Expr::IntrinsicUnary, Expr::UnaryPlus) 
DERIVED_GET_SEMA( Expr::IntrinsicUnary, Expr::Negate) 
DERIVED_GET_SEMA( Expr::IntrinsicUnary, Expr::NOT) 

DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Power) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Multiply) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Divide) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Add) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Subtract) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::Concat) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::LT) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::LE) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::GT) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::GE) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::NE) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::EQ) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::EQV) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::NEQV) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::XOR) 
DERIVED_GET_SEMA( Expr::IntrinsicBinary, Expr::ComplexConstructor) 
#endif

// // Retreive the semantic information attached to a parser-tree node.
// template <typename T> Semantic<T> & GetSema(const T &node) { 
//   // Do not use the default implementation!
//   // If the following assert fails, then a DECLARE_SEMANTIC is missing above 
//   assert(Semantic<T>::IS_DECLARED); 
//   assert(node.s) ;
//   return *(node.s) ;
// } 



} // of namespace Fortran::semantics

#endif // FORTRAN_SEMA_DATA_H_
