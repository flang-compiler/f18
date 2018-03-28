#include "../parser/idioms.h"
#include "../parser/parse-tree.h"

#include "SemanticData.h"
#include "GetValue.h"
#include "attr.h"
#include "type.h"
#include "scope.h"
#include "context.h"
#include "expr-types.h"

#include <cassert>

namespace sm  = Fortran::semantics ;
namespace psr = Fortran::parser ;

// The purpose of GetValue(x) is to provide a shortcut to the actual
// parse-tree node by skiping the qualifiers Indirection<>, Scalar<>, 
// Constant<>, Logical<>, Integer<> and DefaultChar<> 
//
// That name is quite ambigous and will likely be changed but 
// for now, use SkipQualifiers() as an alias. 
// 
#define SkipQualifiers(x) GetValue(x) 

#define INTERNAL_ERROR(msg) ctxt.InternalError(__FILE__ , __LINE__, msg)

namespace Fortran::semantics {

using sm::Context ;
using psr::visitors ;
using psr::Indirection ;

// Power, Multiply, Divide, Add, Subtract, Concat,
// LT, LE, EQ, NE, GE, GT, NOT, AND, OR, XOR, EQV, NEQV
using Oper = psr::DefinedOperator::IntrinsicOperator ;

static bool DoAnalysis(const Context &ctxt, const psr::CharLiteralConstantSubstring &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false; 
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::LiteralConstant &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::Designator &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::ArrayConstructor &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::StructureConstructor &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::TypeParamInquiry &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}

static bool DoAnalysis(const Context &ctxt, const psr::FunctionReference &expr) 
{
  bool ok = true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok; 
}


static bool DoAnalysis(const Context &ctxt, const psr::Expr::Parentheses &expr) 
{
  bool ok = true;
  const psr::Expr & arg = SkipQualifiers(expr.v); 

  ok &= TypeAnalysis( ctxt.sub(), arg  );   

  // TODO: Copy the type of arg 
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  
  return ok;
}

// Perform the type DoAnalysis on all intrinsic binary operations 
static bool DoUnaryAnalysis(const Context &ctxt, 
                            Oper oper,
                            const psr::Expr::IntrinsicUnary &expr
                            ) 
{
  bool ok = true ;
  const psr::Expr & arg = *(expr.v); 

  ok &= TypeAnalysis( ctxt.sub(), arg);   

  if (ok) {
    // Compute the return type.
    INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  } 
  return ok; 
}


//static bool DoAnalysis(const Context &ctxt, const psr::Expr:UnaryPlus &expr) 

static bool DoAnalysis(const Context &ctxt, const psr::Expr::UnaryPlus &expr) 
{
  return DoUnaryAnalysis(ctxt, Oper::Add, expr);
}


static bool DoAnalysis(const Context &ctxt, const psr::Expr::Negate &expr) 
{
  return DoUnaryAnalysis(ctxt, Oper::Subtract, expr);
}


static bool DoAnalysis(const Context &ctxt, const psr::Expr::NOT &expr) 
{
  return DoUnaryAnalysis(ctxt, Oper::NOT, expr);
}

// Perform the type DoAnalysis on all intrinsic binary operations 
static bool DoBinaryAnalysis(const Context &ctxt, 
                             Oper oper,
                             const psr::Expr::IntrinsicBinary &expr
                             ) 
{
  bool ok = true;

  const psr::Expr & arg1 = *std::get<0>(expr.t) ;
  const psr::Expr & arg2 = *std::get<1>(expr.t) ;

  ok &= TypeAnalysis( ctxt.sub(), arg1 ); 
  ok &= TypeAnalysis( ctxt.sub(), arg2 ); 

  if (ok) {
   
    const ExpressionType & type1 = GetType(arg1);
    const ExpressionType & type2 = GetType(arg2);
    
    // TODO: Compute and store the type of the result    

    // TODO: get the computed type of each operand,
    //       figure out if this is a valid operation,
    //       and compute its result type.
    //
    // Ideally, their should not be any needs to 
    // differentiate the operators. Everything 
    // could be handled as a generic function call.
    // 
    //
    // This is of course assuming that the system 
    // scope provides specific functions for each 
    // possible variation of the types (all functions
    // are elemental).
    //
    // For instance, let's assume an architecture with 
    //   - INTEGER of kind 1, 2, 4 and 8 
    //   - REAL of kind 4 and 8 
    //   - COMPLEX of kind 4 and 8
    // 
    // So a total of 8 numerical types.
    //
    // The intrinsic interface for OPERATOR+ shall 
    // provide 8*8 = 64 specific elemental functions 
    // to cover all supported cases: 
    //
    //    INTEGER(1) + INTEGER(1)  ->  INTEGER(1)
    //    INTEGER(1) + INTEGER(2)  ->  INTEGER(2)
    //    INTEGER(2) + INTEGER(1)  ->  INTEGER(2)
    //    ...
    //    REAL(4) + INTEGER(8) -> REAL(4) 
    //    ...
    //    COMPLEX(8) + COMPLEX(8) -> COMPLEX(8)
    // 
    // Another possibility could be to operate as 
    // in the C language; define only a subset of 
    // the operations and some well defined implicit
    // conversions rules. (i.e. in C++, the expression
    //   1.23 * 4  is equivalent to 1.23 * double(4) )
    // 
    // If we go that way, we only need to define a 
    // small subset of operations:
    //
    //    INTEGER(1) + INTEGER(1)  ->  INTEGER(1)
    //    INTEGER(2) + INTEGER(2)  ->  INTEGER(2)
    //    INTEGER(4) + INTEGER(4)  ->  INTEGER(4)
    //    INTEGER(8) + INTEGER(8)  ->  INTEGER(8)
    //    REAL(4) + REAL(4)  ->  REAL(4)
    //    REAL(8) + REAL(8)  ->  REAL(8)
    //    COMPLEX(4) + COMPLEX(4)  ->  COMPLEX(4)
    //    COMPLEX(8) + COMPLEX(8)  ->  COMPLEX(8)
    //   
    // However, we also need to introduce implicit
    // conversion rules that do not really exist 
    // in the Fortran language.
    // 
    // Another possibility could be to annotate 
    // the types in the intrinsic interfaces 
    // with some extensions to allow more flexibility
    // on the types. I do not think that this is a good
    // idea because that would probably lead to some 
    // ambiguities. For instance, assuming that
    // we only defined the specifics for the 8 
    // types above, which one of those would handle
    //     
    //   REAL(8) + COMPLEX(4) -> COMPLEX(8) 
    //
    // We see here that the result type COMPLEX(8) is 
    // not equal to any of the 2 operand types. That semantic 
    // is difficult to describe using simple type annotations. 
    //

    (void) type1;
    (void) type2;

    INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  } 
  
  return ok;
}


static bool DoAnalysis(const Context &ctxt, const psr::Expr::Power &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::Power, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::Multiply &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::Multiply, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::Divide &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::Divide, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::Add &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::Add, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::Subtract &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::Subtract, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::LT &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::LT, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::LE &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::LE, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::EQ &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::EQ, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::NE &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::NE, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::GE &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::GE, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::GT &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::GT, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::AND &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::AND, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::OR &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::OR, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::EQV &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::EQV, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::NEQV &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::NEQV, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::XOR &expr) 
{
  return DoBinaryAnalysis(ctxt, Oper::XOR, expr) ;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::Concat &expr) 
{  
  // Remark: strictly speaking, Concat is also a binary operator
  //         but its behavior is slightly unusal in the sense that 
  //         the 'len' of its result type is obtained by 
  //         adding its the 'len' of both operands. 
  //         This is something that can probably not be described
  //         using an intrinsic 'interface'.          
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::DefinedUnary &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::DefinedBinary &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::ComplexConstructor &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok;
}

static bool DoAnalysis(const Context &ctxt, const psr::Expr::PercentLoc &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok;
}

bool TypeAnalysis(const Context &ctxt, const psr::Designator &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok ; 
}

bool TypeAnalysis(const Context &ctxt, const psr::Variable &expr) 
{
  bool ok=true;
  INTERNAL_ERROR("Not Yet Implemented"); ok=false;
  return ok ; 
}


bool TypeAnalysis(const Context &ctxt, const psr::Expr &expr) 
{
  bool ok = true ; 

  // TODO: Before going recursive, we need to resolve ambiguous constructs:
  //   - functions call vs subscript vs substring vs ArrayConstructor
  // 
  
  // Now, perform the analysis recursively 
  std::visit( 
      [&](const auto &x) { 
        ok &= DoAnalysis(ctxt, SkipQualifiers(x) ); 
      }
      , expr.u);

  return ok ;
}


bool TypeAnalysis(const Context &ctxt, const psr::LogicalExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetLogical(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::DefaultCharExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetDefaultChar(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::IntExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetInteger(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ConstantExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetConstant(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::IntConstantExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetInteger(true);
  subctxt.SetConstant(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ScalarLogicalExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetScalar(true);
  subctxt.SetLogical(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ScalarIntExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetInteger(true);
  subctxt.SetScalar(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ScalarIntConstantExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetScalar(true);
  subctxt.SetInteger(true);
  subctxt.SetConstant(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ScalarDefaultCharExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetScalar(true);
  subctxt.SetDefaultChar(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}

bool TypeAnalysis(const Context &ctxt, const psr::ScalarDefaultCharConstantExpr &expr)
{
  Context subctxt(ctxt);
  subctxt.SetScalar(true);
  subctxt.SetDefaultChar(true);
  subctxt.SetConstant(true);
  return TypeAnalysis( subctxt, SkipQualifiers(expr) ); 
}


std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt,const psr::ScalarIntConstantExpr &expr) 
{
  return EvaluateAsInt( ctxt, SkipQualifiers(expr) ) ; 
}

std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt,const psr::ScalarIntExpr &expr) 
{
  return EvaluateAsInt( ctxt, SkipQualifiers(expr) ) ; 
}

std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt,const psr::Expr &expr) 
{ 
  bool constant_mode = ctxt.IsConstant() ;

  // For now, perform a simple evaluation of literal constants.
  // TODO: perform a complete recursive evaluation of the expression.
  // TODO: the type of the expression should be known to be integer scalar.

  // For now, only perform a simple evaluation of literal constants.
  if (std::holds_alternative<parser::LiteralConstant>(expr.u)) {
    auto &lc = std::get<parser::LiteralConstant>(expr.u);
    if (std::holds_alternative<parser::IntLiteralConstant>(lc.u)) {
      auto &ilc = std::get<parser::IntLiteralConstant>(lc.u);
      return std::get<std::uint64_t>(ilc.t) ;
    }
  }

  if ( constant_mode ) {
    INTERNAL_ERROR("constant evaluation was not possible in EvaluateAsInt");
  }
  return {};
}


std::optional<std::string> 
EvaluateAsString(const Context &ctxt,const psr::ScalarDefaultCharConstantExpr &expr)
{
  // TODO
  return {} ; 
}



std::optional<std::string> 
EvaluateAsString(const Context &ctxt,const psr::Expr &expr)
{
  // TODO 
  return {} ; 
}


const ExpressionType& 
GetType(const psr::Expr &expr)
{
  const ExpressionType * type = NULL ; 
  std::visit( 
      [&](const auto &x) { 
        auto & node = SkipQualifiers(x); 
        auto & sema = GetSema(node);
        type = sema.expr_type; 
      }
      , expr.u);
  assert(type) ; // TODO: internal error
  return *type; 
} 

const ExpressionType& 
GetType(const psr::LogicalExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::DefaultCharExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::IntExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::ConstantExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::IntConstantExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::ScalarLogicalExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::ScalarIntConstantExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::ScalarDefaultCharExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}

const ExpressionType& 
GetType(const psr::ScalarDefaultCharConstantExpr &expr)
{
  const psr::Expr & node = SkipQualifiers(expr) ;
  return GetType(node) ;  
}


const ReferenceType& 
GetType(const psr::Designator &ref) 
{
  const ReferenceType * type = GetSema(ref).ref_type ;
  assert(type) ; 
  return *type;
}

const ReferenceType& 
GetType(const psr::Variable &ref) {
  const ReferenceType * type = GetSema(ref).ref_type ;
  assert(type) ; 
  return *type;
}

}
