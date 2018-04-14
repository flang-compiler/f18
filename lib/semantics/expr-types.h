#ifndef FORTRAN_SEMANTICS_EXPR_TYPE_H_
#define FORTRAN_SEMANTICS_EXPR_TYPE_H_

#include <cassert> 
#include "context.h" 

//
// The purpose of this file is to provide the API to 
// resolve the type of any Fortran expression. 
//      

namespace Fortran::semantics {

namespace psr = Fortran::parser ;
  
class ExprContext ;

//
// Represent the possible states of a Dynamic<T>
//
//
enum class DynamicState {
    Maybe,       // It is not known if the value can be computed.    
    Known,       // The value is known 
    Available,   // The value is not known but can be computed.
    Unavailable, // The value is not known and cannot be computed.
};

//
// A Dynamic<T> represent a value that it may or not be possible to 
// evaluate at compile-time (I know! the name is not good.) 
//
// Conceptually, this is similar to std::optional<T> but with more possible states. 
//
// It shall be noted that even though a Dynamic<T> can represent an Available 
// value, the class is not in charge of that computation. That state represents 
// the fact tht the value is known to be computable by an external mean.    
//
template <typename T> 
class Dynamic {
public:
  Dynamic(const T& v) : state_(DynamicState::Known) , value_(v) { } 
  Dynamic() : state_(DynamicState::Known) , value_() { } 
    
  DynamicState state() { return state_; }

  void set(const T& v) { 
    state_ = DynamicState::Known;
    value_ = v ;
  } 

  // Mark the value as available. 
  // If the value is already known then it is discarded thus saving memory
  void setAvailable() {
    assert( state_ != DynamicState::Unavailable );
    if (value_) 
      value_.reset() ;
    state_ = DynamicState::Available;
  } 

  void setUnavailable() {
    assert( state_ != DynamicState::Available );
    assert( state_ != DynamicState::Known );
    state_ = DynamicState::Available;
  } 

  const std::optional<T> & get() { return value_; } 

private:
  DynamicState state_;
  std::optional<T> value_; 
};

//
// The type of an expression is composed of two parts; the 
// element type and the rank (from 0 to the maximum allowed). 
//
// For arrays, the extent of each dimension may or may not be 
// known at compile time. When provided, that information will 
// be used to perform additional consistancy checks on array 
// conformances.
//
class ExpressionType {
public:   
  
  ExpressionType(TypeSpec &elemtype, size_t rank) :
      elemtype_(&elemtype),
      shape_(rank)
  {}

  bool valid() { return elemtype_ != NULL; }

  const TypeSpec &type() { return *elemtype_; } 
  size_t rank() { return shape_.size(); } 

  bool isScalar() { return rank() == 0 ; } 
  bool isArray() { return !isScalar() ; }

private:
  const TypeSpec *elemtype_ ; // The type of each element. 
  std::vector< std::optional<int64_t> > shape_ ;  // provide the extent of each dimension when known.
};


// 
// A RefDimension represents the information known about a single dimension  
// within a reference type. 
//
class RefDimension {
public:
  Dynamic<int64_t> lbound; // Lower bound of indices
  Dynamic<int64_t> ubound; // Upper bound of indices
  Dynamic<int64_t> mult; // Multiplier when computing byte offset 
};


// 
// The type of a reference is more complex than the type of an
// expression since it also carries attributes from the referenced 
// object and it may also carry the information required to 
// access the referenced data (i.e. while evaluating constant 
// expressions) 
//
//
class ReferenceType {
public:

  // TODO

  size_t rank() { return shape_.size(); } 

private:

  // The type of each element.
  const TypeSpec *elemtype_ ; 

  // Relevant attributes taken from the referenced object
  Attrs attrs_ ; 
  
  // Describe the whole shape of the array
  std::vector<RefDimension> shape_ ; 

  // Provide the base offset in bytes for accessing the referenced data  
  // compared to the address of the referenced object. 
  //
  // For scalar references (or whole arrays), the base offset is zero.  
  //
  // For structures, the base is typically the offset of the accessed member. 
  //
  // For subscripts, the base can be combined with shape_ and with 
  // the 'mult' component of shape_ to compute the byte offset of 
  // each element A(i0,i1,...,ik,...)
  //   = A + base + i0*shape[0].mult + ... ik*shape[k].mult + ... 
  //
  // This is of course only permitted if 'base_' and all 'mult' factors
  // are known which shall be the case in constant expressions.
  //  
  Dynamic<int64_t> base_;

  // Note: The type of a whole reference will be constructed
  //       mostly from left to right but by fully resolving 
  //       the right part of each member access. 
  //       For instance, consider the following reference  
  //
  //           A%B(:,:)%C(1,2)%D
  //
  //       represented by the parse-tree
  // 
  //          | | DataReference -> StructureComponent
  //          | | | DataReference -> ArrayElement
  //          | | | | DataReference -> StructureComponent
  //          | | | | | DataReference -> ArrayElement
  //          | | | | | | DataReference -> StructureComponent
  //          | | | | | | | DataReference -> Name = A
  //          | | | | | | | Name = B
  //          | | | | | | SectionSubscript -> SubscriptTriplet
  //          | | | | | | SectionSubscript -> SubscriptTriplet
  //          | | | | | Name = C 
  //          | | | | SectionSubscript -> Expr -> LiteralConstant -> IntLiteralConstant
  //          | | | | | int = '1'
  //          | | | | SectionSubscript -> Expr -> LiteralConstant -> IntLiteralConstant
  //          | | | | | int = '2'
  //          | | | Name = D 
  //
  //       The type resolution will proceed as follow:
  //         [a] resolve the type of A 
  //         [b] resolve the type of member B using [a] 
  //         [c] resolve the type of B(:,:) using [b] 
  //         [d] resolve type of A%B(:,:) using [a] and [c] 
  //         [e] resolve the type of member C using [d]
  //         [f] resolve the type of C(1,3) using [e] 
  //         [g] resolve the type of A%B(:,:)%C(1,2) using [d] and [f] 
  //         [h] resolve the type of member D using [g] 
  //         [i] resolve the type ofA%B(:,:)%C(1,2)%D using [g] and [h] 
  //         
  //       In the parse-tree the various components of the reference
  //       are organized from left to right which means that the nodes
  //       can be associated to the following sub-expressions
  //             A, A%B, A%B(:,:), A%B(:,:)%C etc.
  //       It shall be noted that in the method given previously, some 
  //       of those sub-expressions are not given any type. For instance, 
  //       this is the case for A%B and A%B(:,:)%C.
  //       The reason is that Fortran does not allow arrays of arrays which 
  //       would be problematic here for A%B(:,:)%C 
  //         
  //       Another possible strategy is to resolve the type from left 
  //       to right until a member access is performed on an array type 
  //       The type of right-hand side operator is then computed 
  //       independently:
  // 
  //       [a] resolve the type of A 
  //       [b] resolve the type of member B using [a] 
  //       [c] resolve the type of A%B using [a] and [b] 
  //       [d] resolve the type of A%B(:,:) using [c]
  //       ===> A%B(:,:) is a non-scalar structure type so
  //       [e] resolve the type of member C using [d]
  //       [f] resolve the type of C(1,2) using [e]
  //       [g] resolve the type of member D using [f]
  //       [h] resolve the type of member C(1,2)%D using [f] and [g] 
  //       ==> and now merge the array type of A%B(:,:) with the type of C(1,2)%D 
  //       [i] resolve the type of member A%B(:,:)%C(1,2)%D using [d] and [h] 
  //       
  //       With that scheme, the types computed in [e]..[h] are not related 
  //       to the base variable A. They apply to a single element of A%B(:,:).
  //       We could say that those types are 'elemental'. The main disadvantage
  //       of that scheme is that it does not allow to get the true type of 
  //       any 'elemental' expression. For instance, the type of A%B(:,:)%C(1,2) 
  //       is never computed. Though, that is only a minor issue because those 
  //       types shall not be needed in practice
  //  

};

//
// Perform the Semantic analysis of an expression. 
// 
// After calling that function, all types shall be resolved 
// within the given expression.
//
// If the analysis was successful then return true else 
// false.
//
bool TypeAnalysis(const Context &ctxt, const psr::Expr &);

bool TypeAnalysis(const Context &ctxt, const psr::LogicalExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::DefaultCharExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::IntExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ConstantExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::IntConstantExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ScalarLogicalExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ScalarIntExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ScalarIntConstantExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ScalarDefaultCharExpr &);
bool TypeAnalysis(const Context &ctxt, const psr::ScalarDefaultCharConstantExpr &);

// 
// Perform the Semantic analysis of a Designator 
//
// Reminder: a designator is a 'name' with any number of 
//           array subscripts, substrings, and member operations.
//           Examples: X, X%Y, X(4:5)%Y(4)%Z 
//
// If the analysis was successful then return true else 
// false.
// 
bool TypeAnalysis(const Context &ctxt, const psr::Designator &);

// 
// Perform the type analysis of a Variable. 
//
// Reminder: in the context of the parse-tree, a Variable is 
// either a Designator or a FunctionReference returning a 
// pointer type. Conceptually, a Variable is anything that 
// can be found at the left side of an assignment.
//
// If the analysis was successful then return true else 
// false. 
//
bool TypeAnalysis(const Context &ctxt, const psr::Variable &);

//
// Provide the type of an expression.
// 
// That function shall only be used after a successful 
// type analysis was applied to the expression. 
// 
const ExpressionType& GetType(const psr::Expr &); 

// Various helpers to access the type of the underlying Expr
const ExpressionType& GetType(const psr::LogicalExpr &); 
const ExpressionType& GetType(const psr::DefaultCharExpr &); 
const ExpressionType& GetType(const psr::IntExpr &); 
const ExpressionType& GetType(const psr::ConstantExpr &); 
const ExpressionType& GetType(const psr::IntConstantExpr &); 
const ExpressionType& GetType(const psr::ScalarLogicalExpr &); 
const ExpressionType& GetType(const psr::ScalarIntConstantExpr &); 
const ExpressionType& GetType(const psr::ScalarDefaultCharExpr &); 
const ExpressionType& GetType(const psr::ScalarDefaultCharConstantExpr &); 

//
// Provide the type of a designator.
// 
// That function shall only be used after a successful 
// type analysis was applied to the designator. 
// 
// 
const ReferenceType& GetType(const psr::Designator &);

//
// Provide the type of a variable.
// 
// That function shall only be used after a successful 
// type analysis was applied to the varible. 
// 
const ReferenceType& GetType(const psr::Variable &);

// 
// Attempt to evaluate a scalar integer constant expression. 
//
// An unset result indicates that the evaluation could not be performed. 
// In that case, an error message should already have been emited.
//
// Remark: All type related errors are supposed to have been catched 
// earlier but other errors can happen during the evaluation of expressions.
//
// Typical examples of illegal evaluations are divisions by zero or 
// the sqrt of a negative value.
//
std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt, const psr::ScalarIntConstantExpr &) ;

// Attempt to evaluate a scalar integer expression. 
// 
// The evaluation context can be set to be Constant in which case, the evaluation will be 
// subject to the constant expression rules.
//
// If the evaluation contect is not set to be Constant, a unset result may indicate that
// the expression can only be evaluated at runtime.
// 
// If the evaluation contect is set to be Constant then the evaluation is mandatory and
// a unset result indicates that at least one error message was emited  
//
std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt, const psr::ScalarIntExpr &) ;


// Attempt to evaluate an expression as a scalar integer. 
//
// The type of the expression must be scalar integer else an internal error occurs. 
//
// If the evaluation contect is not set to be Constant, a unset result may indicate that
// the expression can only be evaluated at runtime.
// 
// If the evaluation contect is set to be Constant then the evaluation is mandatory and
// a unset result indicates that at least one error message was emited  
//
std::optional<int64_t> 
EvaluateAsInt(const Context &ctxt, const psr::Expr &) ;



std::optional<std::string> 
EvaluateAsString(const Context &ctxt, const psr::ScalarDefaultCharConstantExpr &) ;

std::optional<std::string> 
EvaluateAsString(const Context &ctxt, const psr::Expr &) ;

} // end of namespace Fortran::semantics


#endif // FORTRAN_SEMANTICS_EXPR_TYPE_H_ 
