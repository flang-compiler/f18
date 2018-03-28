#ifndef FORTRAN_SEMANTIC_CONTEXT_H_
#define FORTRAN_SEMANTIC_CONTEXT_H_

#include "scope.h"

namespace Fortran::semantics {

//
// Hold the context while performing semantic analysis. 
//
// It is typically passed around as a 'const reference' while
// performing the semantic analysis and new contexts can be 
// created by copying and modifying the provided 'const' instance. 
//
// Consequently the context shall be a relatively small data structure.
//
// The context is actually composed of two sets of fields. Most of them 
// describe the global context and are passed to subcontexts. The remaining
// fields represent the local context that is not passed to subcontexts.
//

class Context {

public:

  Context() = delete ;
 
  Context(Scope &scope) : 
      err_(std::cerr), 
      scope_(&scope),
      constant_(false)
  {
    ClearLocal();  
  }

  // Create a sub-context suitable to process for instance sub-expression.
  
  Context sub() const 
  { 
    Context ctxt(*this);
    ctxt.ClearLocal();  
    return ctxt; 
  }
  
  // Clear all fields from the local context
  void ClearLocal() 
  {
    local.scalar_ = false ; 
    local.integer_ = false ;
    local.logical_ = false ;
    local.defaultchar_ = false ;
  }

  Context & SetScope(Scope &scope) {
    scope_ = &scope ;
    return *this;
  }

  Context & SetConstant(bool v=true) {
    constant_ = v ;
    return *this;
  }
  
  Context & SetScalar(bool v=true) {
    local.scalar_ = v ;
    return *this;
  }
  
  Context & SetInteger(bool v=true) {
    local.integer_ = v ;
    return *this;
  }
  
  Context & SetLogical(bool v=true) {
    local.logical_ = v ;
    return *this;
  }

  Context & SetDefaultChar(bool v=true) {
    local.defaultchar_ = v ;
    return *this;
  }

  Scope & scope() const { return *scope_ ; }

  bool IsConstant() const { return constant_ ; }
  bool IsScalar() const { return local.scalar_ ; }
  bool IsInteger() const { return local.integer_ ; }
  bool IsLogical() const { return local.logical_ ; }
  bool IsDefaultChar() const { return local.defaultchar_ ; } 

  // Provide a stream to emit error messages.
  // TODO: Need a proper error framework
  std::ostream & err() const { return err_ ; } 
  
  // Emit an error message.   
  // TODO: Need a proper error framework
  void Error(const std::string &msg) const {
    err() << "Error: " << msg << "\n" ;     
  }  

  // Emit an internal error message.
  // TODO: Need a proper error framework
  void InternalError(const char *file, int line, const std::string &msg) const {
    err() << "Internal Error at "
          << file << " line " << line << ": "
          << msg << "\n" ;
  }
  
  // Emit a warning message.   
  // TODO: Need a proper error framework
  void Warning(const std::string &msg) const{
    err() << "Warning: " << msg << "\n" ;
  }

private:
  
  Context( const Context & ) = default ;

private:
    
  std::ostream & err_ ; 

  // The current scope
  Scope * scope_ ;  

  // True while processing a constant expression.
  bool constant_ ;  

  // The 'local' struct contains all the local information
  // that should not be passed to sub-contexts.  
  struct { 
   
    // True for an expression that is required to be scalar.
    // That property is cleared by the copy constructor.
    bool scalar_   ;  
    
    // true for an expression that is required to be integer.
    // That property is cleared by the copy constructor.
    bool integer_  ;  

    // true for an expression that is required to be logical.
    // That property is cleared by the copy constructor.
    bool logical_  ; 
    
    // True for an expression that is required to be of the 
    // default character type. 
    // That property is cleared by the copy constructor.
    bool defaultchar_ ;  

  } local ;

};

} // of namespace Fortran::semantics

#endif // of FORTRAN_SEMANTIC_CONTEXT_H_
