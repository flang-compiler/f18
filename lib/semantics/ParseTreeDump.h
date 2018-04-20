#ifndef FLANG_SEMA_PARSE_TREE_DUMP_H
#define FLANG_SEMA_PARSE_TREE_DUMP_H

#include "symbol.h"
#include "../parser/format-specification.h"
#include "../parser/idioms.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace Fortran::semantics {

// Provide a name for the each type of parse-tree node. 
// For nodes that are implemented as a struct the 
// name is provided by the className() static member.
// 
// For other types, we need to to 

template <typename T> inline std::string GetParserNodeName_base() {
  // The more pactical way of getting a type name is via typeid
  //    return DemangleCxxName( typeid(T).name() ) ;
  // but that is not possible here since LLVM requires no-rtti. 
  // 
  // For classes defined in parse-tree.h, use the name provided by
  // the static className() member. 
  return T::className(); 
}

// But for other types, we need to provide them manually.

#define  FLANG_PARSER_PROVIDE_TYPE_NAME(TYPE,NAME) \
template <> inline std::string GetParserNodeName_base<TYPE>() { \
  return NAME;\
}

FLANG_PARSER_PROVIDE_TYPE_NAME(long unsigned int,"long unsigned int")
FLANG_PARSER_PROVIDE_TYPE_NAME(long int,"long int")
FLANG_PARSER_PROVIDE_TYPE_NAME(int,"int")
FLANG_PARSER_PROVIDE_TYPE_NAME(bool,"bool")
FLANG_PARSER_PROVIDE_TYPE_NAME(const char*,"const char*")

FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::format::ControlEditDesc::Kind,"ControlEditDesc::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::format::IntrinsicTypeDataEditDesc::Kind,"IntrinsicTypeDataEditDesc::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::AccessSpec::Kind,"AccessSpec::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::BindEntity::Kind,"BindEntity::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::ConnectSpec::CharExpr::Kind,"ConnectSpec::CharExpr::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::DefinedOperator::IntrinsicOperator,"DefinedOperator::IntrinsicOperator")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::ImplicitStmt::ImplicitNoneNameSpec,"ImplicitStmt::ImplicitNoneNameSpec")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::InquireSpec::CharVar::Kind,"InquireSpec::CharVar::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::InquireSpec::IntVar::Kind,"InquireSpec::IntVar::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::InquireSpec::LogVar::Kind,"InquireSpec::LogVar::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::IntentSpec::Intent,"IntentSpec::Intent")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::IoControlSpec::CharExpr::Kind,"IoControlSpec::CharExpr::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::ProcedureStmt::Kind,"ProcedureStmt::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::SavedEntity::Kind,"SavedEntity::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::Sign,"Sign")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::StopStmt::Kind,"StopStmt::Kind")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::TypeParamDefStmt::KindOrLen,"TypeParamDefStmt::KindOrLen")
FLANG_PARSER_PROVIDE_TYPE_NAME(Fortran::parser::UseStmt::ModuleNature,"UseStmt::ModuleNature")
FLANG_PARSER_PROVIDE_TYPE_NAME( Fortran::parser::LoopBounds<Fortran::parser::ScalarIntConstantExpr> ,
             "LoopBounds<Expr>")
FLANG_PARSER_PROVIDE_TYPE_NAME( Fortran::parser::LoopBounds<Fortran::parser::ScalarIntExpr> ,
             "LoopBounds<Expr>")

template <typename T> inline std::string GetParserNodeName() {
  return GetParserNodeName_base< 
    typename std::remove_cv<
      typename std::remove_reference<        
        T
        >::type
      >::type
    >();   
} 

// Make it usable on objects
template <typename T> std::string GetParserNodeName(const T &x) {
  return GetParserNodeName<decltype(x)>() ;
} 


} // end of namespace


namespace Fortran::parser {

//
// Dump the Parse Tree hiearchy of any node 'x' of the parse tree.
//
// ParseTreeDumper().run(x)
//

class ParseTreeDumper {
private:
  int indent;
  std::ostream &out ;  
  bool emptyline;
public:
  
  ParseTreeDumper(std::ostream &out_ = std::cerr) : indent(0) , out(out_) , emptyline(false) { }

private:

  static bool startwith( const std::string str, const char *prefix) ;
  static std::string cleanup(const std::string &name) ;


public:

  void out_indent() {
    for (int i=0;i<indent;i++) {
      out << "| " ;
    }
  }


  template <typename T> bool Pre(const T &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }
    if ( UnionTrait<T> || WrapperTrait<T> ) {
      out << cleanup(Fortran::semantics::GetParserNodeName<decltype(x)>()) << " -> "  ;
      //out << cleanup(Fortran::semantics::GetParserNodeName<decltype(x)>()) << " -> "  ;
      emptyline = false ;
    } else {
      out << cleanup(Fortran::semantics::GetParserNodeName<decltype(x)>())    ;
      out << "\n" ; 
      indent++ ;
      emptyline = true ;
    }    
    return true ;
  }
  
  template <typename T> void Post(const T &x) { 
    if ( UnionTrait<T> || WrapperTrait<T> ) {
      if (!emptyline) { 
        out << "\n" ; 
        emptyline = true ; 
      }
    } else {
      indent--;
    }
  }

  bool PutName(const std::string &name, const semantics::Symbol *symbol) {
    if (emptyline) {
      out_indent();
      emptyline = false;
    }
    if (symbol) {
      out << "symbol = " << *symbol;
    } else {
      out << "Name = '" << name << '\'';
    }
    out << '\n';
    indent++;
    emptyline = true;
    return true;
  }

  bool Pre(const parser::Name &x) {
    return PutName(x.ToString(), x.symbol);
  }

  void Post(const parser::Name &) { 
    indent--;
  }

  bool Pre(const std::string &x) { 
    return PutName(x, nullptr);
  }
  
  void Post(const std::string &x) { 
    indent--;
  }

  bool Pre(const std::int64_t &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }    
    out << "int = '" << x << "'\n";
    indent++ ;
    emptyline = true ;    
    return true ;
  }
  
  void Post(const std::int64_t &x) { 
    indent--;
  }

  bool Pre(const std::uint64_t &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }    
    out << "int = '" << x << "'\n";
    indent++ ;
    emptyline = true ;    
    return true ;
  }
  
  void Post(const std::uint64_t &x) { 
    indent--;
  }

  // A few types we want to ignore


  template <typename T> bool Pre(const Fortran::parser::Statement<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Statement<T> &) { 
  }

  template <typename T> bool Pre(const Fortran::parser::Indirection<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Indirection<T> &) { 
  }

  template <typename T> bool Pre(const Fortran::parser::Integer<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Integer<T> &) { 
  }


  template <typename T> bool Pre(const Fortran::parser::Scalar<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Scalar<T> &) { 
  }

  template <typename... A> bool Pre(const std::tuple<A...> &) { 
    return true;
  }

  template <typename... A> void Post(const std::tuple<A...> &) { 
  }

  template <typename... A> bool Pre(const std::variant<A...> &) { 
    return true;
  }
  
  template <typename... A> void Post(const std::variant<A...> &) { 
  }


public:
  
  
}; 


template <typename T>
void DumpTree(const T &x, std::ostream &out=std::cout )
{
  ParseTreeDumper dumper(out);
  Fortran::parser::Walk(x,dumper); 
}


} // of namespace 


namespace Fortran::parser {

// Provide a explicit instantiation for a few selected node types.
// The goal is not to provide the instanciation of all possible 
// types but to insure that a call to DumpTree will not cause
// the instanciation of thousands of types.
//
 

#define FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,TYPE) \
 MODE template void Walk(const TYPE&, Fortran::parser::ParseTreeDumper &);

#define FLANG_PARSE_TREE_DUMPER_INSTANTIATE_ALL(MODE) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ProgramUnit) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SubroutineStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ProgramStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,FunctionStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ModuleStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,Expr) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ActionStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ExecutableConstruct) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,Block)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,DeclarationConstruct)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SpecificationPart)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,OtherSpecificationStmt)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SpecificationConstruct)\



FLANG_PARSE_TREE_DUMPER_INSTANTIATE_ALL(extern) 


} // of namespace 

#endif
