// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mod-file.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "../evaluate/tools.h"
#include "../parser/parsing.h"
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <ostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace Fortran::semantics {

using namespace parser::literals;

// The initial characters of a file that identify it as a .mod file.
// The first three bytes are a Unicode byte order mark that ensures
// that the module file is decoded as UTF-8 even if source files
// are using another encoding.
static constexpr auto magic{"\xef\xbb\xbf!mod$ v1 sum:"};

static std::optional<SourceName> GetSubmoduleParent(const parser::Program &);
static std::string ModFilePath(const std::string &dir, const SourceName &,
    const std::string &ancestor, const std::string &suffix);
static std::vector<const Symbol *> CollectSymbols(const Scope &);
static void PutEntity(std::ostream &, const Symbol &);
static void PutObjectEntity(std::ostream &, const Symbol &);
static void PutProcEntity(std::ostream &, const Symbol &);
static void PutPassName(std::ostream &, const std::optional<SourceName> &);
static void PutTypeParam(std::ostream &, const Symbol &);
static void PutEntity(std::ostream &, const Symbol &, std::function<void()>);
static void PutInit(std::ostream &, const Symbol &, const MaybeExpr &);
static void PutInit(std::ostream &, const MaybeIntExpr &);
static void PutBound(std::ostream &, const Bound &);
static std::ostream &PutAttrs(std::ostream &, Attrs,
    const MaybeExpr & = std::nullopt, std::string before = ","s,
    std::string after = ""s);
static std::ostream &PutAttr(std::ostream &, Attr);
static std::ostream &PutType(std::ostream &, const DeclTypeSpec &);
static std::ostream &PutLower(std::ostream &, const std::string &);
static bool WriteFile(const std::string &, std::string &&);
static bool FileContentsMatch(
    std::fstream &, const std::string &, const std::string &);
static std::string GetHeader(const std::string &);
static std::size_t GetFileSize(const std::string &);

// Collect symbols needed for a subprogram interface
class SubprogramSymbolCollector {
public:
  using SymbolSet = std::set<const Symbol *>;

  SubprogramSymbolCollector(const Symbol &symbol)
    : symbol_{symbol}, scope_{*symbol.scope()} {}
  const SymbolVector &symbols() const { return need_; }
  const std::set<SourceName> &imports() const { return imports_; }
  void Collect();

private:
  const Symbol &symbol_;
  const Scope &scope_;
  bool isInterface_{false};
  SymbolVector need_;  // symbols that are needed
  SymbolSet needSet_;  // symbols already in need_
  SymbolSet useSet_;  // use-associations that might be needed
  std::set<SourceName> imports_;  // imports from host that are needed

  void DoSymbol(const Symbol &);
  void DoSymbol(const SourceName &, const Symbol &);
  void DoType(const DeclTypeSpec *);
  void DoBound(const Bound &);
  void DoParamValue(const ParamValue &);
  bool NeedImport(const SourceName &, const Symbol &);

  template<typename T> void DoExpr(evaluate::Expr<T> expr) {
    for (const Symbol *symbol : evaluate::CollectSymbols(expr)) {
      DoSymbol(DEREF(symbol));
    }
  }
};

bool ModFileWriter::WriteAll() {
  WriteAll(context_.globalScope());
  return !context_.AnyFatalError();
}

void ModFileWriter::WriteAll(const Scope &scope) {
  for (const auto &child : scope.children()) {
    WriteOne(child);
  }
}

void ModFileWriter::WriteOne(const Scope &scope) {
  if (scope.kind() == Scope::Kind::Module) {
    auto *symbol{scope.symbol()};
    if (!symbol->test(Symbol::Flag::ModFile)) {
      Write(*symbol);
    }
    WriteAll(scope);  // write out submodules
  }
}

// Write the module file for symbol, which must be a module or submodule.
void ModFileWriter::Write(const Symbol &symbol) {
  auto *ancestor{symbol.get<ModuleDetails>().ancestor()};
  auto ancestorName{ancestor ? ancestor->GetName().value().ToString() : ""s};
  auto path{ModFilePath(context_.moduleDirectory(), symbol.name(), ancestorName,
      context_.moduleFileSuffix())};
  PutSymbols(*symbol.scope());
  if (!WriteFile(path, GetAsString(symbol))) {
    context_.Say(symbol.name(), "Error writing %s: %s"_err_en_US, path,
        std::strerror(errno));
  }
}

// Return the entire body of the module file
// and clear saved uses, decls, and contains.
std::string ModFileWriter::GetAsString(const Symbol &symbol) {
  std::stringstream all;
  auto &details{symbol.get<ModuleDetails>()};
  if (!details.isSubmodule()) {
    all << "module " << symbol.name();
  } else {
    auto *parent{details.parent()->symbol()};
    auto *ancestor{details.ancestor()->symbol()};
    all << "submodule(" << ancestor->name();
    if (parent != ancestor) {
      all << ':' << parent->name();
    }
    all << ") " << symbol.name();
  }
  all << '\n' << uses_.str();
  uses_.str(""s);
  all << useExtraAttrs_.str();
  useExtraAttrs_.str(""s);
  all << decls_.str();
  decls_.str(""s);
  auto str{contains_.str()};
  contains_.str(""s);
  if (!str.empty()) {
    all << "contains\n" << str;
  }
  all << "end\n";
  return all.str();
}

// Put out the visible symbols from scope.
void ModFileWriter::PutSymbols(const Scope &scope) {
  std::stringstream typeBindings;  // stuff after CONTAINS in derived type
  for (const auto *symbol : CollectSymbols(scope)) {
    PutSymbol(typeBindings, symbol);
  }
  if (auto str{typeBindings.str()}; !str.empty()) {
    CHECK(scope.IsDerivedType());
    decls_ << "contains\n" << str;
  }
}

// Emit a symbol to decls_, except for bindings in a derived type (type-bound
// procedures, type-bound generics, final procedures) which go to typeBindings.
void ModFileWriter::PutSymbol(
    std::stringstream &typeBindings, const Symbol *symbol) {
  if (symbol == nullptr) {
    return;
  }
  std::visit(
      common::visitors{
          [&](const ModuleDetails &) { /* should be current module */ },
          [&](const DerivedTypeDetails &) { PutDerivedType(*symbol); },
          [&](const SubprogramDetails &) { PutSubprogram(*symbol); },
          [&](const GenericDetails &x) {
            PutGeneric(*symbol);
            PutSymbol(typeBindings, x.specific());
            PutSymbol(typeBindings, x.derivedType());
          },
          [&](const UseDetails &) { PutUse(*symbol); },
          [](const UseErrorDetails &) {},
          [&](const ProcBindingDetails &x) {
            bool deferred{symbol->attrs().test(Attr::DEFERRED)};
            typeBindings << "procedure";
            if (deferred) {
              typeBindings << '(' << x.symbol().name() << ')';
            }
            PutPassName(typeBindings, x.passName());
            PutAttrs(typeBindings, symbol->attrs());
            typeBindings << "::" << symbol->name();
            if (!deferred && x.symbol().name() != symbol->name()) {
              typeBindings << "=>" << x.symbol().name();
            }
            typeBindings << '\n';
          },
          [&](const GenericBindingDetails &x) {
            for (const auto *proc : x.specificProcs()) {
              typeBindings << "generic::" << symbol->name() << "=>"
                           << proc->name() << '\n';
            }
          },
          [&](const NamelistDetails &x) {
            decls_ << "namelist/" << symbol->name();
            char sep{'/'};
            for (const auto *object : x.objects()) {
              decls_ << sep << object->name();
              sep = ',';
            }
            decls_ << '\n';
          },
          [&](const CommonBlockDetails &x) {
            decls_ << "common/" << symbol->name();
            char sep = '/';
            for (const auto *object : x.objects()) {
              decls_ << sep << object->name();
              sep = ',';
            }
            decls_ << '\n';
            if (symbol->attrs().test(Attr::BIND_C)) {
              PutAttrs(decls_, symbol->attrs(), x.bindName(), ""s);
              decls_ << "::/" << symbol->name() << "/\n";
            }
          },
          [&](const FinalProcDetails &) {
            typeBindings << "final::" << symbol->name() << '\n';
          },
          [](const HostAssocDetails &) {},
          [](const MiscDetails &) {},
          [&](const auto &) { PutEntity(decls_, *symbol); },
      },
      symbol->details());
}

void ModFileWriter::PutDerivedType(const Symbol &typeSymbol) {
  auto &details{typeSymbol.get<DerivedTypeDetails>()};
  PutAttrs(decls_ << "type", typeSymbol.attrs());
  if (const DerivedTypeSpec * extends{typeSymbol.GetParentTypeSpec()}) {
    decls_ << ",extends(" << extends->name() << ')';
  }
  decls_ << "::" << typeSymbol.name();
  auto &typeScope{*typeSymbol.scope()};
  if (!details.paramNames().empty()) {
    char sep{'('};
    for (const auto &name : details.paramNames()) {
      decls_ << sep << name;
      sep = ',';
    }
    decls_ << ')';
  }
  decls_ << '\n';
  if (details.sequence()) {
    decls_ << "sequence\n";
  }
  PutSymbols(typeScope);
  decls_ << "end type\n";
}

// Attributes that may be in a subprogram prefix
static const Attrs subprogramPrefixAttrs{Attr::ELEMENTAL, Attr::IMPURE,
    Attr::MODULE, Attr::NON_RECURSIVE, Attr::PURE, Attr::RECURSIVE};

void ModFileWriter::PutSubprogram(const Symbol &symbol) {
  auto attrs{symbol.attrs()};
  auto &details{symbol.get<SubprogramDetails>()};
  Attrs bindAttrs{};
  if (attrs.test(Attr::BIND_C)) {
    // bind(c) is a suffix, not prefix
    bindAttrs.set(Attr::BIND_C, true);
    attrs.set(Attr::BIND_C, false);
  }
  Attrs prefixAttrs{subprogramPrefixAttrs & attrs};
  // emit any non-prefix attributes in an attribute statement
  attrs &= ~subprogramPrefixAttrs;
  std::stringstream ss;
  PutAttrs(ss, attrs);
  if (!ss.str().empty()) {
    decls_ << ss.str().substr(1) << "::" << symbol.name() << '\n';
  }
  bool isInterface{details.isInterface()};
  std::ostream &os{isInterface ? decls_ : contains_};
  if (isInterface) {
    os << "interface\n";
  }
  PutAttrs(os, prefixAttrs, std::nullopt, ""s, " "s);
  os << (details.isFunction() ? "function " : "subroutine ");
  os << symbol.name() << '(';
  int n = 0;
  for (const auto &dummy : details.dummyArgs()) {
    if (n++ > 0) {
      os << ',';
    }
    os << dummy->name();
  }
  os << ')';
  PutAttrs(os, bindAttrs, details.bindName(), " "s, ""s);
  if (details.isFunction()) {
    const Symbol &result{details.result()};
    if (result.name() != symbol.name()) {
      os << " result(" << result.name() << ')';
    }
  }
  os << '\n';

  // walk symbols, collect ones needed
  ModFileWriter writer{context_};
  std::stringstream typeBindings;
  SubprogramSymbolCollector collector{symbol};
  collector.Collect();
  for (const Symbol *need : collector.symbols()) {
    writer.PutSymbol(typeBindings, need);
  }
  CHECK(typeBindings.str().empty());
  os << writer.uses_.str();
  for (const SourceName &import : collector.imports()) {
    decls_ << "import::" << import << "\n";
  }
  os << writer.decls_.str();
  os << "end\n";
  if (isInterface) {
    os << "end interface\n";
  }
}

static bool IsDefinedOp(const Symbol &symbol) {
  const auto *details{symbol.GetUltimate().detailsIf<GenericDetails>()};
  return details && details->kind() == GenericKind::DefinedOp;
}

static bool IsIntrinsicOp(const Symbol &symbol) {
  if (const auto *details{symbol.GetUltimate().detailsIf<GenericDetails>()}) {
    GenericKind kind{details->kind()};
    return kind >= GenericKind::OpPower && kind <= GenericKind::OpNEQV;
  } else {
    return false;
  }
}

static std::ostream &PutGenericName(std::ostream &os, const Symbol &symbol) {
  if (IsDefinedOp(symbol)) {
    return os << "operator(" << symbol.name() << ')';
  } else {
    return os << symbol.name();
  }
}

void ModFileWriter::PutGeneric(const Symbol &symbol) {
  auto &details{symbol.get<GenericDetails>()};
  PutGenericName(decls_ << "interface ", symbol) << '\n';
  for (auto *specific : details.specificProcs()) {
    decls_ << "procedure::" << specific->name() << '\n';
  }
  decls_ << "end interface\n";
  if (symbol.attrs().test(Attr::PRIVATE)) {
    PutGenericName(decls_ << "private::", symbol) << '\n';
  }
}

void ModFileWriter::PutUse(const Symbol &symbol) {
  auto &details{symbol.get<UseDetails>()};
  auto &use{details.symbol()};
  uses_ << "use " << details.module().name();
  PutGenericName(uses_ << ",only:", symbol);
  // Can have intrinsic op with different local-name and use-name
  // (e.g. `operator(<)` and `operator(.lt.)`) but rename is not allowed
  if (!IsIntrinsicOp(symbol) && use.name() != symbol.name()) {
    PutGenericName(uses_ << "=>", use);
  }
  uses_ << '\n';
  PutUseExtraAttr(Attr::VOLATILE, symbol, use);
  PutUseExtraAttr(Attr::ASYNCHRONOUS, symbol, use);
}

// We have "USE local => use" in this module. If attr was added locally
// (i.e. on local but not on use), also write it out in the mod file.
void ModFileWriter::PutUseExtraAttr(
    Attr attr, const Symbol &local, const Symbol &use) {
  if (local.attrs().test(attr) && !use.attrs().test(attr)) {
    PutAttr(useExtraAttrs_, attr) << "::";
    useExtraAttrs_ << local.name() << '\n';
  }
}

// Collect the symbols of this scope sorted by their original order, not name.
// Namelists are an exception: they are sorted after other symbols.
std::vector<const Symbol *> CollectSymbols(const Scope &scope) {
  std::set<const Symbol *> symbols;  // to prevent duplicates
  std::vector<const Symbol *> sorted;
  std::vector<const Symbol *> namelist;
  std::vector<const Symbol *> common;
  sorted.reserve(scope.size() + scope.commonBlocks().size());
  for (const auto &pair : scope) {
    auto *symbol{pair.second};
    if (!symbol->test(Symbol::Flag::ParentComp) &&
        !symbol->attrs().test(Attr::INTRINSIC)) {
      if (symbols.insert(symbol).second) {
        if (symbol->has<NamelistDetails>()) {
          namelist.push_back(symbol);
        } else {
          sorted.push_back(symbol);
        }
      }
    }
  }
  for (const auto &pair : scope.commonBlocks()) {
    const Symbol *symbol{pair.second};
    if (symbols.insert(symbol).second) {
      common.push_back(symbol);
    }
  }
  // sort normal symbols, then namelists, then common blocks:
  auto compareByOrder = [](const Symbol *x, const Symbol *y) {
    return DEREF(x).name().begin() < DEREF(y).name().begin();
  };
  auto cursor{sorted.begin()};
  std::sort(cursor, sorted.end(), compareByOrder);
  cursor = sorted.insert(sorted.end(), namelist.begin(), namelist.end());
  std::sort(cursor, sorted.end(), compareByOrder);
  cursor = sorted.insert(sorted.end(), common.begin(), common.end());
  std::sort(cursor, sorted.end(), compareByOrder);
  return sorted;
}

void PutEntity(std::ostream &os, const Symbol &symbol) {
  std::visit(
      common::visitors{
          [&](const ObjectEntityDetails &) { PutObjectEntity(os, symbol); },
          [&](const ProcEntityDetails &) { PutProcEntity(os, symbol); },
          [&](const TypeParamDetails &) { PutTypeParam(os, symbol); },
          [&](const auto &) {
            common::die("PutEntity: unexpected details: %s",
                DetailsToString(symbol.details()).c_str());
          },
      },
      symbol.details());
}

void PutShapeSpec(std::ostream &os, const ShapeSpec &x) {
  if (x.lbound().isAssumed()) {
    CHECK(x.ubound().isAssumed());
    os << "..";
  } else {
    if (!x.lbound().isDeferred()) {
      PutBound(os, x.lbound());
    }
    os << ':';
    if (!x.ubound().isDeferred()) {
      PutBound(os, x.ubound());
    }
  }
}
void PutShape(std::ostream &os, const ArraySpec &shape, char open, char close) {
  if (!shape.empty()) {
    os << open;
    bool first{true};
    for (const auto &shapeSpec : shape) {
      if (first) {
        first = false;
      } else {
        os << ',';
      }
      PutShapeSpec(os, shapeSpec);
    }
    os << close;
  }
}

void PutObjectEntity(std::ostream &os, const Symbol &symbol) {
  auto &details{symbol.get<ObjectEntityDetails>()};
  PutEntity(os, symbol, [&]() { PutType(os, DEREF(symbol.GetType())); });
  PutShape(os, details.shape(), '(', ')');
  PutShape(os, details.coshape(), '[', ']');
  PutInit(os, symbol, details.init());
  os << '\n';
}

void PutProcEntity(std::ostream &os, const Symbol &symbol) {
  if (symbol.attrs().test(Attr::INTRINSIC)) {
    return;
  }
  const auto &details{symbol.get<ProcEntityDetails>()};
  const ProcInterface &interface{details.interface()};
  PutEntity(os, symbol, [&]() {
    os << "procedure(";
    if (interface.symbol()) {
      os << interface.symbol()->name();
    } else if (interface.type()) {
      PutType(os, *interface.type());
    }
    os << ')';
    PutPassName(os, details.passName());
  });
  os << '\n';
}

void PutPassName(std::ostream &os, const std::optional<SourceName> &passName) {
  if (passName) {
    os << ",pass(" << *passName << ')';
  }
}

void PutTypeParam(std::ostream &os, const Symbol &symbol) {
  auto &details{symbol.get<TypeParamDetails>()};
  PutEntity(os, symbol, [&]() {
    PutType(os, DEREF(symbol.GetType()));
    PutLower(os << ',', common::EnumToString(details.attr()));
  });
  PutInit(os, details.init());
  os << '\n';
}

void PutInit(std::ostream &os, const Symbol &symbol, const MaybeExpr &init) {
  if (init) {
    if (symbol.attrs().test(Attr::PARAMETER) ||
        symbol.owner().IsDerivedType()) {
      os << (symbol.attrs().test(Attr::POINTER) ? "=>" : "=");
      init->AsFortran(os);
    }
  }
}

void PutInit(std::ostream &os, const MaybeIntExpr &init) {
  if (init) {
    init->AsFortran(os << '=');
  }
}

void PutBound(std::ostream &os, const Bound &x) {
  if (x.isAssumed()) {
    os << '*';
  } else if (x.isDeferred()) {
    os << ':';
  } else {
    x.GetExplicit()->AsFortran(os);
  }
}

// Write an entity (object or procedure) declaration.
// writeType is called to write out the type.
void PutEntity(
    std::ostream &os, const Symbol &symbol, std::function<void()> writeType) {
  writeType();
  MaybeExpr bindName;
  std::visit(
      common::visitors{
          [&](const SubprogramDetails &x) { bindName = x.bindName(); },
          [&](const ObjectEntityDetails &x) { bindName = x.bindName(); },
          [&](const ProcEntityDetails &x) { bindName = x.bindName(); },
          [&](const auto &) {},
      },
      symbol.details());
  PutAttrs(os, symbol.attrs(), bindName);
  os << "::" << symbol.name();
}

// Put out each attribute to os, surrounded by `before` and `after` and
// mapped to lower case.
std::ostream &PutAttrs(std::ostream &os, Attrs attrs, const MaybeExpr &bindName,
    std::string before, std::string after) {
  attrs.set(Attr::PUBLIC, false);  // no need to write PUBLIC
  attrs.set(Attr::EXTERNAL, false);  // no need to write EXTERNAL
  if (bindName) {
    bindName->AsFortran(os << before << "bind(c, name=") << ')' << after;
    attrs.set(Attr::BIND_C, false);
  }
  for (std::size_t i{0}; i < Attr_enumSize; ++i) {
    Attr attr{static_cast<Attr>(i)};
    if (attrs.test(attr)) {
      PutAttr(os << before, attr) << after;
    }
  }
  return os;
}

std::ostream &PutAttr(std::ostream &os, Attr attr) {
  return PutLower(os, AttrToString(attr));
}

std::ostream &PutType(std::ostream &os, const DeclTypeSpec &type) {
  return PutLower(os, type.AsFortran());
}

std::ostream &PutLower(std::ostream &os, const std::string &str) {
  for (char c : str) {
    os << parser::ToLowerCaseLetter(c);
  }
  return os;
}

// Write the module file at path, prepending header. Return false on error.
static bool WriteFile(const std::string &path, std::string &&contents) {
  std::fstream stream;
  auto header{GetHeader(contents)};
  auto size{GetFileSize(path)};
  if (size == header.size() + 1 + contents.size()) {
    // file exists and has the right size, check the contents
    stream.open(path, std::ios::in | std::ios::out);
    if (FileContentsMatch(stream, header, contents)) {
      return true;
    }
    stream.seekp(0);
  } else {
    stream.open(path, std::ios::out);
  }
  stream << header << '\n' << contents;
  stream.close();
  return !stream.fail();
}

// Return true if the stream matches what we would write for the mod file.
static bool FileContentsMatch(std::fstream &stream, const std::string &header,
    const std::string &contents) {
  char c;
  for (std::size_t i{0}; i < header.size(); ++i) {
    if (!stream.get(c) || c != header[i]) {
      return false;
    }
  }
  if (!stream.get(c) || c != '\n') {
    return false;
  }
  for (std::size_t i{0}; i < contents.size(); ++i) {
    if (!stream.get(c) || c != contents[i]) {
      return false;
    }
  }
  return !stream.get(c);
}

// Compute a simple hash of the contents of a module file and
// return it as a string of hex digits.
// This uses the Fowler-Noll-Vo hash function.
template<typename Iter> static std::string CheckSum(Iter begin, Iter end) {
  std::uint64_t hash{0xcbf29ce484222325ull};
  for (auto it{begin}; it != end; ++it) {
    char c{*it};
    hash ^= c & 0xff;
    hash *= 0x100000001b3;
  }
  static const char *digits = "0123456789abcdef";
  std::string result(16, '0');
  for (size_t i{16}; hash != 0; hash >>= 4) {
    result[--i] = digits[hash & 0xf];
  }
  return result;
}

static bool VerifyHeader(const std::string &path) {
  std::fstream stream{path};
  std::string header;
  std::getline(stream, header);
  auto magicLen{strlen(magic)};
  if (header.compare(0, magicLen, magic) != 0) {
    return false;
  }
  std::string expectSum{header.substr(magicLen, 16)};
  std::string actualSum{CheckSum(std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>())};
  return expectSum == actualSum;
}

static std::string GetHeader(const std::string &all) {
  std::stringstream ss;
  ss << magic << CheckSum(all.begin(), all.end());
  return ss.str();
}

static std::size_t GetFileSize(const std::string &path) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) == 0) {
    return static_cast<std::size_t>(statbuf.st_size);
  } else {
    return 0;
  }
}

Scope *ModFileReader::Read(const SourceName &name, Scope *ancestor) {
  std::string ancestorName;  // empty for module
  if (ancestor) {
    if (auto *scope{ancestor->FindSubmodule(name)}) {
      return scope;
    }
    ancestorName = ancestor->GetName().value().ToString();
  } else {
    auto it{context_.globalScope().find(name)};
    if (it != context_.globalScope().end()) {
      return it->second->scope();
    }
  }
  auto path{FindModFile(name, ancestorName)};
  if (!path.has_value()) {
    return nullptr;
  }
  // TODO: We are reading the file once to verify the checksum and then again
  // to parse. Do it only reading the file once.
  if (!VerifyHeader(*path)) {
    context_.Say(name,
        "Module file for '%s' has invalid checksum: %s"_err_en_US, name, *path);
    return nullptr;
  }
  parser::Parsing parsing{context_.allSources()};
  parser::Options options;
  options.isModuleFile = true;
  options.features.Enable(parser::LanguageFeature::BackslashEscapes);
  parsing.Prescan(*path, options);
  parsing.Parse(nullptr);
  auto &parseTree{parsing.parseTree()};
  if (!parsing.messages().empty() || !parsing.consumedWholeFile() ||
      !parseTree.has_value()) {
    context_.Say(
        name, "Module file for '%s' is corrupt: %s"_err_en_US, name, *path);
    return nullptr;
  }
  Scope *parentScope;  // the scope this module/submodule goes into
  if (!ancestor) {
    parentScope = &context_.globalScope();
  } else if (std::optional<SourceName> parent{GetSubmoduleParent(*parseTree)}) {
    parentScope = Read(*parent, ancestor);
  } else {
    parentScope = ancestor;
  }
  // TODO: Check that default kinds of intrinsic types match?
  ResolveNames(context_, *parseTree);
  const auto &it{parentScope->find(name)};
  if (it == parentScope->end()) {
    return nullptr;
  }
  auto &modSymbol{*it->second};
  modSymbol.set(Symbol::Flag::ModFile);
  modSymbol.scope()->set_chars(parsing.cooked());
  return modSymbol.scope();
}

std::optional<std::string> ModFileReader::FindModFile(
    const SourceName &name, const std::string &ancestor) {
  parser::Messages attachments;
  for (auto &dir : context_.searchDirectories()) {
    std::string path{
        ModFilePath(dir, name, ancestor, context_.moduleFileSuffix())};
    std::ifstream ifstream{path};
    if (!ifstream.good()) {
      attachments.Say(name, "%s: %s"_en_US, path, std::strerror(errno));
    } else {
      std::string line;
      std::getline(ifstream, line);
      if (line.compare(0, strlen(magic), magic) == 0) {
        return path;
      }
      attachments.Say(name, "%s: Not a valid module file"_en_US, path);
    }
  }
  auto error{parser::Message{name,
      ancestor.empty()
          ? "Cannot find module file for '%s'"_err_en_US
          : "Cannot find module file for submodule '%s' of module '%s'"_err_en_US,
      name, ancestor}};
  attachments.AttachTo(error);
  context_.Say(std::move(error));
  return std::nullopt;
}

// program was read from a .mod file for a submodule; return the name of the
// submodule's parent submodule, nullptr if none.
static std::optional<SourceName> GetSubmoduleParent(
    const parser::Program &program) {
  CHECK(program.v.size() == 1);
  auto &unit{program.v.front()};
  auto &submod{std::get<common::Indirection<parser::Submodule>>(unit.u)};
  auto &stmt{
      std::get<parser::Statement<parser::SubmoduleStmt>>(submod.value().t)};
  auto &parentId{std::get<parser::ParentIdentifier>(stmt.statement.t)};
  if (auto &parent{std::get<std::optional<parser::Name>>(parentId.t)}) {
    return parent->source;
  } else {
    return std::nullopt;
  }
}

// Construct the path to a module file. ancestorName not empty means submodule.
static std::string ModFilePath(const std::string &dir, const SourceName &name,
    const std::string &ancestorName, const std::string &suffix) {
  std::stringstream path;
  if (dir != "."s) {
    path << dir << '/';
  }
  if (!ancestorName.empty()) {
    path << ancestorName << '-';
  }
  path << name << suffix;
  return path.str();
}

void SubprogramSymbolCollector::Collect() {
  const auto &details{symbol_.get<SubprogramDetails>()};
  isInterface_ = details.isInterface();
  for (const Symbol *dummyArg : details.dummyArgs()) {
    DoSymbol(DEREF(dummyArg));
  }
  if (details.isFunction()) {
    DoSymbol(details.result());
  }
  for (const auto &pair : scope_) {
    const Symbol *symbol{pair.second};
    if (const auto *useDetails{symbol->detailsIf<UseDetails>()}) {
      if (useSet_.count(&useDetails->symbol()) > 0) {
        need_.push_back(symbol);
      }
    }
  }
}

void SubprogramSymbolCollector::DoSymbol(const Symbol &symbol) {
  DoSymbol(symbol.name(), symbol);
}

// Do symbols this one depends on; then add to need_
void SubprogramSymbolCollector::DoSymbol(
    const SourceName &name, const Symbol &symbol) {
  const auto &scope{symbol.owner()};
  if (scope != scope_ && !scope.IsDerivedType()) {
    if (scope != scope_.parent()) {
      useSet_.insert(&symbol);
    }
    if (NeedImport(name, symbol)) {
      imports_.insert(name);
    }
    return;
  }
  if (!needSet_.insert(&symbol).second) {
    return;  // already done
  }
  std::visit(
      common::visitors{
          [this](const ObjectEntityDetails &details) {
            for (const ShapeSpec &spec : details.shape()) {
              DoBound(spec.lbound());
              DoBound(spec.ubound());
            }
            for (const ShapeSpec &spec : details.coshape()) {
              DoBound(spec.lbound());
              DoBound(spec.ubound());
            }
            if (const Symbol * commonBlock{details.commonBlock()}) {
              DoSymbol(*commonBlock);
            }
          },
          [this](const CommonBlockDetails &details) {
            for (const Symbol *object : details.objects()) {
              DoSymbol(*object);
            }
          },
          [](const auto &) {},
      },
      symbol.details());
  if (!symbol.has<UseDetails>()) {
    DoType(symbol.GetType());
  }
  if (!scope.IsDerivedType()) {
    need_.push_back(&symbol);
  }
}

void SubprogramSymbolCollector::DoType(const DeclTypeSpec *type) {
  if (!type) {
    return;
  }
  switch (type->category()) {
  case DeclTypeSpec::Numeric:
  case DeclTypeSpec::Logical: break;  // nothing to do
  case DeclTypeSpec::Character:
    DoParamValue(type->characterTypeSpec().length());
    break;
  default:
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      const auto &typeSymbol{derived->typeSymbol()};
      if (const DerivedTypeSpec * extends{typeSymbol.GetParentTypeSpec()}) {
        DoSymbol(extends->name(), extends->typeSymbol());
      }
      for (const auto pair : derived->parameters()) {
        DoParamValue(pair.second);
      }
      for (const auto pair : *typeSymbol.scope()) {
        const auto &comp{*pair.second};
        DoSymbol(comp);
      }
      DoSymbol(derived->name(), derived->typeSymbol());
    }
  }
}

void SubprogramSymbolCollector::DoBound(const Bound &bound) {
  if (const MaybeSubscriptIntExpr & expr{bound.GetExplicit()}) {
    DoExpr(*expr);
  }
}
void SubprogramSymbolCollector::DoParamValue(const ParamValue &paramValue) {
  if (const auto &expr{paramValue.GetExplicit()}) {
    DoExpr(*expr);
  }
}

// Do we need a IMPORT of this symbol into an interface block?
bool SubprogramSymbolCollector::NeedImport(
    const SourceName &name, const Symbol &symbol) {
  if (!isInterface_) {
    return false;
  } else if (symbol.owner() != scope_.parent()) {
    // detect import from parent of use-associated symbol
    const auto *found{scope_.FindSymbol(name)};
    return DEREF(found).has<UseDetails>() && found->owner() != scope_;
  } else {
    return true;
  }
}

}
