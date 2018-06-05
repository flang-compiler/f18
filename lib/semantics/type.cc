// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "type.h"
#include "attr.h"
#include "symbol.h"
#include <iostream>
#include <set>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &o, const IntExpr &x) {
  return x.Output(o);
}
std::ostream &operator<<(std::ostream &o, const IntConst &x) {
  return o << x.value_;
}

std::unordered_map<std::uint64_t, IntConst> IntConst::cache;

std::ostream &operator<<(std::ostream &o, const KindParamValue &x) {
  return o << x.value_;
}

const IntConst &IntConst::Make(std::uint64_t value) {
  auto it = cache.find(value);
  if (it == cache.end()) {
    it = cache.insert({value, IntConst{value}}).first;
  }
  return it->second;
}

const LogicalTypeSpec &LogicalTypeSpec::Make() { return helper.Make(); }
const LogicalTypeSpec &LogicalTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<LogicalTypeSpec> LogicalTypeSpec::helper{"LOGICAL", 0};
std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x) {
  return LogicalTypeSpec::helper.Output(o, x);
}

const IntegerTypeSpec &IntegerTypeSpec::Make() { return helper.Make(); }
const IntegerTypeSpec &IntegerTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<IntegerTypeSpec> IntegerTypeSpec::helper{"INTEGER", 0};
std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x) {
  return IntegerTypeSpec::helper.Output(o, x);
}

const RealTypeSpec &RealTypeSpec::Make() { return helper.Make(); }
const RealTypeSpec &RealTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<RealTypeSpec> RealTypeSpec::helper{"REAL", 0};
std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x) {
  return RealTypeSpec::helper.Output(o, x);
}

const ComplexTypeSpec &ComplexTypeSpec::Make() { return helper.Make(); }
const ComplexTypeSpec &ComplexTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<ComplexTypeSpec> ComplexTypeSpec::helper{"COMPLEX", 0};
std::ostream &operator<<(std::ostream &o, const ComplexTypeSpec &x) {
  return ComplexTypeSpec::helper.Output(o, x);
}

std::ostream &operator<<(std::ostream &o, const CharacterTypeSpec &x) {
  o << "CHARACTER(" << x.len_;
  if (x.kind_ != CharacterTypeSpec::DefaultKind) {
    o << ", " << x.kind_;
  }
  return o << ')';
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeDef &x) {
  o << "TYPE";
  if (!x.data_.attrs.empty()) {
    o << ", " << x.data_.attrs;
  }
  o << " :: " << x.data_.name.ToString();
  if (x.data_.lenParams.size() > 0 || x.data_.kindParams.size() > 0) {
    o << '(';
    int n = 0;
    for (const auto &param : x.data_.lenParams) {
      if (n++) {
        o << ", ";
      }
      o << param.name();
    }
    for (auto param : x.data_.kindParams) {
      if (n++) {
        o << ", ";
      }
      o << param.name();
    }
    o << ')';
  }
  o << '\n';
  for (const auto &param : x.data_.lenParams) {
    o << "  " << param.type() << ", LEN :: " << param.name() << "\n";
  }
  for (const auto &param : x.data_.kindParams) {
    o << "  " << param.type() << ", KIND :: " << param.name() << "\n";
  }
  if (x.data_.Private) {
    o << "  PRIVATE\n";
  }
  if (x.data_.sequence) {
    o << "  SEQUENCE\n";
  }
  for (const auto &comp : x.data_.dataComps) {
    o << "  " << comp << "\n";
  }
  for (const auto &comp : x.data_.procComps) {
    o << "  " << comp << "\n";
  }
  if (x.data_.hasTbpPart()) {
    o << "CONTAINS\n";
    if (x.data_.bindingPrivate) {
      o << "  PRIVATE\n";
    }
    for (const auto &tbp : x.data_.typeBoundProcs) {
      o << "  " << tbp << "\n";
    }
    for (const auto &tbg : x.data_.typeBoundGenerics) {
      o << "  " << tbg << "\n";
    }
    for (const auto &name : x.data_.finalProcs) {
      o << "  FINAL :: " << name.ToString() << '\n';
    }
  }
  return o << "END TYPE";
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << "TYPE(" << x.name_;
  if (!x.paramValues_.empty()) {
    o << '(';
    int n = 0;
    for (const auto &paramValue : x.paramValues_) {
      if (n++) {
        o << ", ";
      }
      if (paramValue.first) {
        o << *paramValue.first << '=';
      }
      o << paramValue.second;
    }
    o << ')';
  }
  o << ')';
  return o;
}

const Bound Bound::ASSUMED{Bound::Assumed};
const Bound Bound::DEFERRED{Bound::Deferred};

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else {
    x.expr_->Output(o);
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const ShapeSpec &x) {
  if (x.lb_.isAssumed()) {
    CHECK(x.ub_.isAssumed());
    o << "..";
  } else {
    if (!x.lb_.isDeferred()) {
      o << x.lb_;
    }
    o << ':';
    if (!x.ub_.isDeferred()) {
      o << x.ub_;
    }
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const DataComponentDef &x) {
  o << x.type_;
  if (!x.attrs_.empty()) {
    o << ", " << x.attrs_;
  }
  o << " :: " << x.name_.ToString();
  if (!x.arraySpec_.empty()) {
    o << '(';
    int n = 0;
    for (ShapeSpec shape : x.arraySpec_) {
      if (n++) {
        o << ", ";
      }
      o << shape;
    }
    o << ')';
  }
  return o;
}

DataComponentDef::DataComponentDef(const DeclTypeSpec &type,
    const SourceName &name, const Attrs &attrs, const ArraySpec &arraySpec)
  : type_{type}, name_{name}, attrs_{attrs}, arraySpec_{arraySpec} {
  attrs.CheckValid({Attr::PUBLIC, Attr::PRIVATE, Attr::ALLOCATABLE,
      Attr::POINTER, Attr::CONTIGUOUS});
  if (attrs.HasAny({Attr::ALLOCATABLE, Attr::POINTER})) {
    for (const auto &shapeSpec : arraySpec) {
      CHECK(shapeSpec.isDeferred());
    }
  } else {
    for (const auto &shapeSpec : arraySpec) {
      CHECK(shapeSpec.isExplicit());
    }
  }
}

DeclTypeSpec::DeclTypeSpec(const DeclTypeSpec &that)
  : category_{that.category_}, intrinsicTypeSpec_{that.intrinsicTypeSpec_} {
  if (category_ == TypeDerived || category_ == ClassDerived) {
    derivedTypeSpec_ =
        std::make_unique<DerivedTypeSpec>(*that.derivedTypeSpec_);
  }
}

DeclTypeSpec &DeclTypeSpec::operator=(const DeclTypeSpec &that) {
  category_ = that.category_;
  intrinsicTypeSpec_ = that.intrinsicTypeSpec_;
  if (category_ == TypeDerived || category_ == ClassDerived) {
    derivedTypeSpec_ =
        std::make_unique<DerivedTypeSpec>(*that.derivedTypeSpec_);
  }
  return *this;
}

DeclTypeSpec::DeclTypeSpec(
    Category category, std::unique_ptr<DerivedTypeSpec> &&typeSpec)
  : category_{category}, intrinsicTypeSpec_{nullptr}, derivedTypeSpec_{
                                                          std::move(typeSpec)} {
  CHECK(category == TypeDerived || category == ClassDerived);
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  // TODO: need CLASS(...) instead of TYPE() for ClassDerived
  switch (x.category_) {
  case DeclTypeSpec::Intrinsic: return x.intrinsicTypeSpec().Output(o);
  case DeclTypeSpec::TypeDerived: return o << x.derivedTypeSpec();
  case DeclTypeSpec::ClassDerived: return o << x.derivedTypeSpec();
  case DeclTypeSpec::TypeStar: return o << "TYPE(*)";
  case DeclTypeSpec::ClassStar: return o << "CLASS(*)";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const ProcDecl &x) {
  return o << x.name_.ToString();
}

ProcComponentDef::ProcComponentDef(
    const ProcDecl &decl, Attrs attrs, ProcInterface &&interface)
  : decl_{decl}, attrs_{attrs}, interface_{std::move(interface)} {
  CHECK(attrs_.test(Attr::POINTER));
  attrs_.CheckValid(
      {Attr::PUBLIC, Attr::PRIVATE, Attr::NOPASS, Attr::POINTER, Attr::PASS});
}
std::ostream &operator<<(std::ostream &o, const ProcComponentDef &x) {
  o << "PROCEDURE(";
  if (auto *symbol = x.interface_.symbol()) {
    o << symbol->name().ToString();
  } else if (auto *type = x.interface_.type()) {
    o << *type;
  }
  o << "), " << x.attrs_ << " :: " << x.decl_;
  return o;
}

std::ostream &operator<<(std::ostream &o, const GenericSpec &x) {
  switch (x.kind()) {
  case GenericSpec::GENERIC_NAME: return o << x.genericName().ToString();
  case GenericSpec::OP_DEFINED:
    return o << '(' << x.definedOp().ToString() << ')';
  case GenericSpec::ASSIGNMENT: return o << "ASSIGNMENT(=)";
  case GenericSpec::READ_FORMATTED: return o << "READ(FORMATTED)";
  case GenericSpec::READ_UNFORMATTED: return o << "READ(UNFORMATTED)";
  case GenericSpec::WRITE_FORMATTED: return o << "WRITE(FORMATTED)";
  case GenericSpec::WRITE_UNFORMATTED: return o << "WRITE(UNFORMATTED)";
  case GenericSpec::OP_ADD: return o << "OPERATOR(+)";
  case GenericSpec::OP_CONCAT: return o << "OPERATOR(//)";
  case GenericSpec::OP_DIVIDE: return o << "OPERATOR(/)";
  case GenericSpec::OP_MULTIPLY: return o << "OPERATOR(*)";
  case GenericSpec::OP_POWER: return o << "OPERATOR(**)";
  case GenericSpec::OP_SUBTRACT: return o << "OPERATOR(-)";
  case GenericSpec::OP_AND: return o << "OPERATOR(.AND.)";
  case GenericSpec::OP_EQ: return o << "OPERATOR(.EQ.)";
  case GenericSpec::OP_EQV: return o << "OPERATOR(.EQV.)";
  case GenericSpec::OP_GE: return o << "OPERATOR(.GE.)";
  case GenericSpec::OP_GT: return o << "OPERATOR(.GT.)";
  case GenericSpec::OP_LE: return o << "OPERATOR(.LE.)";
  case GenericSpec::OP_LT: return o << "OPERATOR(.LT.)";
  case GenericSpec::OP_NE: return o << "OPERATOR(.NE.)";
  case GenericSpec::OP_NEQV: return o << "OPERATOR(.NEQV.)";
  case GenericSpec::OP_NOT: return o << "OPERATOR(.NOT.)";
  case GenericSpec::OP_OR: return o << "OPERATOR(.OR.)";
  case GenericSpec::OP_XOR: return o << "OPERATOR(.XOR.)";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const TypeBoundProc &x) {
  o << "PROCEDURE(";
  if (x.interface_) {
    o << x.interface_->ToString();
  }
  o << ")";
  if (!x.attrs_.empty()) {
    o << ", " << x.attrs_;
  }
  o << " :: " << x.binding_.ToString();
  if (x.procedure_ != x.binding_) {
    o << " => " << x.procedure_.ToString();
  }
  return o;
}
std::ostream &operator<<(std::ostream &o, const TypeBoundGeneric &x) {
  o << "GENERIC ";
  if (!x.attrs_.empty()) {
    o << ", " << x.attrs_;
  }
  o << " :: " << x.genericSpec_ << " => " << x.name_.ToString();
  return o;
}

DerivedTypeDef::DerivedTypeDef(const DerivedTypeDef::Data &data)
  : data_{data} {}

}  // namespace Fortran::semantics
