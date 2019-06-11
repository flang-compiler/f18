// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

// OpenMP clause validity check for directives

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_CLAUSE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_CLAUSE_H_

#include "semantics.h"
#include "../parser/parse-tree.h"
#include <typeindex>

namespace Fortran::semantics {

class OmpClauseChecker : public virtual BaseChecker {
public:
  OmpClauseChecker(SemanticsContext &context) : context_{context} {}

  void Enter(const parser::OmpBlockDirective &);
  void Enter(const parser::OmpBlockDirective::Parallel &);
  void Leave(const parser::OmpEndBlockDirective &);

  void Enter(const parser::OmpClause &);
  void Enter(const parser::OmpClause::Defaultmap &);
  void Enter(const parser::OmpClause::Inbranch &);
  void Enter(const parser::OmpClause::Mergeable &);
  void Enter(const parser::OmpClause::Nogroup &);
  void Enter(const parser::OmpClause::Notinbranch &);
  void Enter(const parser::OmpClause::Untied &);
  void Enter(const parser::OmpClause::Collapse &);
  void Enter(const parser::OmpClause::Copyin &);
  void Enter(const parser::OmpClause::Copyprivate &);
  void Enter(const parser::OmpClause::Device &);
  void Enter(const parser::OmpClause::DistSchedule &);
  void Enter(const parser::OmpClause::Final &);
  void Enter(const parser::OmpClause::Firstprivate &);
  void Enter(const parser::OmpClause::From &);
  void Enter(const parser::OmpClause::Grainsize &);
  void Enter(const parser::OmpClause::Lastprivate &);
  void Enter(const parser::OmpClause::NumTasks &);
  void Enter(const parser::OmpClause::NumTeams &);
  void Enter(const parser::OmpClause::NumThreads &);
  void Enter(const parser::OmpClause::Ordered &);
  void Enter(const parser::OmpClause::Priority &);
  void Enter(const parser::OmpClause::Private &);
  void Enter(const parser::OmpClause::Safelen &);
  void Enter(const parser::OmpClause::Shared &);
  void Enter(const parser::OmpClause::Simdlen &);
  void Enter(const parser::OmpClause::ThreadLimit &);
  void Enter(const parser::OmpClause::To &);
  void Enter(const parser::OmpClause::Uniform &);
  void Enter(const parser::OmpClause::UseDevicePtr &);

  void Enter(const parser::OmpAlignedClause &);
  void Enter(const parser::OmpDefaultClause &);
  void Enter(const parser::OmpDependClause &);
  void Enter(const parser::OmpIfClause &);
  void Enter(const parser::OmpLinearClause &);
  void Enter(const parser::OmpMapClause &);
  void Enter(const parser::OmpProcBindClause &);
  void Enter(const parser::OmpReductionClause &);
  void Enter(const parser::OmpScheduleClause &);

  void EmptyAllowed() { allowedClauses_ = {}; }
  void SetAllowed(std::set<std::type_index> allowedClauses) {
    allowedClauses_ = allowedClauses;
  }
  template<typename N> void CheckAllowed(const N &node) {
    if (allowedClauses_.empty()) {
      context_.Say(
          currentClauseSource_, "Internal: allowed clauses not set"_err_en_US);
    }
    auto it{allowedClauses_.find(typeid(node))};
    if (it == allowedClauses_.end()) {
      context_.Say(currentClauseSource_, "'%s' not allowed in %s"_err_en_US,
          parser::ToUpperCaseLetters(currentClauseSource_.ToString()),
          "OMP "s +
              parser::ToUpperCaseLetters(currentDirectiveSource_.ToString()));
    }
  }

  void SetCurrentDirectiveSource(const parser::CharBlock &source) {
    currentDirectiveSource_ = source;
  }
  void EmptyCurrentDirectiveSource() { currentDirectiveSource_ = {}; }

  void SetCurrentClauseSource(const parser::CharBlock &source) {
    currentClauseSource_ = source;
  }
  void EmptyCurrentClauseSource() { currentClauseSource_ = {}; }

private:
  SemanticsContext &context_;
  std::set<std::type_index> allowedClauses_;
  parser::CharBlock currentDirectiveSource_;
  parser::CharBlock currentClauseSource_;
  // TODO: 2.17 Nesting of Regions
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_OMP_CLAUSE_H_
