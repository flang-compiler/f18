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

#include "check-omp-clause.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

// 2.5 parallel-clause -> if-clause |
//                        num-threads-clause |
//                        default-clause |
//                        private-clause |
//                        firstprivate-clause |
//                        shared-clause |
//                        copyin-clause |
//                        reduction-clause |
//                        proc-bind-clause
void OmpClauseChecker::Enter(const parser::OmpBlockDirective::Parallel &x) {
  std::set<std::type_index> s{typeid(parser::OmpIfClause),
      typeid(parser::OmpClause::NumThreads), typeid(parser::OmpDefaultClause),
      typeid(parser::OmpClause::Private),
      typeid(parser::OmpClause::Firstprivate),
      typeid(parser::OmpClause::Shared), typeid(parser::OmpClause::Copyin),
      typeid(parser::OmpReductionClause), typeid(parser::OmpProcBindClause)};
  SetAllowed(s);
}

void OmpClauseChecker::Enter(const parser::OmpBlockDirective &x) {
  SetCurrentDirectiveSource(x.source);
}

void OmpClauseChecker::Leave(const parser::OmpEndBlockDirective &x) {
  EmptyAllowed();
  EmptyCurrentDirectiveSource();
  EmptyCurrentClauseSource();
}

void OmpClauseChecker::Enter(const parser::OmpClause &x) {
  SetCurrentClauseSource(x.source);
}

void OmpClauseChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Inbranch &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Mergeable &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Nogroup &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Notinbranch &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Untied &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Collapse &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Copyin &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Copyprivate &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Device &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::DistSchedule &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Final &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::From &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Grainsize &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::NumTasks &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::NumTeams &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::NumThreads &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Priority &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Private &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Safelen &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Simdlen &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::ThreadLimit &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::To &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::Uniform &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpClause::UseDevicePtr &x) {
  CheckAllowed(x);
}

void OmpClauseChecker::Enter(const parser::OmpAlignedClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpDefaultClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpDependClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpIfClause &x) { CheckAllowed(x); }
void OmpClauseChecker::Enter(const parser::OmpLinearClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpMapClause &x) { CheckAllowed(x); }
void OmpClauseChecker::Enter(const parser::OmpProcBindClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpReductionClause &x) {
  CheckAllowed(x);
}
void OmpClauseChecker::Enter(const parser::OmpScheduleClause &x) {
  CheckAllowed(x);
}
}
