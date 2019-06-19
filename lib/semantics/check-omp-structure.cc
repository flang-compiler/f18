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

#include "check-omp-structure.h"
#include "../parser/parse-tree.h"
#include <variant>

namespace Fortran::semantics {

bool OmpStructureChecker::HasInvalidCloselyNestedRegion(
    const parser::CharBlock &source, const OmpDirectiveEnumType &set) {
  if (ompContext_.size() && set.test(GetOmpContext().currentDirectiveEnum)) {
    context_.Say(source,
        "A worksharing region may not be closely nested inside a "
        "worksharing, explicit task, taskloop, critical, ordered, atomic, or "
        "master region."_err_en_US);
    return true;
  }
  return false;
}

void OmpStructureChecker::CheckAllowed(const OmpClauseEnum &type) {
  if (ompContext_.empty()) {
    return;
  }
  bool f{true};
  if (!GetOmpContext().allowedClauses.test(type)) {
    context_.Say(GetOmpContext().currentClauseSource,
        "'%s' clause is not allowed on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(type)),
        parser::ToUpperCaseLetters(
            GetOmpContext().currentDirectiveSource.ToString()));
    f = false;
  }
  if (!GetOmpContext().allowedOnceClauses.empty() &&
      GetOmpContext().allowedOnceClauses.test(type) &&
      GetOmpContext().seenClauses.test(type)) {
    context_.Say(GetOmpContext().currentClauseSource,
        "At most one '%s' clause can appear on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(type)),
        parser::ToUpperCaseLetters(
            GetOmpContext().currentDirectiveSource.ToString()));
    f = false;
  }
  if (f) {
    SetOmpContextSeen(type);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  bool f{false};
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  if (parser::ToUpperCaseLetters(dir.source.ToString()) == "DO") {
    f = HasInvalidCloselyNestedRegion(dir.source,
        {OmpDirectiveEnum::DO, OmpDirectiveEnum::SECTIONS,
            OmpDirectiveEnum::SINGLE, OmpDirectiveEnum::WORKSHARE});
  }
  if (!f) {
    OmpContext ct{dir.source};
    ompContext_.push_back(ct);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &x) {
  if (!ompContext_.empty()) {
    ompContext_.pop_back();
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &dir{std::get<parser::OmpBlockDirective>(x.t)};
  OmpContext ct{dir.source};
  ompContext_.push_back(ct);
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &x) {
  if (!ompContext_.empty()) {
    ompContext_.pop_back();
  }
}

// 2.5 parallel-clause -> if-clause |
//                        num-threads-clause |
//                        default-clause |
//                        private-clause |
//                        firstprivate-clause |
//                        shared-clause |
//                        copyin-clause |
//                        reduction-clause |
//                        proc-bind-clause
void OmpStructureChecker::Enter(const parser::OmpBlockDirective::Parallel &x) {
  if (ompContext_.empty()) {
    return;
  }
  // reserve for nesting check
  SetOmpContextDirectiveEnum(OmpDirectiveEnum::PARALLEL);

  OmpClauseEnumType allowed{OmpClauseEnum::IF, OmpClauseEnum::NUM_THREADS,
      OmpClauseEnum::DEFAULT, OmpClauseEnum::PRIVATE,
      OmpClauseEnum::FIRSTPRIVATE, OmpClauseEnum::SHARED, OmpClauseEnum::COPYIN,
      OmpClauseEnum::REDUCTION, OmpClauseEnum::PROC_BIND};
  SetOmpContextAllowed(allowed);

  OmpClauseEnumType allowedOnce{
      OmpClauseEnum::IF, OmpClauseEnum::NUM_THREADS, OmpClauseEnum::PROC_BIND};
  SetOmpContextAllowedOnce(allowedOnce);
}

// 2.7.1 do-clause -> private-clause |
//                    lastprivate-clause |
//                    linear-clause |
//                    reduction-clause |
//                    schedule-clause |
//                    collapse-clause |
//                    ordered-clause
void OmpStructureChecker::Enter(const parser::OmpLoopDirective::Do &x) {
  if (ompContext_.empty()) {
    return;
  }
  // reserve for nesting check
  SetOmpContextDirectiveEnum(OmpDirectiveEnum::DO);

  OmpClauseEnumType allowed{OmpClauseEnum::PRIVATE, OmpClauseEnum::LASTPRIVATE,
      OmpClauseEnum::LINEAR, OmpClauseEnum::REDUCTION, OmpClauseEnum::SCHEDULE,
      OmpClauseEnum::COLLAPSE, OmpClauseEnum::ORDERED};
  SetOmpContextAllowed(allowed);

  OmpClauseEnumType allowedOnce{
      OmpClauseEnum::SCHEDULE, OmpClauseEnum::COLLAPSE, OmpClauseEnum::ORDERED};
  SetOmpContextAllowedOnce(allowedOnce);
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  if (!ompContext_.empty()) {
    SetOmpContextClauseSource(x.source);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowed(OmpClauseEnum::DEFAULTMAP);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Inbranch &x) {
  CheckAllowed(OmpClauseEnum::INBRANCH);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Mergeable &x) {
  CheckAllowed(OmpClauseEnum::MERGEABLE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Nogroup &x) {
  CheckAllowed(OmpClauseEnum::NOGROUP);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Notinbranch &x) {
  CheckAllowed(OmpClauseEnum::NOTINBRANCH);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Untied &x) {
  CheckAllowed(OmpClauseEnum::UNTIED);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Collapse &x) {
  CheckAllowed(OmpClauseEnum::COLLAPSE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &x) {
  CheckAllowed(OmpClauseEnum::COPYIN);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &x) {
  CheckAllowed(OmpClauseEnum::COPYPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Device &x) {
  CheckAllowed(OmpClauseEnum::DEVICE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::DistSchedule &x) {
  CheckAllowed(OmpClauseEnum::DIST_SCHEDULE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Final &x) {
  CheckAllowed(OmpClauseEnum::FINAL);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowed(OmpClauseEnum::FIRSTPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::From &x) {
  CheckAllowed(OmpClauseEnum::FROM);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Grainsize &x) {
  CheckAllowed(OmpClauseEnum::GRAINSIZE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowed(OmpClauseEnum::LASTPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTasks &x) {
  CheckAllowed(OmpClauseEnum::NUM_TASKS);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTeams &x) {
  CheckAllowed(OmpClauseEnum::NUM_TEAMS);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumThreads &x) {
  CheckAllowed(OmpClauseEnum::NUM_THREADS);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowed(OmpClauseEnum::ORDERED);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Priority &x) {
  CheckAllowed(OmpClauseEnum::PRIORITY);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  CheckAllowed(OmpClauseEnum::PRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Safelen &x) {
  CheckAllowed(OmpClauseEnum::SAFELEN);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowed(OmpClauseEnum::SHARED);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Simdlen &x) {
  CheckAllowed(OmpClauseEnum::SIMDLEN);
}
void OmpStructureChecker::Enter(const parser::OmpClause::ThreadLimit &x) {
  CheckAllowed(OmpClauseEnum::THREAD_LIMIT);
}
void OmpStructureChecker::Enter(const parser::OmpClause::To &x) {
  CheckAllowed(OmpClauseEnum::TO);
}
// TODO: 2.10.6 link-clause -> LINK (variable-name-list)
void OmpStructureChecker::Enter(const parser::OmpClause::Uniform &x) {
  CheckAllowed(OmpClauseEnum::UNIFORM);
}
void OmpStructureChecker::Enter(const parser::OmpClause::UseDevicePtr &x) {
  CheckAllowed(OmpClauseEnum::USE_DEVICE_PTR);
}
// TODO: 2.10.4 is-device-ptr-clause -> IS_DEVICE_PTR (variable-name-list)

void OmpStructureChecker::Enter(const parser::OmpAlignedClause &x) {
  CheckAllowed(OmpClauseEnum::ALIGNED);
}
void OmpStructureChecker::Enter(const parser::OmpDefaultClause &x) {
  CheckAllowed(OmpClauseEnum::DEFAULT);
}
void OmpStructureChecker::Enter(const parser::OmpDependClause &x) {
  CheckAllowed(OmpClauseEnum::DEPEND);
}
void OmpStructureChecker::Enter(const parser::OmpIfClause &x) {
  CheckAllowed(OmpClauseEnum::IF);
}
void OmpStructureChecker::Enter(const parser::OmpLinearClause &x) {
  CheckAllowed(OmpClauseEnum::LINEAR);
}
void OmpStructureChecker::Enter(const parser::OmpMapClause &x) {
  CheckAllowed(OmpClauseEnum::MAP);
}
void OmpStructureChecker::Enter(const parser::OmpProcBindClause &x) {
  CheckAllowed(OmpClauseEnum::PROC_BIND);
}
void OmpStructureChecker::Enter(const parser::OmpReductionClause &x) {
  CheckAllowed(OmpClauseEnum::REDUCTION);
}
void OmpStructureChecker::Enter(const parser::OmpScheduleClause &x) {
  CheckAllowed(OmpClauseEnum::SCHEDULE);
}
}
