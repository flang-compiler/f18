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

#include "canonicalize-omp.h"
#include "parse-tree-visitor.h"
#include "parse-tree.h"

// After Loop Canonicalization, rewrite OpenMP parse tree to make OpenMP
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. move structured DoConstruct and OpenMPEndloopdirective into
//      OpenMPLoopConstruct. Compilation will not proceed in case of errors
//      after this pass.
//   2. TODO: Start and End directive pair matching
//   3. TBD
namespace Fortran::parser {

class CanonicalizationOfOmp {
public:
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  CanonicalizationOfOmp(Messages &messages) : messages_{messages} {}

  void Post(Block &block) {
    for (auto it{block.begin()}; it != block.end(); ++it) {
      if (auto *exec{std::get_if<ExecutableConstruct>(&it->u)}) {
        if (auto *ompCons{
                std::get_if<common::Indirection<OpenMPConstruct>>(&exec->u)}) {
          // OpenMPLoopConstruct
          if (auto *ompLoop{
                  std::get_if<OpenMPLoopConstruct>(&ompCons->value().u)}) {
            RewriteOpenMPLoopConstruct(ompLoop, block, it);
          }
        } else if (auto *endDir{
                       std::get_if<common::Indirection<OpenMPEndLoopDirective>>(
                           &exec->u)}) {
          // Unmatched OpenMPEndloopdirective
          messages_.Say(endDir->value().source,
              "The %s directive must follow the DO loop associated with the "
              "loop construct"_err_en_US,
              parser::ToUpperCaseLetters(endDir->value().source.ToString()));
        }
      }
    }  // Block list
  }

private:
  void RewriteOpenMPLoopConstruct(
      OpenMPLoopConstruct *x, Block &block, Block::iterator it) {
    // Check the sequence of DoConstruct and OpenMPEndLoopDirective
    // in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpLoopDirective
    //     OmpClauseList
    //   ExecutableConstruct -> DoConstruct
    //   ExecutableConstruct -> OpenMPEndLoopDirective (if available)
    //
    // After rewriting:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpLoopDirective
    //     OmpClauseList
    //     DoConstruct
    //     OpenMPEndLoopDirective (if available)
    Block::iterator nextIt;
    OpenMPLoopConstruct *matchedLoopConstruct{nullptr};

    auto &dir{std::get<OmpLoopDirective>(x->t)};
    nextIt = it;
    if (++nextIt != block.end()) {
      if (auto *execNext{std::get_if<ExecutableConstruct>(&nextIt->u)}) {
        if (auto *doCons{
                std::get_if<common::Indirection<DoConstruct>>(&execNext->u)}) {
          const auto &loopControl{doCons->value().GetLoopControl()};
          if (loopControl.has_value()) {
            matchedLoopConstruct = x;
            // move DoConstruct
            std::get<std::optional<DoConstruct>>(matchedLoopConstruct->t) =
                std::move(doCons->value());
            nextIt = block.erase(nextIt);

            // try to match OpenMPEndLoopDirective
            if (auto *execEnd{std::get_if<ExecutableConstruct>(&nextIt->u)}) {
              if (auto *endDir{
                      std::get_if<common::Indirection<OpenMPEndLoopDirective>>(
                          &execEnd->u)}) {
                std::get<std::optional<OpenMPEndLoopDirective>>(
                    matchedLoopConstruct->t) = std::move(endDir->value());
                nextIt = block.erase(nextIt);
              }
            }
          } else {
            messages_.Say(dir.source,
                "DO loop after the %s directive "
                "must have loop control"_err_en_US,
                parser::ToUpperCaseLetters(dir.source.ToString()));
            return;
          }
        }
      }
    }
    if (!matchedLoopConstruct) {
      messages_.Say(dir.source,
          "DO loop is expected after the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(dir.source.ToString()));
    }
  }

  Messages &messages_;
};

bool CanonicalizeOmp(Messages &messages, Program &program) {
  CanonicalizationOfOmp omp{messages};
  Walk(program, omp);
  return !messages.AnyFatalError();
}
}
