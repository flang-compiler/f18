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
//   2. TBD
namespace Fortran::parser {

class CanonicalizationOfOmp {
public:
  template<typename T> bool Pre(T &) { return true; }
  template<typename T> void Post(T &) {}
  CanonicalizationOfOmp(Messages &messages) : messages_{messages} {}

  void Post(Block &block) {
    auto nextIt{block.end()};
    auto expectedEndDirectiveIt{block.end()};
    bool loopMatched{false};
    OpenMPLoopConstruct *matchedLoopConstruct{nullptr};

    for (auto it{block.begin()}, end{block.end()}; it != end; ++it) {
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
      if (auto *exec{std::get_if<ExecutableConstruct>(&it->u)}) {
        // OpenMPLoopConstruct
        if (auto *ompCons{
                std::get_if<common::Indirection<OpenMPConstruct>>(&exec->u)}) {
          if (auto *ompLoop{
                  std::get_if<OpenMPLoopConstruct>(&ompCons->value().u)}) {
            loopMatched = false;
            auto &dir{std::get<OmpLoopDirective>(ompLoop->t)};
            nextIt = it;
            if (++nextIt != end) {
              if (auto *execNext{
                      std::get_if<ExecutableConstruct>(&nextIt->u)}) {
                if (auto *doCons{std::get_if<common::Indirection<DoConstruct>>(
                        &execNext->u)}) {
                  loopMatched = true;
                  matchedLoopConstruct = ompLoop;
                  // move DoConstruct
                  std::get<std::optional<DoConstruct>>(ompLoop->t) =
                      std::move(doCons->value());
                  nextIt = block.erase(nextIt);
                  expectedEndDirectiveIt = nextIt;
                }
              }
            }
            if (!loopMatched) {
              messages_.Say(dir.source,
                  "DO loop is expected after the %s directive"_err_en_US,
                  parser::ToUpperCaseLetters(dir.source.ToString()));
            }
          }
        }

        // OpenMPEndloopdirective (optional)
        if (auto *endDir{
                std::get_if<common::Indirection<OpenMPEndLoopDirective>>(
                    &exec->u)}) {
          if (loopMatched && it == expectedEndDirectiveIt) {
            std::get<std::optional<OpenMPEndLoopDirective>>(
                matchedLoopConstruct->t) = std::move(endDir->value());
            it = block.erase(it);
            --it;
          } else {
            messages_.Say(endDir->value().source,
                "The %s must follow the DO loop associated with the "
                "loop construct"_err_en_US,
                parser::ToUpperCaseLetters(endDir->value().source.ToString()));
          }
          loopMatched = false;
          expectedEndDirectiveIt = end;
        }
      }
    }  // Block list
  }

private:
  Messages &messages_;
};

bool CanonicalizeOmp(Messages &messages, Program &program) {
  CanonicalizationOfOmp omp{messages};
  Walk(program, omp);
  return !messages.AnyFatalError();
}

}
