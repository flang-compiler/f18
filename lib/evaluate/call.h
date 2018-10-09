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

#ifndef FORTRAN_EVALUATE_CALL_H_
#define FORTRAN_EVALUATE_CALL_H_

#include "common.h"
#include "type.h"
#include "../common/indirection.h"
#include "../parser/char-block.h"
#include <optional>
#include <ostream>
#include <vector>

namespace Fortran::evaluate {

struct ActualArgument {
  explicit ActualArgument(CopyableIndirection<Expr<SomeType>> &&v)
    : value{std::move(v)} {}
  std::optional<DynamicType> GetType() const;
  int Rank() const;
  std::ostream &Dump(std::ostream &) const;

  std::optional<parser::CharBlock> keyword;
  bool isAssumedRank{false};
  bool isAlternateReturn{false};
  std::optional<int> vectorSize;  // TODO
  CopyableIndirection<Expr<SomeType>> value;
};

using Arguments = std::vector<ActualArgument>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_CALL_H_
