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

namespace Fortran::runtime {

IntrinsicType::IntrinsicType()
  : defaultInteger_{nullptr /* fixed below due to circularity */, {4}, 4},
    genericInteger_{
        Type::Classification::Integer, {{"KIND", defaultInteger_, 4}}, {}},
    genericReal_{
        Type::Classification::Real, {{"KIND", defaultInteger_, 4}}, {}},
    genericComplex_{
        Type::Classification::Complex, {{"KIND", defaultInteger_, 4}}, {}},
    genericCharacter_{Type::Classification::Character,
        {{"KIND", defaultInteger_, 1}}, {{"LEN", defaultInteger_, 1}}},
    genericLogical_{
        Type::Classification::Logical, {{"KIND", defaultInteger_, 4}}, {}},
    kindSpecificInteger_{{&genericInteger_, {1}, 1}, {&genericInteger_, {2}, 2},
        {&genericInteger_, {4}, 4}, {&genericInteger_, {8}, 8}},
    kindSpecificReal_{{&genericReal_, {2}, 2}, {&genericReal_, {4}, 4},
        {&genericReal_, {8}, 8}, {&genericReal_, {10}, 16},
        {&genericReal_, {16}, 16}},
    kindSpecificComplex_{{&genericComplex_, {2}, 4}, {&genericComplex_, {4}, 8},
        {&genericComplex_, {8}, 16}, {&genericComplex_, {10}, 32},
        {&genericComplex_, {16}, 32}},
    kindSpecificCharacter_{{&genericCharacter_, {1}, 1}},
    kindSpecificLogical_{{&genericLogical_, {1}, 1}, {&genericLogical_, {2}, 2},
        {&genericLogical_, {4}, 4}, {&genericLogical_, {8}, 8}} {
  defaultInteger_.set_type(&genericInteger_);
}

const KindSpecificType *IntrinsicType::Find(
    const std::vector<KindSpecificType> &types,
    const std::optional<DefaultInteger> &kind) const {
  std::int64_t k{kind.has_value()
          ? *kind
          : types[0].type().KindParameter(0).defaultValue()};
  for (const auto &t : types) {
    if (t.KindParameterValue(0) == k) {
      return &t;
    }
  }
  return nullptr;
}

}  // namespace Fortran::runtime
