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

#include "../../lib/evaluate/bit-population-count.h"
#include "testing.h"

using Fortran::evaluate::BitPopulationCount;
using Fortran::evaluate::Parity;

int main() {
  MATCH(0, BitPopulationCount(std::uint64_t{0}));
  MATCH(false, Parity(std::uint64_t{0}));
  MATCH(64, BitPopulationCount(~std::uint64_t{0}));
  MATCH(false, Parity(~std::uint64_t{0}));
  for (int j{0}; j < 64; ++j) {
    std::uint64_t x = std::uint64_t{1} << j;
    MATCH(1, BitPopulationCount(x));
    MATCH(true, Parity(x));
    MATCH(63, BitPopulationCount(~x));
    MATCH(true, Parity(~x));
    for (int k{0}; k < j; ++k) {
      std::uint64_t y = x | (std::uint64_t{1} << k);
      MATCH(2, BitPopulationCount(y));
      MATCH(false, Parity(y));
      MATCH(62, BitPopulationCount(~y));
      MATCH(false, Parity(~y));
    }
  }
  MATCH(0, BitPopulationCount(std::uint32_t{0}));
  MATCH(false, Parity(std::uint32_t{0}));
  MATCH(32, BitPopulationCount(~std::uint32_t{0}));
  MATCH(false, Parity(~std::uint32_t{0}));
  for (int j{0}; j < 32; ++j) {
    std::uint32_t x = std::uint32_t{1} << j;
    MATCH(1, BitPopulationCount(x));
    MATCH(true, Parity(x));
    MATCH(31, BitPopulationCount(~x));
    MATCH(true, Parity(~x));
    for (int k{0}; k < j; ++k) {
      std::uint32_t y = x | (std::uint32_t{1} << k);
      MATCH(2, BitPopulationCount(y));
      MATCH(false, Parity(y));
      MATCH(30, BitPopulationCount(~y));
      MATCH(false, Parity(~y));
    }
  }
  MATCH(0, BitPopulationCount(std::uint16_t{0}));
  MATCH(false, Parity(std::uint16_t{0}));
  MATCH(16, BitPopulationCount(static_cast<std::uint16_t>(~0)));
  MATCH(false, Parity(static_cast<std::uint16_t>(~0)));
  for (int j{0}; j < 16; ++j) {
    std::uint16_t x = std::uint16_t{1} << j;
    MATCH(1, BitPopulationCount(x));
    MATCH(true, Parity(x));
    MATCH(15, BitPopulationCount(static_cast<std::uint16_t>(~x)));
    MATCH(true, Parity(static_cast<std::uint16_t>(~x)));
    for (int k{0}; k < j; ++k) {
      std::uint16_t y = x | (std::uint16_t{1} << k);
      MATCH(2, BitPopulationCount(y));
      MATCH(false, Parity(y));
      MATCH(14, BitPopulationCount(static_cast<std::uint16_t>(~y)));
      MATCH(false, Parity(static_cast<std::uint16_t>(~y)));
    }
  }
  MATCH(0, BitPopulationCount(std::uint8_t{0}));
  MATCH(false, Parity(std::uint8_t{0}));
  MATCH(8, BitPopulationCount(static_cast<std::uint8_t>(~0)));
  MATCH(false, Parity(static_cast<std::uint8_t>(~0)));
  for (int j{0}; j < 8; ++j) {
    std::uint8_t x = std::uint8_t{1} << j;
    MATCH(1, BitPopulationCount(x));
    MATCH(true, Parity(x));
    MATCH(7, BitPopulationCount(static_cast<std::uint8_t>(~x)));
    MATCH(true, Parity(static_cast<std::uint8_t>(~x)));
    for (int k{0}; k < j; ++k) {
      std::uint8_t y = x | (std::uint8_t{1} << k);
      MATCH(2, BitPopulationCount(y));
      MATCH(false, Parity(y));
      MATCH(6, BitPopulationCount(static_cast<std::uint8_t>(~y)));
      MATCH(false, Parity(static_cast<std::uint8_t>(~y)));
    }
  }
  return testing::Complete();
}
