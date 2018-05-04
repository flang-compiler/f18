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

#ifndef FORTRAN_PARSER_CHAR_BLOCK_H_
#define FORTRAN_PARSER_CHAR_BLOCK_H_

// Describes a contiguous block of characters; does not own their storage.

#include "interval.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace Fortran::parser {

class CharBlock {
public:
  constexpr CharBlock() {}
  constexpr CharBlock(const char *x, std::size_t n = 1) : interval_{x, n} {}
  constexpr CharBlock(const char *b, const char *ep1)
    : interval_{b, static_cast<std::size_t>(ep1 - b)} {}
  CharBlock(const std::string &s) : interval_{s.data(), s.size()} {}
  constexpr CharBlock(const CharBlock &) = default;
  constexpr CharBlock(CharBlock &&) = default;
  constexpr CharBlock &operator=(const CharBlock &) = default;
  constexpr CharBlock &operator=(CharBlock &&) = default;

  constexpr bool empty() const { return interval_.empty(); }
  constexpr std::size_t size() const { return interval_.size(); }
  constexpr const char *begin() const { return interval_.start(); }
  constexpr const char *end() const {
    return interval_.start() + interval_.size();
  }
  constexpr const char &operator[](std::size_t j) const {
    return interval_.start()[j];
  }

  bool IsBlank() const {
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        return false;
      }
    }
    return true;
  }

  std::string ToString() const {
    return std::string{interval_.start(), interval_.size()};
  }

  // Convert to string, stopping early at any embedded '\0'.
  std::string NULTerminatedToString() const {
    return std::string{interval_.start(),
        /*not in std::*/ strnlen(interval_.start(), interval_.size())};
  }

  bool operator<(const CharBlock &that) const { return Compare(that) < 0; }
  bool operator<=(const CharBlock &that) const { return Compare(that) <= 0; }
  bool operator==(const CharBlock &that) const { return Compare(that) == 0; }
  bool operator!=(const CharBlock &that) const { return Compare(that) != 0; }
  bool operator>=(const CharBlock &that) const { return Compare(that) >= 0; }
  bool operator>(const CharBlock &that) const { return Compare(that) > 0; }

private:
  int Compare(const CharBlock &that) const {
    std::size_t bytes{std::min(size(), that.size())};
    int cmp{std::memcmp(static_cast<const void *>(begin()),
        static_cast<const void *>(that.begin()), bytes)};
    if (cmp != 0) {
      return cmp;
    }
    return size() < that.size() ? -1 : size() > that.size();
  }

  Interval<const char *> interval_{nullptr, 0};
};

}  // namespace Fortran::parser

// Specializations to enable std::unordered_map<CharBlock, ...> &c.
template<> struct std::hash<Fortran::parser::CharBlock> {
  std::size_t operator()(const Fortran::parser::CharBlock &x) const {
    std::size_t hash{0}, bytes{x.size()};
    for (std::size_t j{0}; j < bytes; ++j) {
      hash = (hash * 31) ^ x[j];
    }
    return hash;
  }
};
#endif  // FORTRAN_PARSER_CHAR_BLOCK_H_
