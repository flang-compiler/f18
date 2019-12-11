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

#ifndef FORTRAN_BURNSIDE_MANGLER_H_
#define FORTRAN_BURNSIDE_MANGLER_H_

#include <string>

namespace fir {
struct NameMangler;
}

namespace llvm {
class StringRef;
}

namespace Fortran {
namespace common {
template<typename> class Reference;
}

namespace semantics {
class Symbol;
}

namespace burnside {
using SymbolRef = common::Reference<const semantics::Symbol>;

namespace mangle {

/// Convert a front-end Symbol to an internal name
std::string mangleName(fir::NameMangler &mangler, const SymbolRef symbol);

std::string demangleName(llvm::StringRef name);

}  // mangle
}  // burnside
}  // Fortran

#endif  // FORTRAN_BURNSIDE_MANGLER_H_
