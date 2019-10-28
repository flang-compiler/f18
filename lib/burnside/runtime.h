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

#ifndef FORTRAN_BURNSIDE_RUNTIME_H_
#define FORTRAN_BURNSIDE_RUNTIME_H_

#include <optional>

namespace llvm {
class StringRef;
}
namespace mlir {
class Type;
class FunctionType;
class MLIRContext;
class OpBuilder;
class FuncOp;
}

namespace Fortran::burnside {

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

/// Define a simple static runtime description that different runtime can
/// derived from (e.g io, maths ...).
/// This base class only define enough to generate the functuion declarations,
// it is up to the actual runtime descriptions to define a way to organize these
// descriptions in a meaningful way.
/// It is constexpr constructible so that static tables of such descriptions can
/// be safely stored as global variables without requiring global constructors.
class RuntimeStaticDescription {
public:
  /// Define possible runtime function argument/return type used in signature
  /// descriptions. They follow mlir standard types naming. MLIR types cannot
  /// directly be used because they can only be dynamically built.
  enum TypeCode { i32, i64, f32, f64, c32, c64, IOCookie };
  using MaybeTypeCode = std::optional<TypeCode>;  // for results
  static constexpr MaybeTypeCode voidTy{MaybeTypeCode{std::nullopt}};

  /// C++ does not provide variable size constexpr container yet. TypeVector
  /// implements one for Type elements. It works because Type is an enumeration.
  struct TypeCodeVector {
    template<TypeCode... v> struct Storage {
      static constexpr TypeCode values[]{v...};
    };
    template<TypeCode... v> static constexpr TypeCodeVector create() {
      const TypeCode *start{&Storage<v...>::values[0]};
      return TypeCodeVector{start, start + sizeof...(v)};
    }
    template<> constexpr TypeCodeVector create<>() { return TypeCodeVector{}; }
    const TypeCode *start{nullptr};
    const TypeCode *end{nullptr};
  };
  constexpr RuntimeStaticDescription(
      const char *s, MaybeTypeCode r, TypeCodeVector a)
    : symbol{s}, resultTypeCode{r}, argumentTypeCodes{a} {}
  const char *getSymbol() const { return symbol; }
  /// Conversion between types of the static representation and MLIR types.
  mlir::FunctionType getMLIRFunctionType(mlir::MLIRContext *) const;
  mlir::FuncOp getFuncOp(mlir::OpBuilder &) const;
  static mlir::Type getMLIRType(TypeCode, mlir::MLIRContext *);

private:
  const char *symbol{nullptr};
  MaybeTypeCode resultTypeCode;
  TypeCodeVector argumentTypeCodes;
};

/// StaticMultimapView is a constexpr friendly multimap
/// implementation over sorted constexpr arrays.
/// As the View name suggests, it does not duplicate the
/// sorted array but only brings range and search concepts
/// over it. It provides compile time search and can also
/// provide dynamic search (currently linear, can be improved to
/// log(n) due to the sorted array property).

// TODO: Find a better place for this if this is retained.
// This is currently here because this was designed to provide
// maps over runtime description without the burden of having to
// instantiate these maps dynamically and to keep they somewhere.
template<typename Value> class StaticMultimapView {
public:
  using Key = typename Value::Key;
  struct Range {
    using const_iterator = const Value *;
    constexpr const_iterator begin() const { return startPtr; }
    constexpr const_iterator end() const { return endPtr; }
    constexpr bool empty() const {
      return startPtr == nullptr || endPtr == nullptr || endPtr <= startPtr;
    }
    constexpr std::size_t size() const {
      return empty() ? 0 : static_cast<std::size_t>(endPtr - startPtr);
    }
    const Value *startPtr{nullptr};
    const Value *endPtr{nullptr};
  };
  using const_iterator = typename Range::const_iterator;

  template<std::size_t N>
  constexpr StaticMultimapView(const Value (&array)[N])
    : range{&array[0], &array[0] + N} {}
  template<typename Key> constexpr bool verify() {
    // TODO: sorted
    // non empty increasing pointer direction
    return !range.empty();
  };
  constexpr const_iterator begin() const { return range.begin(); }
  constexpr const_iterator end() const { return range.end(); }

  // Assume array is sorted.
  // TODO make it a log(n) search based on sorted property
  // std::equal_range will be constexpr in C++20 only.
  constexpr Range getRange(const Key &key) const {
    bool matched{false};
    const Value *start{nullptr}, *end{nullptr};
    for (const auto &desc : range) {
      if (desc.key == key) {
        if (!matched) {
          start = &desc;
          matched = true;
        }
      } else if (matched) {
        end = &desc;
        matched = false;
      }
    }
    if (matched) {
      end = range.end();
    }
    return Range{start, end};
  }

  constexpr std::pair<const_iterator, const_iterator> equal_range(
      const Key &key) const {
    Range range{getRange(key)};
    return {range.begin(), range.end()};
  }

  constexpr typename Range::const_iterator find(Key key) const {
    const Range subRange{getRange(key)};
    return subRange.size() == 1 ? subRange.begin() : end();
  }

private:
  Range range{nullptr, nullptr};
};
// TODO get rid of fake runtime below

#define DEFINE_RUNTIME_ENTRY(A, B, C, D) FIRT_##A,
enum RuntimeEntryCode {
#include "runtime.def"
  FIRT_LAST_ENTRY_CODE
};

llvm::StringRef getRuntimeEntryName(RuntimeEntryCode code);

mlir::FunctionType getRuntimeEntryType(
    RuntimeEntryCode code, mlir::MLIRContext &mlirContext, int kind);

mlir::FunctionType getRuntimeEntryType(RuntimeEntryCode code,
    mlir::MLIRContext &mlirContext, int inpKind, int resKind);

}  // Fortran::burnside

#endif  // FORTRAN_BURNSIDE_RUNTIME_H_
