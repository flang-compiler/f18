//===-- lib/lower/runtime.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_RUNTIME_H_
#define FORTRAN_LOWER_RUNTIME_H_

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
} // namespace mlir

namespace Fortran::lower {

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

/// C++ does not provide variable size constexpr container yet.
/// StaticVector is a class that can be used to hold constexpr data as if it was
/// a vector (i.e, the number of element is not reflected in the
/// container type). This is useful to use in classes that need to be constexpr
/// and where leaking the size as a template type would make it harder to
/// manipulate. It can hold whatever data that can appear as non-type templates
/// (integers, enums, pointer to objects, function pointers...).
/// Example usage:
///
///  enum class Enum {A, B};
///  constexpr StaticVector<Enum> vec{StaticVector::create<Enum::A, Enum::B>()};
///  for (const Enum& code : vec) { /*...*/ }
///

/// This is the class where the constexpr data is "allocated". In fact
/// the data is stored "in" the type. Objects of this type are not meant to
/// be ever constructed.
template <typename T, T... v>
struct StaticVectorStorage {
  static constexpr T values[]{v...};
  static constexpr const T *start{&values[0]};
  static constexpr const T *end{start + sizeof...(v)};
};
template <typename T>
struct StaticVectorStorage<T> {
  static constexpr const T *start{nullptr}, *end{nullptr};
};

/// StaticVector cannot be directly constructed, instead its
/// `create` static method has to be used to create StaticVector objects.
/// StaticVector are views over the StaticVectorStorage type that was built
/// while instantiating the create method. They do not duplicate the values from
/// these read-only storages.
template <typename T>
struct StaticVector {
  template <T... v>
  static constexpr StaticVector create() {
    using storage = StaticVectorStorage<T, v...>;
    return StaticVector{storage::start, storage::end};
  }
  using const_iterator = const T *;
  constexpr const_iterator begin() const { return startPtr; }
  constexpr const_iterator end() const { return endPtr; }
  const T *startPtr{nullptr};
  const T *endPtr{nullptr};
};

/// Define a simple static runtime description that different runtime can
/// derived from (e.g io, maths ...).
/// This base class only define enough to generate the functuion declarations,
/// it is up to the actual runtime descriptions to define a way to organize
/// these descriptions in a meaningful way.
/// It is constexpr constructible so that static tables of such descriptions can
/// be safely stored as global variables without requiring global constructors.

class RuntimeStaticDescription {
public:
  /// Define possible runtime function argument/return type used in signature
  /// descriptions. They follow mlir standard types naming. MLIR types cannot
  /// directly be used because they can only be dynamically built.
  enum TypeCode { i32, i64, f32, f64, c32, c64, IOCookie };
  using MaybeTypeCode = std::optional<TypeCode>; // for results
  using TypeCodeVector = StaticVector<TypeCode>; // for arguments
  static constexpr MaybeTypeCode voidTy{MaybeTypeCode{std::nullopt}};

  constexpr RuntimeStaticDescription(const char *s, MaybeTypeCode r,
                                     TypeCodeVector a)
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
// instantiate these maps dynamically and to keep them somewhere.
template <typename V>
class StaticMultimapView {
public:
  using Key = typename V::Key;
  struct Range {
    using const_iterator = const V *;
    constexpr const_iterator begin() const { return startPtr; }
    constexpr const_iterator end() const { return endPtr; }
    constexpr bool empty() const {
      return startPtr == nullptr || endPtr == nullptr || endPtr <= startPtr;
    }
    constexpr std::size_t size() const {
      return empty() ? 0 : static_cast<std::size_t>(endPtr - startPtr);
    }
    const V *startPtr{nullptr};
    const V *endPtr{nullptr};
  };
  using const_iterator = typename Range::const_iterator;

  template <std::size_t N>
  constexpr StaticMultimapView(const V (&array)[N])
      : range{&array[0], &array[0] + N} {}
  template <typename Key>
  constexpr bool verify() {
    // TODO: sorted
    // non empty increasing pointer direction
    return !range.empty();
  }
  constexpr const_iterator begin() const { return range.begin(); }
  constexpr const_iterator end() const { return range.end(); }

  // Assume array is sorted.
  // TODO make it a log(n) search based on sorted property
  // std::equal_range will be constexpr in C++20 only.
  constexpr Range getRange(const Key &key) const {
    bool matched{false};
    const V *start{nullptr}, *end{nullptr};
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

  constexpr std::pair<const_iterator, const_iterator>
  equal_range(const Key &key) const {
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

mlir::FunctionType getRuntimeEntryType(RuntimeEntryCode code,
                                       mlir::MLIRContext &mlirContext,
                                       int kind);

mlir::FunctionType getRuntimeEntryType(RuntimeEntryCode code,
                                       mlir::MLIRContext &mlirContext,
                                       int inpKind, int resKind);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_RUNTIME_H_
