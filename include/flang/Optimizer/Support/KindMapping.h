//===-- Optimizer/Support/KindMapping.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_SUPPORT_KINDMAPPING_H
#define OPTIMIZER_SUPPORT_KINDMAPPING_H

#include "llvm/IR/Type.h"
#include <map>

namespace llvm {
template <typename>
class Optional;
struct fltSemantics;
} // namespace llvm

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace fir {

/// The kind mapping is an encoded string that informs FIR how the Fortran KIND
/// values from the front-end should be converted to LLVM IR types.  This
/// encoding allows the mapping from front-end KIND values to backend LLVM IR
/// types to be customized by the front-end.
///
/// The provided string uses the following syntax.
///
///   intrinsic-key `:` kind-value (`,` intrinsic-key `:` kind-value)*
///
/// intrinsic-key is a single character for the intrinsic type.
///   'i' : INTEGER   (size in bits)
///   'l' : LOGICAL   (size in bits)
///   'a' : CHARACTER (size in bits)
///   'r' : REAL    (encoding value)
///   'c' : COMPLEX (encoding value)
///
/// kind-value is either an unsigned integer (for 'i', 'l', and 'a') or one of
/// 'Half', 'Float', 'Double', 'X86_FP80', or 'FP128' (for 'r' and 'c').
///
/// If LLVM adds support for new floating-point types, the final list should be
/// extended.
class KindMapping {
public:
  using KindTy = unsigned;
  using Bitsize = unsigned;
  using LLVMTypeID = llvm::Type::TypeID;
  using MatchResult = llvm::Optional<bool>;

  explicit KindMapping(mlir::MLIRContext *context);
  explicit KindMapping(mlir::MLIRContext *context, llvm::StringRef map);

  /// Get the size in bits of !fir.char<kind>
  Bitsize getCharacterBitsize(KindTy kind);

  /// Get the size in bits of !fir.int<kind>
  Bitsize getIntegerBitsize(KindTy kind);

  /// Get the size in bits of !fir.logical<kind>
  Bitsize getLogicalBitsize(KindTy kind);

  /// Get the LLVM Type::TypeID of !fir.real<kind>
  LLVMTypeID getRealTypeID(KindTy kind);

  /// Get the LLVM Type::TypeID of !fir.complex<kind>
  LLVMTypeID getComplexTypeID(KindTy kind);

  mlir::MLIRContext *getContext() const { return context; }

  /// Get the float semantics of !fir.real<kind>
  const llvm::fltSemantics &getFloatSemantics(KindTy kind);

private:
  MatchResult badMapString(llvm::Twine const &ptr);
  MatchResult parse(llvm::StringRef kindMap);

  mlir::MLIRContext *context;
  std::map<char, std::map<KindTy, Bitsize>> intMap;
  std::map<char, std::map<KindTy, LLVMTypeID>> floatMap;
};

} // namespace fir

#endif // OPTIMIZER_SUPPORT_KINDMAPPING_H
