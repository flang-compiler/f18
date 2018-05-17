
#include <llvm/ADT/SmallString.h>

#include "target-info-from-clang.h"

namespace Fortran {

// Catch the diagnostics emited while building the clang::TargetInfo.
// For now, this is just a proof of concept so print them on stderr.
// They should eventually be forwarded to our own Fortran DiagnosticConsumer
// once we have one.
class ClangToFortranDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  virtual void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
      const clang::Diagnostic &info) {
    clang::SmallString<256> out;
    info.FormatDiagnostic(out);
    std::cerr << "ERROR: " << out.c_str() << "\n";
  }
};

// Retrieves and sort integer keys from a map.
template<typename T>
static std::vector<int> get_sorted_keys(std::map<int, T> const &input) {
  std::vector<int> output;
  for (auto const &element : input) {
    output.push_back(element.first);
  }
  std::sort(output.begin(), output.end());
  return output;
}

TargetInfoFromClang::TargetInfoFromClang(
    const llvm::Triple &tp /*, options */) {
  clang::DiagnosticOptions diag_opts;
  clang::DiagnosticConsumer *diag_printer =
      new ClangToFortranDiagnosticConsumer();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids;

  auto *diag_eng =
      new clang::DiagnosticsEngine(diag_ids, &diag_opts, diag_printer);

  const auto ctarget_opts = std::make_shared<clang::TargetOptions>();
  ctarget_opts->Triple = tp.str();

  CTarget = clang::TargetInfo::CreateTargetInfo(*diag_eng, ctarget_opts);

  if (!CTarget) {
    std::cerr << "No  clang::TargetInfo\n";
    valid_ = false;
    return;
  }

  valid_ = init();
}

bool TargetInfoFromClang::init() {
  PopulateInteger();
  if (!hasInteger(defaultIntegerKind_)) {
    return false;
  }
  integerKinds_ = get_sorted_keys(integerTypes_);

  // Fill  ISO_C_BINDING constants describing integer kinds.
  // Reminder: The rule is
  //
  //    -1  if the C language defines the type but there
  //        is no equivalent in Fortran
  //
  //    -2  if the C language does not defines the type
  //
  // TODO: Should probably be moved somewhere else
  //
#define MAP_INT(ctype) \
  ((ctype) == clang::TargetInfo::NoInt) \
      ? -2 \
      : findIntegerByWidth(CTarget->getTypeWidth(ctype), -1)

  icb_.c_int = MAP_INT(clang::TargetInfo::SignedInt);
  icb_.c_short = MAP_INT(clang::TargetInfo::SignedShort);
  icb_.c_long = MAP_INT(clang::TargetInfo::SignedLong);
  icb_.c_long_long = MAP_INT(clang::TargetInfo::SignedLongLong);
  icb_.c_signed_char = MAP_INT(clang::TargetInfo::SignedChar);

  icb_.c_int8_t = findIntegerByWidth(8, -1);
  icb_.c_int16_t = findIntegerByWidth(16, -1);
  icb_.c_int32_t = findIntegerByWidth(32, -1);
  icb_.c_int64_t = findIntegerByWidth(64, -1);
  icb_.c_int128_t = CTarget->hasInt128Type() ? findIntegerByWidth(128, -1) : -2;

  icb_.c_int_least8_t = findIntegerByLeastWidth(8, -1);
  icb_.c_int_least16_t = findIntegerByLeastWidth(16, -1);
  icb_.c_int_least32_t = findIntegerByLeastWidth(32, -1);
  icb_.c_int_least64_t = findIntegerByLeastWidth(64, -1);
  icb_.c_int_fast8_t = icb_.c_int8_t;  // TODO
  icb_.c_int_fast16_t = icb_.c_int16_t;  // TODO
  icb_.c_int_fast32_t = icb_.c_int32_t;  // TODO
  icb_.c_int_fast64_t = icb_.c_int64_t;  // TODO
  icb_.c_int_fast128_t = icb_.c_int128_t;  // TODO

  icb_.c_size_t = MAP_INT(CTarget->getSizeType());
  icb_.c_intmax_t = MAP_INT(CTarget->getIntMaxType());
  icb_.c_intptr_t = MAP_INT(CTarget->getIntPtrType());
  icb_.c_ptrdiff_t = MAP_INT(CTarget->getPtrDiffType(0));

#undef MAP_INT

  // Fill INT8, INT16, INT32 and INT64 from ISO_FORTRAN_ENV
  // Reminder: When there is no exact integer type for the
  //           given width, the rule is that -2 shall be
  //           used if a larger integer type exists else
  //           -1  shall be used

  ife_.int8 = findIntegerByLeastWidth(8, -1);
  if (ife_.int8 == -1 && getIntegerWidth(ife_.int8) != 8) {
    ife_.int8 = -2;
  }

  ife_.int16 = findIntegerByLeastWidth(16, -1);
  if (ife_.int16 == -1 && getIntegerWidth(ife_.int16) != 16) {
    ife_.int16 = -2;
  }

  ife_.int32 = findIntegerByLeastWidth(32, -1);
  if (ife_.int32 == -1 && getIntegerWidth(ife_.int32) != 32) {
    ife_.int32 = -2;
  }

  ife_.int64 = findIntegerByLeastWidth(64, -1);
  if (ife_.int64 == -1 && getIntegerWidth(ife_.int64) != 64) {
    ife_.int64 = -2;
  }

  if (ife_.atomic_int_kind == 0) {
    ife_.atomic_int_kind = defaultIntegerKind_;
  }

  PopulateLogical();
  if (!hasLogical(defaultLogicalKind_)) {
    return false;
  }
  logicalKinds_ = get_sorted_keys(logicalTypes_);

  if (ife_.atomic_logical_kind == 0) {
    ife_.atomic_logical_kind = defaultLogicalKind_;
  }

  PopulateReal();
  if (!hasReal(defaultRealKind_)) {
    return false;
  }
  realKinds_ = get_sorted_keys(realTypes_);

  // If not already done, fill REAL32, REAL64 and REAL128 from ISO_FORTRAN_ENV.
  // Reminder: For a non-existing type the rule is to use -2 if there is a
  // larger
  //           type otherwise use -1.
  if (ife_.real32 == 0) {
    ife_.real32 = -1;
  }
  if (ife_.real64 == 0) {
    ife_.real64 = -1;
  }
  if (ife_.real128 == 0) {
    ife_.real128 = -1;
  }
  for (real_k kind : realKinds_) {
    auto *info = getRealInfo(kind);
    if (info->width == 32 && ife_.real32 < 0) {
      ife_.real32 = kind;
    }
    if (info->width == 64 && ife_.real64 < 0) {
      ife_.real64 = kind;
    }
    if (info->width == 128 && ife_.real128 < 0) {
      ife_.real128 = kind;
    }
    if (info->width > 32 && ife_.real32 == -1) {
      ife_.real32 = -2;
    }
    if (info->width > 64 && ife_.real64 == -1) {
      ife_.real64 = -2;
    }
    if (info->width > 128 && ife_.real128 == -1) {
      ife_.real128 = -2;
    }
  }

  PopulateChar();
  if (!hasCharacter(defaultCharacterKind_)) {
    return false;
  }
  characterKinds_ = get_sorted_keys(characterTypes_);

  return true;
}

void TargetInfoFromClang::PopulateInteger() {
  // The default behavior is to have the default Fortran integer
  // match the C 'int' type.
  int default_width = CTarget->getIntWidth();

  // Consider the usual case of INTEGER of kinds 1, 2, 4, 8 and, for
  // some targets, 16.
  int max_int_kind = AllowInteger128() ? 16 : 8;
  for (integer_k kind = 1; kind <= max_int_kind; kind *= 2) {
    int width = kind * 8;
    auto t = CTarget->getIntTypeByWidth(width, true);
    if (t != clang::TargetInfo::NoInt) {
      int align = CTarget->getTypeAlign(t);
      integerTypes_.try_emplace(kind, width, align);
      integerSizeToKind_[kind] = kind;
      if (width == default_width) {
        defaultIntegerKind_ = kind;
      }
    }
  }
}

void TargetInfoFromClang::PopulateLogical() {
  int bool_width = CTarget->getBoolWidth();

  // TODO: Need to take the encoding into account

  // The default behavior is to create a Logical type
  // for each supported Integer type.
  for (int kind : getIntegerKinds()) {
    int width = getIntegerWidth(kind);
    int align = getIntegerAlign(kind);
    logicalTypes_.try_emplace(kind, width, align);
    logicalSizeToKind_[kind] = kind;
    // Fill C_BOOL of ISO_C_BINDING.
    // TODO: Assuming here that the encoding for
    //       the Fortran LOGICAL is compatible with
    //       the bool encoding.
    if (bool_width == width) {
      icb_.c_bool = kind;
    }
  }

  // And use the same kinds (and sizes) for the default
  // logical and integer types.
  defaultLogicalKind_ = defaultIntegerKind_;
}

void TargetInfoFromClang::PopulateReal() {
  // Add the C float type as our default REAL

  unsigned float_width = CTarget->getFloatWidth();
  unsigned float_align = CTarget->getFloatAlign();
  const llvm::fltSemantics *float_format = &CTarget->getFloatFormat();

  real_k float_kind = float_width / 8;
  realTypes_.try_emplace(float_kind, float_width, float_align, float_format);
  icb_.c_float = float_kind;
  defaultRealKind_ = float_kind;

  // Add 'double' type if different from 'float'

  unsigned dbl_width = CTarget->getDoubleWidth();
  unsigned dbl_align = CTarget->getDoubleAlign();
  const llvm::fltSemantics *dbl_format = &CTarget->getDoubleFormat();

  real_k dbl_kind = dbl_width / 8;
  if (dbl_kind != float_kind) {
    realTypes_.try_emplace(dbl_kind, dbl_width, dbl_align, dbl_format);
  }
  icb_.c_double = dbl_kind;
  defaultDoubleKind_ = dbl_kind;

  // Add 'long double' if different from 'float' and 'double'
  unsigned long_dbl_width = CTarget->getLongDoubleWidth();
  unsigned long_dbl_align = CTarget->getLongDoubleAlign();
  const llvm::fltSemantics *long_dbl_format = &CTarget->getLongDoubleFormat();
  real_k long_dbl_kind = long_dbl_width / 8;

  if (long_dbl_kind != float_kind && long_dbl_kind != dbl_kind) {
    realTypes_.try_emplace(
        long_dbl_kind, long_dbl_width, long_dbl_align, long_dbl_format);
  }
  icb_.c_long_double = long_dbl_kind;

  // TODO: How to handle the type __float128 when it exists?
  //       This is a tricky case because x87DoubleExtended, IEEEquad
  //       and PPCDoubleDouble may all be of width 128 (i.e. in x86_64,
  //       x87DoubleExtended is padded from 80 to 128 bits)
  //       That means that we may end up with 'long double' and __float128
  //       being of width 128 but with different formats.
  //       We cannot have two REAL(16) types!
  //
}

void TargetInfoFromClang::PopulateChar() {
  // Default character has kind=1 and shall be 8bit.
  // TODO: add encoding information?
  defaultCharacterKind_ = 1;
  characterTypes_.try_emplace(defaultCharacterKind_, 8, 8);
  // For now, assume that our default character kind is compatible with
  // the C 'char'.
  icb_.c_char = defaultCharacterKind_;
}

}  // namespace Fortran
