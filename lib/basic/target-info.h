#ifndef FORTRAN_BASIC_TARGET_INFO_H_
#define FORTRAN_BASIC_TARGET_INFO_H_

#include <climits>
#include <iostream>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#include <map>
#include <vector>

namespace Fortran {

class TargetInfo {
public:
  virtual ~TargetInfo();

  // A few type aliases to 'int' in order to clarify the intent.

  typedef int integer_k;  // An int representing a INTEGER kind
  typedef int real_k;  // An int representing a REAL or COMPLEX kind
  typedef int character_k;  // An int representing a CHARACTER kind
  typedef int logical_k;  // An int representing a LOGICAL kind

  typedef int integer_s;  // An int representing a INTEGER size as in INTEGER*4
  typedef int real_s;  // An int representing a REAL size as in REAL*4
  typedef int logical_s;  // An int representing a LOGICAL size as in LOGICAL*4

  // Named constants for module ISO_C_BINDING
  struct IsoCBinding {
    integer_k c_int = 0;
    integer_k c_short = 0;
    integer_k c_long = 0;
    integer_k c_long_long = 0;
    integer_k c_signed_char = 0;
    integer_k c_size_t = 0;
    integer_k c_int8_t = 0;
    integer_k c_int16_t = 0;
    integer_k c_int32_t = 0;
    integer_k c_int64_t = 0;
    integer_k c_int128_t = 0;
    integer_k c_int_least8_t = 0;
    integer_k c_int_least16_t = 0;
    integer_k c_int_least32_t = 0;
    integer_k c_int_least64_t = 0;
    integer_k c_int_fast8_t = 0;
    integer_k c_int_fast16_t = 0;
    integer_k c_int_fast32_t = 0;
    integer_k c_int_fast64_t = 0;
    integer_k c_int_fast128_t = 0;
    integer_k c_intmax_t = 0;
    integer_k c_intptr_t = 0;
    integer_k c_ptrdiff_t = 0;

    real_k c_float = 0;  // also used for C_FLOAT_COMPLEX
    real_k c_double = 0;  // also used for C_DOUBLE_COMPLEX
    real_k c_long_double = 0;  // also used for C_LONG_DOUBLE_COMPLEX
    real_k c_float128 = 0;  // also used for C_FLOAT128_COMPLEX

    logical_k c_bool = 0;

    character_k c_char = 0;

    // Assume that CHARACTER are ASCII by default.
    int c_null_char = 0;
    int c_alert = 7;
    int c_backspace = 8;
    int c_form_feed = 12;
    int c_new_line = 10;
    int c_carriage_return = 13;
    int c_horizontal_tab = 9;
    int c_vertical_tab = 11;
  } icb_;

  // Information for module ISO_FORTRAN_ENV.
  // The kind values are expected to be filled during constructor
  // or during the finalize() call.
  struct IsoFortranEnv {

    integer_k atomic_int_kind = 0;
    logical_k atomic_logical_kind = 0;

    int character_storage_size = IFE_CHARACTER_STORAGE_SIZE;
    int file_storage_size = IFE_FILE_STORAGE_SIZE;
    int numeric_storage_size = IFE_NUMERIC_STORAGE_SIZE;

    int current_team = IFE_CURRENT_TEAM;
    int parent_team = IFE_PARENT_TEAM;
    int initial_team = IFE_INITIAL_TEAM;

    int error_unit = IFE_ERROR_UNIT;
    int output_unit = IFE_OUTPUT_UNIT;
    int input_unit = IFE_INPUT_UNIT;

    integer_k int8 = 0;
    integer_k int16 = 0;
    integer_k int32 = 0;
    integer_k int64 = 0;

    real_k real32 = 0;
    real_k real64 = 0;
    real_k real128 = 0;

    int iostat_end = IFE_IOSTAT_END;
    int iostat_eor = IFE_IOSTAT_EOR;

    int iostat_inquire_internal_unit = IFE_IOSTAT_INQUIRE_INTERNAL_UNIT;
    int stat_failed_image = IFE_STAT_FAILED_IMAGE_no;
    int stat_locked = IFE_STAT_LOCKED;
    int stat_locked_other_image = IFE_STAT_LOCKED_OTHER_IMAGE;
    int stat_stopped_image = IFE_STAT_STOPPED_IMAGE;
    int stat_unlocked = IFE_STAT_UNLOCKED;
    int stat_unlocked_failed_image = IFE_STAT_UNLOCKED_FAILED_IMAGE;

  } ife_;

  int getDefaultLogicalKind() const { return defaultLogicalKind_; }
  int getDefaultIntegerKind() const { return defaultIntegerKind_; }
  int getDefaultCharacterKind() const { return defaultCharacterKind_; }
  int getDefaultRealKind() const { return defaultRealKind_; }
  int getDoublePrecisionKind() const { return defaultDoubleKind_; }

  bool hasInteger(int kind) { return bool(getIntegerInfo(kind)); }
  bool hasLogical(int kind) { return bool(getLogicalInfo(kind)); }
  bool hasReal(int kind) { return bool(getRealInfo(kind)); }
  bool hasCharacter(int kind) { return bool(getCharacterInfo(kind)); }

  int getIntegerWidth(int kind) const {
    if (auto *info = getIntegerInfo(kind)) {
      return info->width;
    } else {
      return 0;
    }
  }

  int getLogicalWidth(int kind) const {
    if (auto *info = getLogicalInfo(kind)) {
      return info->width;
    } else {
      return 0;
    }
  }

  int getRealWidth(int kind) const {
    if (auto *info = getRealInfo(kind)) {
      return info->width;
    } else {
      return 0;
    }
  }

  int getCharacterWidth(int kind) const {
    if (auto *info = getCharacterInfo(kind)) {
      return info->width;
    } else {
      return 0;
    }
  }

  int getIntegerAlign(int kind) {
    if (auto *info = getIntegerInfo(kind)) {
      return info->align;
    } else {
      return 0;
    }
  }

  int getLogicalAlign(int kind) const {
    if (auto *info = getLogicalInfo(kind)) {
      return info->align;
    } else {
      return 0;
    }
  }

  int getRealAlign(int kind) const {
    if (auto *info = getRealInfo(kind)) {
      return info->align;
    } else {
      return 0;
    }
  }

  int getCharacterAlign(int kind) const {
    if (auto *info = getCharacterInfo(kind)) {
      return info->align;
    } else {
      return 0;
    }
  }

  const std::vector<integer_k> &getIntegerKinds() const {
    return integerKinds_;
  }

  const std::vector<logical_k> &getLogicalKinds() const {
    return logicalKinds_;
  }

  const std::vector<character_k> &getCharacterKinds() const {
    return characterKinds_;
  }

  const std::vector<real_k> &getRealKinds() const { return realKinds_; }

  // Provide the INTEGER kind corresponding to INTEGER*size
  //
  // A return value of -1 indicates that INTEGER*size is
  // not supported.
  //
  // Remark: Do not confuse that function with findIntegerByWidth
  // which looks for the integer kind with the specified data size
  // expressed in bits.
  //
  integer_k findIntegerBySize(int size) const {
    auto it = integerSizeToKind_.find(size);
    if (it != integerSizeToKind_.end()) {
      return it->second;
    }
    return -1;
  }

  // Provide the REAL kind corresponding to REAL*size
  real_k findRealBySize(int size) const {
    auto it = realSizeToKind_.find(size);
    if (it != realSizeToKind_.end()) {
      return it->second;
    }
    return -1;
  }

  // Provide the COMPLEX kind corresponding to COMPLEX*size
  real_k findComplexBySize(int size) const {
    auto it = realSizeToKind_.find(size / 2);
    if (it != realSizeToKind_.end()) {
      return it->second;
    }
    return -1;
  }

  // Provide the LOGICAL kind corresponding to LOGICAL*size
  real_k findLogicalBySize(int size) const {
    auto it = logicalSizeToKind_.find(size);
    if (it != logicalSizeToKind_.end()) {
      return it->second;
    }
    return -1;
  }

  integer_k findIntegerByWidth(int width, integer_k fallback = -1) const {
    for (auto elem : integerTypes_) {
      if (elem.second.width == width) {
        return elem.first;
      }
    }
    return fallback;
  }

  // TODO: the name of that function is bad
  integer_k findIntegerByLeastWidth(int width, integer_k fallback = -1) const {
    integer_k kind = fallback;
    int found_width = INT_MAX;
    for (auto elem : integerTypes_) {
      if (elem.second.width >= width && elem.second.width < found_width) {
        kind = elem.first;
        found_width = elem.second.width;
      }
    }
    return kind;
  }

  // Create a Fortran target description
  static TargetInfo *Create(const llvm::Triple &tp /*,options*/);

  bool valid() { return valid_; }

  void Dump(std::ostream &out) const;

  const IsoCBinding &getIsoCBindingValues() { return icb_; }
  const IsoFortranEnv &getIsoFortranEndValues() { return ife_; }

protected:
  TargetInfo(){};

  struct IntegerInfo {
    IntegerInfo(int w, int a) : width(w), align(a) {}
    int width;  // width in bits
    int align;  // alignment in bits
  };

  struct LogicalInfo {
    LogicalInfo(int w, int a) : width(w), align(a) {}
    int width;  // width in bits
    int align;  // alignment in bits
  };

  struct CharacterInfo {
    CharacterInfo(int w, int a) : width(w), align(a) {}
    int width;  // width in bits
    int align;  // alignment in bits
    // TODO: add an enum describing the encoding?
  };

  struct RealInfo {
    RealInfo(int w, int a, const llvm::fltSemantics *f)
      : width(w), align(a), format(f) {}
    int width;  // width in bits
    int align;  // alignment in bits
    const llvm::fltSemantics *format;
  };

  const LogicalInfo *getLogicalInfo(int kind) const {
    auto it = logicalTypes_.find(kind);
    if (it != logicalTypes_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  const IntegerInfo *getIntegerInfo(int kind) const {
    auto it = integerTypes_.find(kind);
    if (it != integerTypes_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  const RealInfo *getRealInfo(int kind) const {
    auto it = realTypes_.find(kind);
    if (it != realTypes_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  const CharacterInfo *getCharacterInfo(int kind) const {
    auto it = characterTypes_.find(kind);
    if (it != characterTypes_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  std::map<integer_k, IntegerInfo> integerTypes_;
  std::map<logical_k, LogicalInfo> logicalTypes_;
  std::map<real_k, RealInfo> realTypes_;
  std::map<character_k, CharacterInfo> characterTypes_;

  // All supported kinds for a given type.
  // The kind values are sorted in increasing order.
  std::vector<integer_k> integerKinds_;
  std::vector<real_k> realKinds_;
  std::vector<logical_k> logicalKinds_;
  std::vector<character_k> characterKinds_;

  //
  // Provide the mapping between the old  'TYPE*size' syntax
  // to the new 'TYPE(kind)' syntax.
  //

  std::map<integer_s, integer_k> integerSizeToKind_;
  std::map<logical_s, logical_k> logicalSizeToKind_;
  std::map<real_s, real_k> realSizeToKind_;

  integer_k defaultIntegerKind_ = 0;
  real_k defaultRealKind_ = 0;
  real_k defaultDoubleKind_ = 0;
  logical_k defaultLogicalKind_ = 0;
  character_k defaultCharacterKind_ = 0;

  // This enum provides some reasonnable values for
  // some fields of the IsoFortranEnv structure below.
  //
  // Beware: Those values are not necessarily identical to the
  //         one used by that the actual target.
  enum {
    IFE_CHARACTER_STORAGE_SIZE = 8,
    IFE_FILE_STORAGE_SIZE = 8,
    IFE_NUMERIC_STORAGE_SIZE = 8,

    IFE_PARENT_TEAM = -1,  // TODO: choose a value
    IFE_CURRENT_TEAM = -2,  // TODO: choose a value
    IFE_INITIAL_TEAM = -3,  // TODO: choose a value

    IFE_IOSTAT_END = -1,
    IFE_IOSTAT_EOR = -2,

    IFE_ERROR_UNIT = 0,
    IFE_INPUT_UNIT = 5,
    IFE_OUTPUT_UNIT = 6,

    // The values for those 'constants' are not precisely
    // defined by the standard but they should all be different.
    // (see 16.10.2.3).
    // Remark: Some values have a sign requirement which is made
    //         explicit here with '+' or '-'
    // Remark: Strangely, the standard does not define a 'stat'
    //         value for success. Is that supposed to be 0?
    IFE_IOSTAT_INQUIRE_INTERNAL_UNIT = 9999,
    // STAT_FAILED_IMAGE may exist in two versions with different
    // constaints.
    IFE_STAT_FAILED_IMAGE_no = -111,
    IFE_STAT_FAILED_IMAGE_yes = +111,
    IFE_STAT_LOCKED = 222,
    IFE_STAT_LOCKED_OTHER_IMAGE = 333,
    IFE_STAT_STOPPED_IMAGE = +444,
    IFE_STAT_UNLOCKED = 555,
    IFE_STAT_UNLOCKED_FAILED_IMAGE = 666,
  };

  bool valid_ = false;

private:
  // Finalize a target.
  // Return true if the target is valid.
  bool finalize(void);

};  // of class TargetInfo

}  // namespace Fortran

#endif  // FORTRAN_BASIC_TARGET_INFO_H_
