
#include "target-info.h"
#include "targets/target-info-from-clang.h"

namespace Fortran {

// TargetInfoFromClang can be specialized for architectures with
// specific needs.
// Remark: This is a temporary example.
class TargetInfoNvptx : public TargetInfoFromClang {
public:
  TargetInfoNvptx(const llvm::Triple &tp /*, options */)
    : TargetInfoFromClang(tp) {}

protected:
  bool AllowHalfFloat() override { return false; }
};

// Some vendors may prefer to have full control over the TargetInfo
// implementation.
//
//
class TargetInfoPgi : public TargetInfo {
public:
  TargetInfoPgi(const llvm::Triple &tp /*, options */) {
    // ...
  }

protected:
};

TargetInfo::~TargetInfo() {}

TargetInfo *TargetInfo::Create(const llvm::Triple &tp /*, options */) {
  TargetInfo *target = 0;
  if (tp.getVendorName() == "pgi") {
    // TODO: Allow other vendors can create custom TargetInfo
    target = new TargetInfoPgi(tp);
  } else if (tp.getArch() == llvm::Triple::nvptx ||
      tp.getArch() == llvm::Triple::nvptx64) {
    // ...
    target = new TargetInfoNvptx(tp);
  } else {
    // Use Clang target info to create a sensible one for Fortran
    target = new TargetInfoFromClang(tp);
  }

  if (!target->finalize()) {
    std::cout << target << "\n";
    delete target;
    return nullptr;
  }

  return target;
}

bool TargetInfo::finalize() {
  // Perform consistancy checks and fill missing values when possible

  // TODO: handle the msg. Need an error manager
#define FAIL(msg) \
  do { \
    valid_ = false; \
    return false; \
  } while (0)

#define VERIFY(cond, msg) \
  if (!(cond)) { \
    FAIL(msg); \
  }

  VERIFY(hasInteger(defaultIntegerKind_), "Invalid default integer kind");
  VERIFY(hasInteger(defaultRealKind_), "Invalid default real kind");
  VERIFY(hasInteger(defaultDoubleKind_), "Invalid double precision kind");
  VERIFY(hasInteger(defaultLogicalKind_), "Invalid logical precision kind");

  VERIFY(hasInteger(icb_.c_int), "Invalid value for C_INT");
  VERIFY(hasInteger(icb_.c_short), "Invalid value for C_SHORT");
  VERIFY(hasInteger(icb_.c_long), "Invalid value for C_LONG");
  VERIFY(hasInteger(icb_.c_long_long), "Invalid value for C_LONG_LONG");

  VERIFY(hasInteger(icb_.c_float), "Invalid value for C_FLOAT");
  VERIFY(hasInteger(icb_.c_double), "Invalid value for C_DOUBLE");

  // TODO ...

#undef FAIL
#undef VERIFY

  return valid_;
}

void TargetInfo::Dump(std::ostream &out) const {
#define DUMP_B(member) \
  out << #member << " = " << (member ? "TRUE" : "FALSE") << "\n"
#define DUMP_ICB(member) out << "  " #member << " = " << icb_.member << "\n"
#define DUMP_IFE(member) out << "  " #member << " = " << ife_.member << "\n"

  out << " ========== Dump of Fortran::TargetInfo ============= \n";
  DUMP_B(valid_);

  for (auto elem : integerTypes_) {
    out << "INTEGER(kind=" << elem.first << ") :\n";
    out << "   | width = " << elem.second.width << "\n";
    out << "   | align = " << elem.second.align << "\n";
  }

  for (int size = 1; size <= 16; size *= 2) {
    int kind = findIntegerBySize(size);
    if (kind >= 0) {
      out << "INTEGER*" << size << " is INTEGER(" << kind << ")\n";
    }
  }

  for (auto elem : logicalTypes_) {
    out << "LOGICAL(kind=" << elem.first << ") :\n";
    out << "   | width = " << elem.second.width << "\n";
    out << "   | align = " << elem.second.align << "\n";
  }

  for (int size = 1; size <= 16; size *= 2) {
    int kind = findLogicalBySize(size);
    if (kind >= 0) {
      out << "LOGICAL*" << size << " is LOGICAL(" << kind << ")\n";
    }
  }

  for (auto elem : realTypes_) {
    out << "REAL(kind=" << elem.first << ") :\n";
    out << "   | width = " << elem.second.width << "\n";
    out << "   | align = " << elem.second.align << "\n";
    out << "   | format = ";
    if (elem.second.format == &llvm::APFloat::IEEEhalf())
      out << "IEEEhalf";
    else if (elem.second.format == &llvm::APFloat::IEEEsingle())
      out << "IEEEsingle";
    else if (elem.second.format == &llvm::APFloat::IEEEdouble())
      out << "IEEEdouble";
    else if (elem.second.format == &llvm::APFloat::IEEEquad())
      out << "IEEEquad";
    else if (elem.second.format == &llvm::APFloat::PPCDoubleDouble())
      out << "PPCDoubleDouble";
    else if (elem.second.format == &llvm::APFloat::x87DoubleExtended())
      out << "x87DoubleExtended";
    out << "\n";
  }

  for (int size = 1; size <= 16; size *= 2) {
    int kind = findLogicalBySize(size);
    if (kind >= 0) {
      out << "REAL*" << size << " is REAL(" << kind << ")\n";
    }
  }

  for (auto elem : characterTypes_) {
    out << "CHARACTER(kind=" << elem.first << ") :\n";
    out << "   | width = " << elem.second.width << "\n";
    out << "   | align = " << elem.second.align << "\n";
  }

  out << "Default INTEGER kind = " << getDefaultLogicalKind() << "\n";
  out << "Default REAL kind = " << getDefaultRealKind() << "\n";
  out << "DOUBLE PRECISION kind = " << getDoublePrecisionKind() << "\n";
  out << "Default LOGICAL kind = " << getDefaultLogicalKind() << "\n";
  out << "Default CHARACTER kind = " << getDefaultCharacterKind() << "\n";

  out << "ISO_C_BINDING:\n";
  DUMP_ICB(c_int);
  DUMP_ICB(c_short);
  DUMP_ICB(c_long);
  DUMP_ICB(c_long_long);
  DUMP_ICB(c_signed_char);
  DUMP_ICB(c_size_t);
  DUMP_ICB(c_int8_t);
  DUMP_ICB(c_int16_t);
  DUMP_ICB(c_int32_t);
  DUMP_ICB(c_int64_t);
  DUMP_ICB(c_int128_t);
  DUMP_ICB(c_int_least8_t);
  DUMP_ICB(c_int_least16_t);
  DUMP_ICB(c_int_least32_t);
  DUMP_ICB(c_int_fast8_t);
  DUMP_ICB(c_int_fast16_t);
  DUMP_ICB(c_int_fast32_t);
  DUMP_ICB(c_int_fast64_t);
  DUMP_ICB(c_int_fast128_t);
  DUMP_ICB(c_intmax_t);
  DUMP_ICB(c_intptr_t);
  DUMP_ICB(c_ptrdiff_t);

  DUMP_ICB(c_float);
  DUMP_ICB(c_double);
  DUMP_ICB(c_long_double);
  DUMP_ICB(c_float128);

  DUMP_ICB(c_bool);

  DUMP_ICB(c_char);

  DUMP_ICB(c_null_char);
  DUMP_ICB(c_alert);
  DUMP_ICB(c_backspace);
  DUMP_ICB(c_form_feed);
  DUMP_ICB(c_new_line);
  DUMP_ICB(c_carriage_return);
  DUMP_ICB(c_horizontal_tab);
  DUMP_ICB(c_vertical_tab);

  out << "ISO_FORTRAN_ENV:\n";
  DUMP_IFE(atomic_int_kind);
  DUMP_IFE(atomic_logical_kind);
  DUMP_IFE(character_storage_size);
  DUMP_IFE(file_storage_size);
  DUMP_IFE(numeric_storage_size);
  DUMP_IFE(current_team);
  DUMP_IFE(initial_team);
  DUMP_IFE(parent_team);
  DUMP_IFE(error_unit);
  DUMP_IFE(output_unit);
  DUMP_IFE(input_unit);
  DUMP_IFE(int8);
  DUMP_IFE(int16);
  DUMP_IFE(int32);
  DUMP_IFE(int64);
  DUMP_IFE(real32);
  DUMP_IFE(real64);
  DUMP_IFE(real128);
  DUMP_IFE(iostat_end);
  DUMP_IFE(iostat_eor);
  DUMP_IFE(iostat_inquire_internal_unit);
  DUMP_IFE(stat_failed_image);
  DUMP_IFE(stat_locked);
  DUMP_IFE(stat_locked_other_image);
  DUMP_IFE(stat_stopped_image);
  DUMP_IFE(stat_unlocked);
  DUMP_IFE(stat_unlocked_failed_image);

  out << " ==================================================== \n";
#undef DUMP_B
#undef DUMP_ICB
#undef DUMP_IFE
}

}  // namespace Fortran
