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

#include "intrinsics.h"
#include "expression.h"
#include "type.h"
#include "../common/enum-set.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include <map>
#include <string>
#include <utility>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

using common::TypeCategory;

// This file defines the supported intrinsic procedures and implements
// their recognition and validation.  It is largely table-driven.  See
// documentation/intrinsics.md and section 16 of the Fortran 2018 standard
// for full details on each of the intrinsics.  Be advised, they have
// complicated details, and the design of these tables has to accommodate
// that complexity.

// Dummy arguments to generic intrinsic procedures are each specified by
// their keyword name (rarely used, but always defined), allowable type
// categories, a kind pattern, a rank pattern, and information about
// optionality and defaults.  The kind and rank patterns are represented
// here with code values that are significant to the matching/validation engine.

// These are small bit-sets of type category enumerators.
// Note that typeless (BOZ literal) values don't have a distinct type category.
// These typeless arguments are represented in the tables as if they were
// INTEGER with a special "typeless" kind code.  Arguments of intrinsic types
// that can also be be typeless values are encoded with an "elementalOrBOZ"
// rank pattern.
using CategorySet = common::EnumSet<TypeCategory, 8>;
static constexpr CategorySet Int{TypeCategory::Integer};
static constexpr CategorySet Real{TypeCategory::Real};
static constexpr CategorySet Complex{TypeCategory::Complex};
static constexpr CategorySet Char{TypeCategory::Character};
static constexpr CategorySet Logical{TypeCategory::Logical};
static constexpr CategorySet IntOrReal{Int | Real};
static constexpr CategorySet Floating{Real | Complex};
static constexpr CategorySet Numeric{Int | Real | Complex};
static constexpr CategorySet Relatable{Int | Real | Char};
static constexpr CategorySet IntrinsicType{
    Int | Real | Complex | Char | Logical};
static constexpr CategorySet AnyType{
    IntrinsicType | CategorySet{TypeCategory::Derived}};

enum class KindCode {
  none,
  defaultIntegerKind,
  defaultRealKind,  // is also the default COMPLEX kind
  doublePrecision,
  defaultCharKind,
  defaultLogicalKind,
  any,  // matches any kind value; each instance is independent
  typeless,  // BOZ literals are INTEGER with this kind
  teamType,  // TEAM_TYPE from module ISO_FORTRAN_ENV (for coarrays)
  kindArg,  // this argument is KIND=
  effectiveKind,  // for function results: same "kindArg", possibly defaulted
  dimArg,  // this argument is DIM=
  same,  // match any kind; all "same" kinds must be equal
  likeMultiply,  // for DOT_PRODUCT and MATMUL
};

struct TypePattern {
  CategorySet categorySet;
  KindCode kindCode{KindCode::none};
};

// Abbreviations for argument and result patterns in the intrinsic prototypes:

// Match specific kinds of intrinsic types
static constexpr TypePattern DftInt{Int, KindCode::defaultIntegerKind};
static constexpr TypePattern DftReal{Real, KindCode::defaultRealKind};
static constexpr TypePattern DftComplex{Complex, KindCode::defaultRealKind};
static constexpr TypePattern DftChar{Char, KindCode::defaultCharKind};
static constexpr TypePattern DftLogical{Logical, KindCode::defaultLogicalKind};
static constexpr TypePattern BOZ{Int, KindCode::typeless};
static constexpr TypePattern TEAM_TYPE{Int, KindCode::teamType};
static constexpr TypePattern DoublePrecision{Real, KindCode::doublePrecision};

// Match any kind of some intrinsic or derived types
static constexpr TypePattern AnyInt{Int, KindCode::any};
static constexpr TypePattern AnyReal{Real, KindCode::any};
static constexpr TypePattern AnyIntOrReal{IntOrReal, KindCode::any};
static constexpr TypePattern AnyComplex{Complex, KindCode::any};
static constexpr TypePattern AnyNumeric{Numeric, KindCode::any};
static constexpr TypePattern AnyChar{Char, KindCode::any};
static constexpr TypePattern AnyLogical{Logical, KindCode::any};
static constexpr TypePattern AnyRelatable{Relatable, KindCode::any};
static constexpr TypePattern Anything{AnyType, KindCode::any};

// Match some kind of some intrinsic type(s); all "Same" values must match,
// even when not in the same category (e.g., SameComplex and SameReal).
// Can be used to specify a result so long as at least one argument is
// a "Same".
static constexpr TypePattern SameInt{Int, KindCode::same};
static constexpr TypePattern SameReal{Real, KindCode::same};
static constexpr TypePattern SameIntOrReal{IntOrReal, KindCode::same};
static constexpr TypePattern SameComplex{Complex, KindCode::same};
static constexpr TypePattern SameFloating{Floating, KindCode::same};
static constexpr TypePattern SameNumeric{Numeric, KindCode::same};
static constexpr TypePattern SameChar{Char, KindCode::same};
static constexpr TypePattern SameLogical{Logical, KindCode::same};
static constexpr TypePattern SameRelatable{Relatable, KindCode::same};
static constexpr TypePattern SameIntrinsic{IntrinsicType, KindCode::same};
static constexpr TypePattern SameDerivedType{
    CategorySet{TypeCategory::Derived}, KindCode::same};
static constexpr TypePattern SameType{AnyType, KindCode::same};

// For DOT_PRODUCT and MATMUL, the result type depends on the arguments
static constexpr TypePattern ResultLogical{Logical, KindCode::likeMultiply};
static constexpr TypePattern ResultNumeric{Numeric, KindCode::likeMultiply};

// Result types with known category and KIND=
static constexpr TypePattern KINDInt{Int, KindCode::effectiveKind};
static constexpr TypePattern KINDReal{Real, KindCode::effectiveKind};
static constexpr TypePattern KINDComplex{Complex, KindCode::effectiveKind};
static constexpr TypePattern KINDChar{Char, KindCode::effectiveKind};
static constexpr TypePattern KINDLogical{Logical, KindCode::effectiveKind};

// The default rank pattern for dummy arguments and function results is
// "elemental".
enum class Rank {
  elemental,  // scalar, or array that conforms with other array arguments
  elementalOrBOZ,  // elemental, or typeless BOZ literal scalar
  scalar,
  vector,
  shape,  // INTEGER vector of known length and no negative element
  matrix,
  array,  // not scalar, rank is known and greater than zero
  known,  // rank is known and can be scalar
  anyOrAssumedRank,  // rank can be unknown
  conformable,  // scalar, or array of same rank & shape as "array" argument
  dimReduced,  // scalar if no DIM= argument, else rank(array)-1
  dimRemoved,  // scalar, or rank(array)-1
  rankPlus1,  // rank(known)+1
  shaped,  // rank is length of SHAPE vector
};

enum class Optionality {
  required,
  optional,
  defaultsToSameKind,  // for MatchingDefaultKIND
  defaultsToDefaultForResult,  // for DefaultingKIND
};

struct IntrinsicDummyArgument {
  const char *keyword{nullptr};
  TypePattern typePattern;
  Rank rank{Rank::elemental};
  Optionality optionality{Optionality::required};
};

// constexpr abbreviations for popular arguments:
// DefaultingKIND is a KIND= argument whose default value is the appropriate
// KIND(0), KIND(0.0), KIND(''), &c. value for the function result.
static constexpr IntrinsicDummyArgument DefaultingKIND{"kind",
    {Int, KindCode::kindArg}, Rank::scalar,
    Optionality::defaultsToDefaultForResult};
// MatchingDefaultKIND is a KIND= argument whose default value is the
// kind of any "Same" function argument (viz., the one whose kind pattern is
// "same").
static constexpr IntrinsicDummyArgument MatchingDefaultKIND{"kind",
    {Int, KindCode::kindArg}, Rank::scalar, Optionality::defaultsToSameKind};
static constexpr IntrinsicDummyArgument OptionalDIM{
    "dim", {Int, KindCode::dimArg}, Rank::scalar, Optionality::optional};
static constexpr IntrinsicDummyArgument OptionalMASK{
    "mask", AnyLogical, Rank::conformable, Optionality::optional};

struct IntrinsicInterface {
  static constexpr int maxArguments{7};
  const char *name{nullptr};
  IntrinsicDummyArgument dummy[maxArguments];
  TypePattern result;
  Rank rank{Rank::elemental};
  std::optional<SpecificIntrinsic> Match(const CallCharacteristics &,
      const IntrinsicTypeDefaultKinds &,
      parser::ContextualMessages &messages) const;
};

static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", SameIntOrReal}}, SameIntOrReal},
    {"abs", {{"a", SameComplex}}, SameReal},
    {"achar", {{"i", SameInt}, DefaultingKIND}, KINDChar},
    {"acos", {{"x", SameFloating}}, SameFloating},
    {"acosh", {{"x", SameFloating}}, SameFloating},
    {"adjustl", {{"string", SameChar}}, SameChar},
    {"adjustr", {{"string", SameChar}}, SameChar},
    {"aimag", {{"x", SameComplex}}, SameReal},
    {"aint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"all", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"anint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"any", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"asin", {{"x", SameFloating}}, SameFloating},
    {"asinh", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"y", SameReal}, {"x", SameReal}}, SameReal},
    {"atan2", {{"y", SameReal}, {"x", SameReal}}, SameReal},
    {"atanh", {{"x", SameFloating}}, SameFloating},
    {"bessel_j0", {{"x", SameReal}}, SameReal},
    {"bessel_j1", {{"x", SameReal}}, SameReal},
    {"bessel_jn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_jn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector},
    {"bessel_y0", {{"x", SameReal}}, SameReal},
    {"bessel_y1", {{"x", SameReal}}, SameReal},
    {"bessel_yn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_yn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector},
    {"bge",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DftLogical},
    {"bgt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DftLogical},
    {"ble",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DftLogical},
    {"blt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DftLogical},
    {"btest", {{"i", AnyInt}, {"pos", AnyInt}}, DftLogical},
    {"ceiling", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"char", {{"i", AnyInt}, DefaultingKIND}, KINDChar},
    {"cmplx", {{"x", AnyComplex}, DefaultingKIND}, KINDComplex},
    {"cmplx",
        {{"x", SameIntOrReal, Rank::elementalOrBOZ},
            {"y", SameIntOrReal, Rank::elementalOrBOZ}, DefaultingKIND},
        KINDComplex},
    {"command_argument_count", {}, DftInt, Rank::scalar},
    {"conjg", {{"z", SameComplex}}, SameComplex},
    {"cos", {{"x", SameFloating}}, SameFloating},
    {"cosh", {{"x", SameFloating}}, SameFloating},
    {"count", {{"mask", AnyLogical, Rank::array}, OptionalDIM, DefaultingKIND},
        KINDInt, Rank::dimReduced},
    {"cshift",
        {{"array", SameType, Rank::array}, {"shift", AnyInt, Rank::dimRemoved},
            OptionalDIM},
        SameType, Rank::array},
    {"dim", {{"x", SameIntOrReal}, {"y", SameIntOrReal}}, SameIntOrReal},
    {"dot_product",
        {{"vector_a", AnyLogical, Rank::vector},
            {"vector_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::scalar},
    {"dot_product",
        {{"vector_a", AnyComplex, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar},  // conjugates vector_a
    {"dot_product",
        {{"vector_a", AnyIntOrReal, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar},
    {"dprod", {{"x", DftReal}, {"y", DftReal}}, DoublePrecision},
    {"dshiftl",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftl", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"dshiftr",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftr", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"eoshift",
        {{"array", SameIntrinsic, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameIntrinsic, Rank::dimRemoved,
                Optionality::optional},
            OptionalDIM},
        SameIntrinsic, Rank::array},
    {"eoshift",
        {{"array", SameDerivedType, Rank::array},
            {"shift", AnyInt, Rank::dimRemoved},
            {"boundary", SameDerivedType, Rank::dimRemoved}, OptionalDIM},
        SameDerivedType, Rank::array},
    {"erf", {{"x", SameReal}}, SameReal},
    {"erfc", {{"x", SameReal}}, SameReal},
    {"erfc_scaled", {{"x", SameReal}}, SameReal},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"exponent", {{"x", AnyReal}}, DftInt},
    {"findloc",
        {{"array", SameNumeric, Rank::array},
            {"value", SameNumeric, Rank::scalar}, OptionalDIM, OptionalMASK,
            DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", SameChar, Rank::array}, {"value", SameChar, Rank::scalar},
            OptionalDIM, OptionalMASK, DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, OptionalDIM, OptionalMASK,
            DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"floor", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"fraction", {{"x", SameReal}}, SameReal},
    {"gamma", {{"x", SameReal}}, SameReal},
    {"hypot", {{"x", SameReal}, {"y", SameReal}}, SameReal},
    {"iachar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"iall", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iany", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iparity", {{"array", SameInt, Rank::array}, OptionalDIM, OptionalMASK},
        SameInt, Rank::dimReduced},
    {"iand", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"iand", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ibclr", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ibits", {{"i", SameInt}, {"pos", AnyInt}, {"len", AnyInt}}, SameInt},
    {"ibset", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ichar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"ieor", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ieor", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"image_status",
        {{"image", SameInt},
            {"team", TEAM_TYPE, Rank::scalar, Optionality::optional}},
        DftInt},
    {"index",
        {{"string", SameChar}, {"substring", SameChar},
            {"back", AnyLogical, Rank::scalar, Optionality::optional},
            DefaultingKIND},
        KINDInt},
    {"int", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND}, KINDInt},
    {"ior", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ior", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ishft", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"ishftc",
        {{"i", SameInt}, {"shift", AnyInt},
            {"size", AnyInt, Rank::elemental, Optionality::optional}},
        SameInt},
    {"is_iostat_end", {{"i", AnyInt}}, DftLogical},
    {"is_iostat_eor", {{"i", AnyInt}}, DftLogical},
    {"lbound", {{"array", Anything, Rank::anyOrAssumedRank}, DefaultingKIND},
        KINDInt, Rank::vector},
    {"lbound",
        {{"array", Anything, Rank::anyOrAssumedRank},
            {"dim", {Int, KindCode::dimArg}, Rank::scalar}, DefaultingKIND},
        KINDInt, Rank::scalar},
    {"leadz", {{"i", AnyInt}}, DftInt},
    {"len", {{"string", AnyChar}, DefaultingKIND}, KINDInt},
    {"len_trim", {{"string", AnyChar}, DefaultingKIND}, KINDInt},
    {"lge", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"lgt", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"lle", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"llt", {{"string_a", SameChar}, {"string_b", SameChar}}, DftLogical},
    {"log", {{"x", SameFloating}}, SameFloating},
    {"log10", {{"x", SameReal}}, SameReal},
    {"logical", {{"l", AnyLogical}, DefaultingKIND}, KINDLogical},
    {"log_gamma", {{"x", SameReal}}, SameReal},
    {"matmul",
        {{"array_a", AnyLogical, Rank::vector},
            {"array_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::vector},
    {"matmul",
        {{"array_a", AnyLogical, Rank::matrix},
            {"array_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::vector},
    {"matmul",
        {{"array_a", AnyLogical, Rank::matrix},
            {"array_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::matrix},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::vector},
            {"array_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::vector},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::matrix},
            {"array_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::vector},
    {"matmul",
        {{"array_a", AnyNumeric, Rank::matrix},
            {"array_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::matrix},
    {"maskl", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maskr", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maxloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"maxval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"merge_bits",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"merge_bits",
        {{"i", BOZ}, {"j", SameInt}, {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, OptionalDIM, OptionalMASK,
            DefaultingKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::dimReduced},
    {"minval",
        {{"array", SameRelatable, Rank::array}, OptionalDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced},
    {"mod", {{"a", SameIntOrReal}, {"p", SameIntOrReal}}, SameIntOrReal},
    {"modulo", {{"a", SameIntOrReal}, {"p", SameIntOrReal}}, SameIntOrReal},
    {"nearest", {{"x", SameReal}, {"s", AnyReal}}, SameReal},
    {"nint", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"norm2", {{"x", SameReal, Rank::array}, OptionalDIM}, SameReal,
        Rank::dimReduced},
    {"not", {{"i", SameInt}}, SameInt},
    // pmk WIP continue here in transformationals with NULL
    {"out_of_range",
        {{"x", SameIntOrReal}, {"mold", AnyIntOrReal, Rank::scalar}},
        DftLogical},
    {"out_of_range",
        {{"x", AnyReal}, {"mold", AnyInt, Rank::scalar},
            {"round", AnyLogical, Rank::scalar, Optionality::optional}},
        DftLogical},
    {"out_of_range", {{"x", AnyReal}, {"mold", AnyReal}}, DftLogical},
    {"pack",
        {{"array", SameType, Rank::array},
            {"mask", AnyLogical, Rank::conformable},
            {"vector", SameType, Rank::vector, Optionality::optional}},
        SameType, Rank::vector},
    {"parity", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced},
    {"popcnt", {{"i", AnyInt}}, DftInt},
    {"poppar", {{"i", AnyInt}}, DftInt},
    {"product",
        {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"real", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND},
        KINDReal},
    {"reshape",
        {{"source", SameType, Rank::array}, {"shape", AnyInt, Rank::shape},
            {"pad", SameType, Rank::array, Optionality::optional},
            {"order", AnyInt, Rank::vector, Optionality::optional}},
        SameType, Rank::shaped},
    {"rrspacing", {{"x", SameReal}}, SameReal},
    {"scale", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"scan",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            DefaultingKIND},
        KINDInt},
    {"set_exponent", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"shifta", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftl", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftr", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"sign", {{"a", SameIntOrReal}, {"b", SameIntOrReal}}, SameIntOrReal},
    {"sin", {{"x", SameFloating}}, SameFloating},
    {"sinh", {{"x", SameFloating}}, SameFloating},
    {"size", {{"array", Anything, Rank::anyOrAssumedRank}, DefaultingKIND},
        KINDInt, Rank::vector},
    {"size",
        {{"array", Anything, Rank::anyOrAssumedRank},
            {"dim", {Int, KindCode::dimArg}, Rank::scalar}, DefaultingKIND},
        KINDInt, Rank::scalar},
    {"spacing", {{"x", SameReal}}, SameReal},
    {"spread",
        {{"source", SameType, Rank::known},
            {"dim", {Int, KindCode::dimArg}, Rank::scalar /*not optional*/},
            {"ncopies", AnyInt, Rank::scalar}},
        SameType, Rank::rankPlus1},
    {"sqrt", {{"x", SameFloating}}, SameFloating},
    {"sum", {{"array", SameNumeric, Rank::array}, OptionalDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced},
    {"tan", {{"x", SameFloating}}, SameFloating},
    {"tanh", {{"x", SameFloating}}, SameFloating},
    {"trailz", {{"i", AnyInt}}, DftInt},
    {"transfer",
        {{"source", Anything, Rank::known}, {"mold", SameType, Rank::scalar}},
        SameType, Rank::scalar},
    {"transfer",
        {{"source", Anything, Rank::known}, {"mold", SameType, Rank::array}},
        SameType, Rank::vector},
    {"transfer",
        {{"source", Anything, Rank::known}, {"mold", SameType, Rank::known},
            {"size", AnyInt, Rank::scalar}},
        SameType, Rank::vector},
    {"transpose", {{"matrix", SameType, Rank::matrix}}, SameType, Rank::matrix},
    {"trim", {{"string", AnyChar, Rank::scalar}}, SameChar, Rank::scalar},
    {"ubound", {{"array", Anything, Rank::anyOrAssumedRank}, DefaultingKIND},
        KINDInt, Rank::vector},
    {"ubound",
        {{"array", Anything, Rank::anyOrAssumedRank},
            {"dim", {Int, KindCode::dimArg}, Rank::scalar}, DefaultingKIND},
        KINDInt, Rank::scalar},
    {"unpack",
        {{"vector", SameType, Rank::vector}, {"mask", AnyLogical, Rank::array},
            {"field", SameType, Rank::conformable}},
        SameType, Rank::conformable},
    {"verify",
        {{"string", SameChar}, {"set", SameChar},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            DefaultingKIND},
        KINDInt},
};

// Not covered by the table above:
// MAX, MIN, MERGE

struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
};

static const SpecificIntrinsicInterface specificIntrinsicFunction[]{
    {{"abs", {{"a", DftReal}}, DftReal}},
    {{"acos", {{"x", DftReal}}, DftReal}},
    {{"aimag", {{"z", DftComplex}}, DftReal}},
    {{"aint", {{"a", DftReal}}, DftReal}},
    {{"alog", {{"x", DftReal}}, DftReal}, "log"},
    {{"alog10", {{"x", DftReal}}, DftReal}, "log10"},
    {{"amod", {{"a", DftReal}, {"p", DftReal}}, DftReal}, "mod"},
    {{"anint", {{"a", DftReal}}, DftReal}},
    {{"asin", {{"x", DftReal}}, DftReal}},
    {{"atan", {{"x", DftReal}}, DftReal}},
    {{"atan2", {{"y", DftReal}, {"x", DftReal}}, DftReal}},
    {{"cabs", {{"a", DftComplex}}, DftReal}, "abs"},
    {{"ccos", {{"a", DftComplex}}, DftComplex}, "cos"},
    {{"cexp", {{"a", DftComplex}}, DftComplex}, "exp"},
    {{"clog", {{"a", DftComplex}}, DftComplex}, "log"},
    {{"conjg", {{"a", DftComplex}}, DftComplex}},
    {{"cos", {{"x", DftReal}}, DftReal}},
    {{"csin", {{"a", DftComplex}}, DftComplex}, "sin"},
    {{"csqrt", {{"a", DftComplex}}, DftComplex}, "sqrt"},
    {{"ctan", {{"a", DftComplex}}, DftComplex}, "tan"},
    {{"dabs", {{"a", DoublePrecision}}, DoublePrecision}, "abs"},
    {{"dacos", {{"x", DoublePrecision}}, DoublePrecision}, "acos"},
    {{"dasin", {{"x", DoublePrecision}}, DoublePrecision}, "asin"},
    {{"datan", {{"x", DoublePrecision}}, DoublePrecision}, "atan"},
    {{"datan2", {{"y", DoublePrecision}, {"x", DoublePrecision}},
         DoublePrecision},
        "atan2"},
    {{"dble", {{"a", DftReal}, DefaultingKIND}, DoublePrecision}, "real"},
    {{"dcos", {{"x", DoublePrecision}}, DoublePrecision}, "cos"},
    {{"dcosh", {{"x", DoublePrecision}}, DoublePrecision}, "cosh"},
    {{"ddim", {{"x", DoublePrecision}, {"y", DoublePrecision}},
         DoublePrecision},
        "dim"},
    {{"dexp", {{"x", DoublePrecision}}, DoublePrecision}, "exp"},
    {{"dim", {{"x", DftReal}, {"y", DftReal}}, DftReal}},
    {{"dint", {{"a", DoublePrecision}}, DoublePrecision}, "aint"},
    {{"dlog", {{"x", DoublePrecision}}, DoublePrecision}, "log"},
    {{"dlog10", {{"x", DoublePrecision}}, DoublePrecision}, "log10"},
    {{"dmod", {{"a", DoublePrecision}, {"p", DoublePrecision}},
         DoublePrecision},
        "mod"},
    {{"dnint", {{"a", DoublePrecision}}, DoublePrecision}, "anint"},
    {{"dprod", {{"x", DftReal}, {"y", DftReal}}, DoublePrecision}},
    {{"dsign", {{"a", DoublePrecision}, {"b", DoublePrecision}},
         DoublePrecision},
        "sign"},
    {{"dsin", {{"x", DoublePrecision}}, DoublePrecision}, "sin"},
    {{"dsinh", {{"x", DoublePrecision}}, DoublePrecision}, "sinh"},
    {{"dsqrt", {{"x", DoublePrecision}}, DoublePrecision}, "sqrt"},
    {{"dtan", {{"x", DoublePrecision}}, DoublePrecision}, "tan"},
    {{"dtanh", {{"x", DoublePrecision}}, DoublePrecision}, "tanh"},
    {{"exp", {{"x", DftReal}}, DftReal}},
    {{"float", {{"i", DftInt}}, DftReal}, "real"},
    {{"iabs", {{"a", DftInt}}, DftInt}, "abs"},
    {{"idim", {{"x", DftInt}, {"y", DftInt}}, DftInt}, "dim"},
    {{"idint", {{"a", DoublePrecision}}, DftInt}, "int"},
    {{"idnint", {{"a", DoublePrecision}}, DftInt}, "nint"},
    {{"ifix", {{"a", DftReal}}, DftInt}, "int"},
    {{"index", {{"string", DftChar}, {"substring", DftChar}}, DftInt}},
    {{"isign", {{"a", DftInt}, {"b", DftInt}}, DftInt}, "sign"},
    {{"len", {{"string", DftChar}}, DftInt}},
    {{"log", {{"x", DftReal}}, DftReal}},
    {{"log10", {{"x", DftReal}}, DftReal}},
    {{"mod", {{"a", DftInt}, {"p", DftInt}}, DftInt}},
    {{"nint", {{"a", DftReal}}, DftInt}},
    {{"sign", {{"a", DftReal}, {"b", DftReal}}, DftReal}},
    {{"sin", {{"x", DftReal}}, DftReal}},
    {{"sinh", {{"x", DftReal}}, DftReal}},
    {{"sngl", {{"a", DoublePrecision}}, DftReal}, "real"},
    {{"sqrt", {{"x", DftReal}}, DftReal}},
    {{"tan", {{"x", DftReal}}, DftReal}},
    {{"tanh", {{"x", DftReal}}, DftReal}},
};

// Some entries in the table above are "restricted" specifics:
//   DBLE, FLOAT, IDINT, IFIX, SNGL
// Additional "restricted" specifics not covered by the table above:
//   AMAX0, AMAX1, AMIN0, AMIN1, DMAX1, DMIN1, MAX0, MAX1, MIN0, MIN1

// Intrinsic interface matching against the arguments of a particular
// procedure reference.
std::optional<SpecificIntrinsic> IntrinsicInterface::Match(
    const CallCharacteristics &call, const IntrinsicTypeDefaultKinds &defaults,
    parser::ContextualMessages &messages) const {
  // Attempt to construct a 1-1 correspondence between the dummy arguments in
  // a particular intrinsic procedure's generic interface and the actual
  // arguments in a procedure reference.
  const ActualArgument *actualForDummy[maxArguments];
  int dummies{0};
  for (; dummies < maxArguments && dummy[dummies].keyword != nullptr;
       ++dummies) {
    actualForDummy[dummies] = nullptr;
  }
  for (const ActualArgument &arg : call.argument) {
    if (arg.isAlternateReturn) {
      messages.Say(
          "alternate return specifier not acceptable on call to intrinsic '%s'"_err_en_US,
          call.name.ToString().data());
      return std::nullopt;
    }
    bool found{false};
    for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
      if (actualForDummy[dummyArgIndex] == nullptr) {
        if (!arg.keyword.has_value() ||
            *arg.keyword == dummy[dummyArgIndex].keyword) {
          actualForDummy[dummyArgIndex] = &arg;
          found = true;
          break;
        }
      }
      if (!found) {
        if (arg.keyword.has_value()) {
          messages.Say(*arg.keyword,
              "unknown keyword argument to intrinsic '%'"_err_en_US,
              call.name.ToString().data());
        } else {
          messages.Say("too many actual arguments"_err_en_US);
        }
        return std::nullopt;
      }
    }
  }

  // Check types and kinds of the actual arguments against the intrinsic's
  // interface.  Ensure that two or more arguments that have to have the same
  // type and kind do so.  Check for missing non-optional arguments now, too.
  const ActualArgument *sameArg{nullptr};
  const IntrinsicDummyArgument *kindDummyArg{nullptr};
  const ActualArgument *kindArg{nullptr};
  bool hasDimArg{false};
  for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
    const IntrinsicDummyArgument &d{dummy[dummyArgIndex]};
    if (d.typePattern.kindCode == KindCode::kindArg) {
      CHECK(kindDummyArg == nullptr);
      kindDummyArg = &d;
    }
    const ActualArgument *arg{actualForDummy[dummyArgIndex]};
    if (!arg) {
      if (d.optionality == Optionality::required) {
        messages.Say("missing '%s' argument"_err_en_US, d.keyword);
        return std::nullopt;  // missing non-OPTIONAL argument
      } else {
        continue;
      }
    }
    std::optional<DynamicType> type{arg->GetType()};
    if (!type.has_value()) {
      CHECK(arg->Rank() == 0);
      if (d.typePattern.kindCode == KindCode::typeless ||
          d.rank == Rank::elementalOrBOZ) {
        continue;
      }
      messages.Say("typeless (BOZ) not allowed for '%s'"_err_en_US, d.keyword);
      return std::nullopt;
    } else if (!d.typePattern.categorySet.test(type->category)) {
      messages.Say("actual argument for '%s' has bad type '%s'"_err_en_US,
          d.keyword, type->Dump().data());
      return std::nullopt;  // argument has invalid type category
    }
    bool argOk{false};
    switch (d.typePattern.kindCode) {
    case KindCode::none:
    case KindCode::typeless:
    case KindCode::teamType:  // TODO: TEAM_TYPE
      argOk = false;
      break;
    case KindCode::defaultIntegerKind:
      argOk = type->kind == defaults.defaultIntegerKind;
      break;
    case KindCode::defaultRealKind:
      argOk = type->kind == defaults.defaultRealKind;
      break;
    case KindCode::doublePrecision:
      argOk = type->kind == defaults.defaultDoublePrecisionKind;
      break;
    case KindCode::defaultCharKind:
      argOk = type->kind == defaults.defaultCharacterKind;
      break;
    case KindCode::defaultLogicalKind:
      argOk = type->kind == defaults.defaultLogicalKind;
      break;
    case KindCode::any: argOk = true; break;
    case KindCode::kindArg:
      CHECK(type->category == TypeCategory::Integer);
      CHECK(kindArg == nullptr);
      kindArg = arg;
      argOk = true;
      break;
    case KindCode::dimArg:
      CHECK(type->category == TypeCategory::Integer);
      hasDimArg = true;
      argOk = true;
      break;
    case KindCode::same:
      if (sameArg == nullptr) {
        sameArg = arg;
      }
      argOk = *type == sameArg->GetType();
      break;
    case KindCode::effectiveKind:
      common::die("INTERNAL: KindCode::effectiveKind appears on argument '%s' "
                  "for intrinsic '%s'",
          d.keyword, name);
      break;
    default: CRASH_NO_CASE;
    }
    if (!argOk) {
      messages.Say(
          "actual argument for '%s' has bad type or kind '%s'"_err_en_US,
          d.keyword, type->Dump().data());
      return std::nullopt;
    }
  }

  // Check the ranks of the arguments against the intrinsic's interface.
  const ActualArgument *arrayArg{nullptr};
  const ActualArgument *knownArg{nullptr};
  const ActualArgument *shapeArg{nullptr};
  int elementalRank{0};
  for (int dummyArgIndex{0}; dummyArgIndex < dummies; ++dummyArgIndex) {
    const IntrinsicDummyArgument &d{dummy[dummyArgIndex]};
    if (const ActualArgument * arg{actualForDummy[dummyArgIndex]}) {
      if (arg->isAssumedRank && d.rank != Rank::anyOrAssumedRank) {
        messages.Say(
            "assumed-rank array cannot be used for '%s' argument"_err_en_US,
            d.keyword);
        return std::nullopt;
      }
      int rank{arg->Rank()};
      bool argOk{false};
      switch (d.rank) {
      case Rank::elemental:
      case Rank::elementalOrBOZ:
        if (elementalRank == 0) {
          elementalRank = rank;
        }
        argOk = rank == 0 || rank == elementalRank;
        break;
      case Rank::scalar: argOk = rank == 0; break;
      case Rank::vector: argOk = rank == 1; break;
      case Rank::shape:
        CHECK(shapeArg == nullptr);
        shapeArg = arg;
        argOk = rank == 1 && arg->vectorSize.has_value();
        break;
      case Rank::matrix: argOk = rank == 2; break;
      case Rank::array:
        argOk = rank > 0;
        if (!arrayArg) {
          arrayArg = arg;
        } else {
          argOk &= rank == arrayArg->Rank();
        }
        break;
      case Rank::known:
        CHECK(knownArg == nullptr);
        knownArg = arg;
        argOk = true;
        break;
      case Rank::anyOrAssumedRank: argOk = true; break;
      case Rank::conformable:
        CHECK(arrayArg != nullptr);
        argOk = rank == 0 || rank == arrayArg->Rank();
        break;
      case Rank::dimRemoved:
        CHECK(arrayArg != nullptr);
        if (hasDimArg) {
          argOk = rank + 1 == arrayArg->Rank();
        } else {
          argOk = rank == 0;
        }
        break;
      case Rank::dimReduced:
      case Rank::rankPlus1:
      case Rank::shaped:
        common::die("INTERNAL: result-only rank code appears on argument '%s' "
                    "for intrinsic '%s'",
            d.keyword, name);
      default: CRASH_NO_CASE;
      }
      if (!argOk) {
        messages.Say("'%s' argument has unacceptable rank %d"_err_en_US,
            d.keyword, rank);
        return std::nullopt;
      }
    }
  }

  // Calculate the characteristics of the function result, if any
  if (result.categorySet.empty()) {
    CHECK(result.kindCode == KindCode::none);
    return std::make_optional<SpecificIntrinsic>(name);
  }
  // Determine the result type.
  DynamicType resultType{*result.categorySet.LeastElement(), 0};
  switch (result.kindCode) {
  case KindCode::defaultIntegerKind:
    CHECK(result.categorySet == Int);
    CHECK(resultType.category == TypeCategory::Integer);
    resultType.kind = defaults.defaultIntegerKind;
    break;
  case KindCode::defaultRealKind:
    CHECK(result.categorySet == CategorySet{resultType.category});
    CHECK(Floating.test(resultType.category));
    resultType.kind = defaults.defaultRealKind;
    break;
  case KindCode::doublePrecision:
    CHECK(result.categorySet == Real);
    CHECK(resultType.category == TypeCategory::Real);
    resultType.kind = defaults.defaultDoublePrecisionKind;
    break;
  case KindCode::defaultCharKind:
    CHECK(result.categorySet == Char);
    CHECK(resultType.category == TypeCategory::Character);
    resultType.kind = defaults.defaultCharacterKind;
    break;
  case KindCode::defaultLogicalKind:
    CHECK(result.categorySet == Logical);
    CHECK(resultType.category == TypeCategory::Logical);
    resultType.kind = defaults.defaultLogicalKind;
    break;
  case KindCode::same:
    CHECK(sameArg != nullptr);
    resultType = *sameArg->GetType();
    CHECK(result.categorySet.test(resultType.category));
    break;
  case KindCode::effectiveKind:
    CHECK(kindDummyArg != nullptr);
    CHECK(result.categorySet == CategorySet{resultType.category});
    if (kindArg != nullptr) {
      if (auto *jExpr{std::get_if<Expr<SomeInteger>>(&kindArg->value->u)}) {
        CHECK(jExpr->Rank() == 0);
        if (auto value{jExpr->ScalarValue()}) {
          if (auto code{value->ToInt64()}) {
            if (IsValidKindOfIntrinsicType(resultType.category, *code)) {
              resultType.kind = *code;
              break;
            }
          }
        }
      }
      messages.Say("'kind' argument must be a constant scalar integer "
                   "whose value is a supported kind for the "
                   "intrinsic result type"_err_en_US);
      return std::nullopt;
    } else if (kindDummyArg->optionality == Optionality::defaultsToSameKind) {
      CHECK(sameArg != nullptr);
      resultType = *sameArg->GetType();
    } else {
      CHECK(
          kindDummyArg->optionality == Optionality::defaultsToDefaultForResult);
      resultType.kind = defaults.DefaultKind(resultType.category);
    }
    break;
  case KindCode::likeMultiply:
    CHECK(dummies >= 2);
    CHECK(actualForDummy[0] != nullptr);
    CHECK(actualForDummy[1] != nullptr);
    resultType = actualForDummy[0]->GetType()->ResultTypeForMultiply(
        *actualForDummy[1]->GetType());
    break;
  case KindCode::typeless:
  case KindCode::teamType:
  case KindCode::any:
  case KindCode::kindArg:
  case KindCode::dimArg:
    common::die(
        "INTERNAL: bad KindCode appears on intrinsic '%s' result", name);
    break;
  default: CRASH_NO_CASE;
  }

  // At this point, the call is acceptable.
  // Determine the rank of the function result.
  int resultRank{0};
  switch (rank) {
  case Rank::elemental: resultRank = elementalRank; break;
  case Rank::scalar: resultRank = 0; break;
  case Rank::vector: resultRank = 1; break;
  case Rank::matrix: resultRank = 2; break;
  case Rank::conformable:
    CHECK(arrayArg != nullptr);
    resultRank = arrayArg->Rank();
    break;
  case Rank::dimReduced:
    CHECK(arrayArg != nullptr);
    resultRank = hasDimArg ? arrayArg->Rank() - 1 : 0;
    break;
  case Rank::rankPlus1:
    CHECK(knownArg != nullptr);
    resultRank = knownArg->Rank() + 1;
    break;
  case Rank::shaped:
    CHECK(shapeArg != nullptr);
    CHECK(shapeArg->vectorSize.has_value());
    resultRank = *shapeArg->vectorSize;
    break;
  case Rank::elementalOrBOZ:
  case Rank::shape:
  case Rank::array:
  case Rank::known:
  case Rank::anyOrAssumedRank:
  case Rank::dimRemoved:
    common::die("INTERNAL: bad Rank code on intrinsic '%s' result", name);
    break;
  default: CRASH_NO_CASE;
  }
  CHECK(resultRank >= 0);

  return std::make_optional<SpecificIntrinsic>(
      name, elementalRank > 0, resultType, resultRank);
}

struct IntrinsicProcTable::Implementation {
  explicit Implementation(const IntrinsicTypeDefaultKinds &dfts)
    : defaults{dfts} {
    for (const IntrinsicInterface &f : genericIntrinsicFunction) {
      genericFuncs.insert(std::make_pair(std::string{f.name}, &f));
    }
    for (const SpecificIntrinsicInterface &f : specificIntrinsicFunction) {
      specificFuncs.insert(std::make_pair(std::string{f.name}, &f));
    }
  }

  std::optional<SpecificIntrinsic> Probe(
      const CallCharacteristics &, parser::ContextualMessages *) const;

  IntrinsicTypeDefaultKinds defaults;
  std::multimap<std::string, const IntrinsicInterface *> genericFuncs;
  std::multimap<std::string, const SpecificIntrinsicInterface *> specificFuncs;
};

// Probe the configured intrinsic procedure pattern tables in search of a
// match for a given procedure reference.
std::optional<SpecificIntrinsic> IntrinsicProcTable::Implementation::Probe(
    const CallCharacteristics &call,
    parser::ContextualMessages *messages) const {
  if (call.isSubroutineCall) {
    return std::nullopt;  // TODO
  }
  // A given intrinsic may have multiple patterns in the maps.  If any of them
  // succeeds, the buffered messages from previous failed pattern matches are
  // discarded.  Otherwise, all messages generated by the failing patterns are
  // returned if the caller wants them.
  parser::Messages buffer;
  parser::ContextualMessages errors{
      messages ? messages->at() : call.name, &buffer};
  // Probe the specific intrinsic functions first.
  std::string name{call.name.ToString()};
  auto specificRange{specificFuncs.equal_range(name)};
  for (auto iter{specificRange.first}; iter != specificRange.second; ++iter) {
    if (auto specific{iter->second->Match(call, defaults, errors)}) {
      specific->name = iter->second->generic;
      return specific;
    }
  }
  auto genericRange{specificFuncs.equal_range(name)};
  for (auto iter{genericRange.first}; iter != genericRange.second; ++iter) {
    if (auto specific{iter->second->Match(call, defaults, errors)}) {
      return specific;
    }
  }
  CHECK(!buffer.empty());
  if (messages != nullptr && messages->messages() != nullptr) {
    messages->messages()->Annex(std::move(buffer));
  }
  return std::nullopt;
}

IntrinsicProcTable::~IntrinsicProcTable() {
  // Discard the configured tables.
  delete impl_;
  impl_ = nullptr;
}

IntrinsicProcTable IntrinsicProcTable::Configure(
    const IntrinsicTypeDefaultKinds &defaults) {
  IntrinsicProcTable result;
  result.impl_ = new IntrinsicProcTable::Implementation(defaults);
  return result;
}

std::optional<SpecificIntrinsic> IntrinsicProcTable::Probe(
    const CallCharacteristics &call,
    parser::ContextualMessages *messages) const {
  CHECK(impl_ != nullptr || !"IntrinsicProcTable: not configured");
  return impl_->Probe(call, messages);
}
}  // namespace Fortran::evaluate
