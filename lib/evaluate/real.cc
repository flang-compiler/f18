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

#include "real.h"
#include "int-power.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
#include <limits>

namespace Fortran::evaluate::value {

template<typename W, int P, bool IM>
Relation Real<W, P, IM>::Compare(const Real &y) const {
  if (IsNotANumber() || y.IsNotANumber()) {  // NaN vs x, x vs NaN
    return Relation::Unordered;
  } else if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (IsNegative()) {  // -Inf vs +/-Inf
        return y.IsNegative() ? Relation::Equal : Relation::Less;
      } else {  // +Inf vs +/-Inf
        return y.IsNegative() ? Relation::Greater : Relation::Equal;
      }
    } else {  // +/-Inf vs finite
      return IsNegative() ? Relation::Less : Relation::Greater;
    }
  } else if (y.IsInfinite()) {  // finite vs +/-Inf
    return y.IsNegative() ? Relation::Greater : Relation::Less;
  } else {  // two finite numbers
    bool isNegative{IsNegative()};
    if (isNegative != y.IsNegative()) {
      if (word_.IOR(y.word_).IBCLR(bits - 1).IsZero()) {
        return Relation::Equal;  // +/-0.0 == -/+0.0
      } else {
        return isNegative ? Relation::Less : Relation::Greater;
      }
    } else {
      // same sign
      Ordering order{evaluate::Compare(Exponent(), y.Exponent())};
      if (order == Ordering::Equal) {
        order = GetSignificand().CompareUnsigned(y.GetSignificand());
      }
      if (isNegative) {
        order = Reverse(order);
      }
      return RelationFromOrdering(order);
    }
  }
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Add(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber();  // NaN + x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
    return result;
  }
  bool isNegative{IsNegative()};
  bool yIsNegative{y.IsNegative()};
  if (IsInfinite()) {
    if (y.IsInfinite()) {
      if (isNegative == yIsNegative) {
        result.value = *this;  // +/-Inf + +/-Inf -> +/-Inf
      } else {
        result.value = NotANumber();  // +/-Inf + -/+Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      }
    } else {
      result.value = *this;  // +/-Inf + x -> +/-Inf
    }
    return result;
  }
  if (y.IsInfinite()) {
    result.value = y;  // x + +/-Inf -> +/-Inf
    return result;
  }
  std::uint64_t exponent{Exponent()};
  std::uint64_t yExponent{y.Exponent()};
  if (exponent < yExponent) {
    // y is larger in magnitude; simplify by reversing operands
    return y.Add(*this, rounding);
  }
  if (exponent == yExponent && isNegative != yIsNegative) {
    Ordering order{GetSignificand().CompareUnsigned(y.GetSignificand())};
    if (order == Ordering::Less) {
      // Same exponent, opposite signs, and y is larger in magnitude
      return y.Add(*this, rounding);
    }
    if (order == Ordering::Equal) {
      // x + (-x) -> +0.0 unless rounding is directed downwards
      if (rounding == Rounding::Down) {
        result.value.word_ = result.value.word_.IBSET(bits - 1);  // -0.0
      }
      return result;
    }
  }
  // Our exponent is greater than y's, or the exponents match and y is not
  // of the opposite sign and greater magnitude.  So (x+y) will have the
  // same sign as x.
  Fraction fraction{GetFraction()};
  Fraction yFraction{y.GetFraction()};
  int rshift = exponent - yExponent;
  if (exponent > 0 && yExponent == 0) {
    --rshift;  // correct overshift when only y is denormal
  }
  RoundingBits roundingBits{yFraction, rshift};
  yFraction = yFraction.SHIFTR(rshift);
  bool carry{false};
  if (isNegative != yIsNegative) {
    // Opposite signs: subtract via addition of two's complement of y and
    // the rounding bits.
    yFraction = yFraction.NOT();
    carry = roundingBits.Negate();
  }
  auto sum{fraction.AddUnsigned(yFraction, carry)};
  fraction = sum.value;
  if (isNegative == yIsNegative && sum.carry) {
    roundingBits.ShiftRight(sum.value.BTEST(0));
    fraction = fraction.SHIFTR(1).IBSET(fraction.bits - 1);
    ++exponent;
  }
  NormalizeAndRound(
      result, isNegative, exponent, fraction, rounding, roundingBits);
  return result;
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Multiply(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber();  // NaN * x -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite() || y.IsInfinite()) {
      if (IsZero() || y.IsZero()) {
        result.value = NotANumber();  // 0 * Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {
        result.value = Infinity(isNegative);
      }
    } else {
      auto product{GetFraction().MultiplyUnsigned(y.GetFraction())};
      std::int64_t exponent{CombineExponents(y, false)};
      if (exponent < 1) {
        int rshift = 1 - exponent;
        exponent = 1;
        bool sticky{false};
        if (rshift >= product.upper.bits + product.lower.bits) {
          sticky = !product.lower.IsZero() || !product.upper.IsZero();
        } else if (rshift >= product.lower.bits) {
          sticky = !product.lower.IsZero() ||
              !product.upper
                   .IAND(product.upper.MASKR(rshift - product.lower.bits))
                   .IsZero();
        } else {
          sticky = !product.lower.IAND(product.lower.MASKR(rshift)).IsZero();
        }
        product.lower = product.lower.DSHIFTR(product.upper, rshift);
        product.upper = product.upper.SHIFTR(rshift);
        if (sticky) {
          product.lower = product.lower.IBSET(0);
        }
      }
      int leadz{product.upper.LEADZ()};
      if (leadz >= product.upper.bits) {
        leadz += product.lower.LEADZ();
      }
      int lshift{leadz};
      if (lshift > exponent - 1) {
        lshift = exponent - 1;
      }
      exponent -= lshift;
      product.upper = product.upper.DSHIFTL(product.lower, lshift);
      product.lower = product.lower.SHIFTL(lshift);
      RoundingBits roundingBits{product.lower, product.lower.bits};
      NormalizeAndRound(result, isNegative, exponent, product.upper, rounding,
          roundingBits, true /*multiply*/);
    }
  }
  return result;
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Divide(
    const Real &y, Rounding rounding) const {
  ValueWithRealFlags<Real> result;
  if (IsNotANumber() || y.IsNotANumber()) {
    result.value = NotANumber();  // NaN / x -> NaN, x / NaN -> NaN
    if (IsSignalingNaN() || y.IsSignalingNaN()) {
      result.flags.set(RealFlag::InvalidArgument);
    }
  } else {
    bool isNegative{IsNegative() != y.IsNegative()};
    if (IsInfinite()) {
      if (y.IsInfinite()) {
        result.value = NotANumber();  // Inf/Inf -> NaN
        result.flags.set(RealFlag::InvalidArgument);
      } else {  // Inf/x -> Inf,  Inf/0 -> Inf
        result.value = Infinity(isNegative);
      }
    } else if (y.IsZero()) {
      if (IsZero()) {  // 0/0 -> NaN
        result.value = NotANumber();
        result.flags.set(RealFlag::InvalidArgument);
      } else {  // x/0 -> Inf, Inf/0 -> Inf
        result.value = Infinity(isNegative);
        result.flags.set(RealFlag::DivideByZero);
      }
    } else if (IsZero() || y.IsInfinite()) {  // 0/x, x/Inf -> 0
      if (isNegative) {
        result.value.word_ = result.value.word_.IBSET(bits - 1);
      }
    } else {
      // dividend and divisor are both finite and nonzero numbers
      Fraction top{GetFraction()}, divisor{y.GetFraction()};
      std::int64_t exponent{CombineExponents(y, true)};
      Fraction quotient;
      bool msb{false};
      if (!top.BTEST(top.bits - 1) || !divisor.BTEST(divisor.bits - 1)) {
        // One or two denormals
        int topLshift{top.LEADZ()};
        top = top.SHIFTL(topLshift);
        int divisorLshift{divisor.LEADZ()};
        divisor = divisor.SHIFTL(divisorLshift);
        exponent += divisorLshift - topLshift;
      }
      for (int j{1}; j <= quotient.bits; ++j) {
        if (NextQuotientBit(top, msb, divisor)) {
          quotient = quotient.IBSET(quotient.bits - j);
        }
      }
      bool guard{NextQuotientBit(top, msb, divisor)};
      bool round{NextQuotientBit(top, msb, divisor)};
      bool sticky{msb || !top.IsZero()};
      RoundingBits roundingBits{guard, round, sticky};
      if (exponent < 1) {
        std::int64_t rshift{1 - exponent};
        for (; rshift > 0; --rshift) {
          roundingBits.ShiftRight(quotient.BTEST(0));
          quotient = quotient.SHIFTR(1);
        }
        exponent = 1;
      }
      NormalizeAndRound(
          result, isNegative, exponent, quotient, rounding, roundingBits);
    }
  }
  return result;
}

template<typename W, int P, bool IM>
RealFlags Real<W, P, IM>::Normalize(bool negative, std::uint64_t exponent,
    const Fraction &fraction, Rounding rounding, RoundingBits *roundingBits) {
  std::uint64_t lshift = fraction.LEADZ();
  if (lshift == fraction.bits /* fraction is zero */ &&
      (roundingBits == nullptr || roundingBits->empty())) {
    // No fraction, no rounding bits -> +/-0.0
    exponent = lshift = 0;
  } else if (lshift < exponent) {
    exponent -= lshift;
  } else if (exponent > 0) {
    lshift = exponent - 1;
    exponent = 0;
  } else if (lshift == 0) {
    exponent = 1;
  } else {
    lshift = 0;
  }
  if (exponent >= maxExponent) {
    // Infinity or overflow
    if (rounding == Rounding::TiesToEven ||
        rounding == Rounding::TiesAwayFromZero ||
        (rounding == Rounding::Up && !negative) ||
        (rounding == Rounding::Down && negative)) {
      word_ = Word{maxExponent}.SHIFTL(significandBits);  // Inf
    } else {
      // directed rounding: round to largest finite value rather than infinity
      // (x86 does this, not sure whether it's standard behavior)
      word_ = Word{word_.MASKR(word_.bits - 1)}.IBCLR(significandBits);
    }
    if (negative) {
      word_ = word_.IBSET(bits - 1);
    }
    RealFlags flags{RealFlag::Overflow};
    if (!fraction.IsZero()) {
      flags.set(RealFlag::Inexact);
    }
    return flags;
  }
  word_ = Word::ConvertUnsigned(fraction).value;
  if (lshift > 0) {
    word_ = word_.SHIFTL(lshift);
    if (roundingBits != nullptr) {
      for (; lshift > 0; --lshift) {
        if (roundingBits->ShiftLeft()) {
          word_ = word_.IBSET(lshift - 1);
        }
      }
    }
  }
  if constexpr (implicitMSB) {
    word_ = word_.IBCLR(significandBits);
  }
  word_ = word_.IOR(Word{exponent}.SHIFTL(significandBits));
  if (negative) {
    word_ = word_.IBSET(bits - 1);
  }
  return {};
}

template<typename W, int P, bool IM>
RealFlags Real<W, P, IM>::Round(
    Rounding rounding, const RoundingBits &bits, bool multiply) {
  std::uint64_t origExponent{Exponent()};
  RealFlags flags;
  bool inexact{!bits.empty()};
  if (inexact) {
    flags.set(RealFlag::Inexact);
  }
  if (origExponent < maxExponent &&
      bits.MustRound(rounding, IsNegative(), word_.BTEST(0) /* is odd */)) {
    typename Fraction::ValueWithCarry sum{
        GetFraction().AddUnsigned(Fraction{}, true)};
    std::uint64_t newExponent{origExponent};
    if (sum.carry) {
      // The fraction was all ones before rounding; sum.value is now zero
      sum.value = sum.value.IBSET(precision - 1);
      if (++newExponent >= maxExponent) {
        flags.set(RealFlag::Overflow);  // rounded away to an infinity
      }
    }
    flags |= Normalize(IsNegative(), newExponent, sum.value);
  }
  if (inexact && origExponent == 0) {
    // inexact denormal input
    if (Exponent() == 0) {
      flags.set(RealFlag::Underflow);  // output still denormal -> Underflow
    } else {
      // Rounding went up to the smallest normal number.
      // Still signal Underflow unless we're in a weird x86 edge case with
      // multiplication: if the sticky bit is set (i.e., the lower half of
      // the full product had bits below the top 2), Underflow gets set in
      // a directed rounding mode only if the guard bit was also set.
      if (multiply && bits.sticky() &&
          (bits.guard() ||
              !(rounding == Rounding::Up || rounding == Rounding::Down))) {
      } else {
        flags.set(RealFlag::Underflow);
      }
    }
  }
  return flags;
}

template<typename W, int P, bool IM>
void Real<W, P, IM>::NormalizeAndRound(ValueWithRealFlags<Real> &result,
    bool isNegative, std::uint64_t exponent, const Fraction &fraction,
    Rounding rounding, RoundingBits roundingBits, bool multiply) {
  result.flags |= result.value.Normalize(
      isNegative, exponent, fraction, rounding, &roundingBits);
  result.flags |= result.value.Round(rounding, roundingBits, multiply);
}

template<typename W, int P, bool IM>
ValueWithRealFlags<Real<W, P, IM>> Real<W, P, IM>::Read(
    const char *&p, Rounding rounding) {
  ValueWithRealFlags<Real> result;
  Real ten{FromInteger(Integer<32>{10}).value};
  for (; parser::IsDecimalDigit(*p); ++p) {
    result.value =
        result.value.Multiply(ten, rounding).AccumulateFlags(result.flags);
    result.value =
        result.value.Add(FromInteger(Integer<32>{*p - '0'}).value, rounding)
            .AccumulateFlags(result.flags);
  }
  std::int64_t exponent{0};
  if (*p == '.') {
    for (++p; parser::IsDecimalDigit(*p); ++p) {
      --exponent;
      result.value =
          result.value.Multiply(ten, rounding).AccumulateFlags(result.flags);
      result.value =
          result.value.Add(FromInteger(Integer<32>{*p - '0'}).value, rounding)
              .AccumulateFlags(result.flags);
    }
  }
  if (parser::IsLetter(*p)) {
    bool negExpo{false};
    if (*++p == '-') {
      negExpo = true;
      ++p;
    } else if (*p == '+') {
      ++p;
    }
    auto expo{Integer<32>::ReadUnsigned(p)};
    std::int64_t expoVal{expo.value.ToInt64()};
    if (expo.overflow) {
      expoVal = std::numeric_limits<std::int32_t>::max();
    } else if (negExpo) {
      expoVal *= -1;
    }
    exponent += expoVal;
  }
  if (exponent == 0) {
    return result;
  }
  Real tenPower{IntPower(ten, Integer<64>{std::abs(exponent)}, rounding)
                    .AccumulateFlags(result.flags)};
  if (exponent > 0) {
    result.value =
        result.value.Multiply(tenPower, rounding).AccumulateFlags(result.flags);
  } else {
    result.value =
        result.value.Divide(tenPower, rounding).AccumulateFlags(result.flags);
  }
  return result;
}

template<typename W, int P, bool IM>
std::string Real<W, P, IM>::DumpHexadecimal() const {
  if (IsNotANumber()) {
    return "NaN 0x"s + word_.Hexadecimal();
  } else if (IsNegative()) {
    return "-"s + Negate().DumpHexadecimal();
  } else if (IsInfinite()) {
    return "Inf"s;
  } else if (IsZero()) {
    return "0.0"s;
  } else {
    Fraction frac{GetFraction()};
    std::string result{"0x"};
    char intPart = '0' + frac.BTEST(frac.bits - 1);
    result += intPart;
    result += '.';
    int trailz{frac.TRAILZ()};
    if (trailz >= frac.bits - 1) {
      result += '0';
    } else {
      int remainingBits{frac.bits - 1 - trailz};
      int wholeNybbles{remainingBits / 4};
      int lostBits{remainingBits - 4 * wholeNybbles};
      if (wholeNybbles > 0) {
        std::string fracHex{frac.SHIFTR(trailz + lostBits)
                                .IAND(frac.MASKR(4 * wholeNybbles))
                                .Hexadecimal()};
        std::size_t field = wholeNybbles;
        if (fracHex.size() < field) {
          result += std::string(field - fracHex.size(), '0');
        }
        result += fracHex;
      }
      if (lostBits > 0) {
        result += frac.SHIFTR(trailz)
                      .IAND(frac.MASKR(lostBits))
                      .SHIFTL(4 - lostBits)
                      .Hexadecimal();
      }
    }
    result += 'p';
    int exponent = Exponent() - exponentBias;
    result += Integer<32>{exponent}.SignedDecimal();
    return result;
  }
}

template class Real<Integer<16>, 11>;
template class Real<Integer<32>, 24>;
template class Real<Integer<64>, 53>;
template class Real<Integer<80>, 64, false>;
template class Real<Integer<128>, 112>;
}  // namespace Fortran::evaluate::value
