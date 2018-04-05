#ifndef FORTRAN_ENUM_SET_H_
#define FORTRAN_ENUM_SET_H_

// Implements a set of enums as a std::bitset<>.  APIs from bitset<> and set<>
// can be used on these sets, whichever might be more clear to the user.

#include <bitset>
#include <cstddef>
#include <functional>

namespace Fortran {
namespace semantics {

template<typename ENUM, ENUM maxval> class EnumSet {
private:
  static constexpr std::size_t bits_{static_cast<std::size_t>(maxval) + 1};
public:
  using bitsetType = std::bitset<bits_>;
  using enumerationType = ENUM;

  constexpr EnumSet() {}
  constexpr EnumSet(const std::initializer_list<enumerationType> &enums) {
    for (auto x : enums) {
      set(x);
    }
  }
  constexpr EnumSet(const EnumSet &) = default;
  constexpr EnumSet(EnumSet &&) = default;

  constexpr EnumSet &operator=(const EnumSet &) = default;
  constexpr EnumSet &operator=(EnumSet &&) = default;

  const bitsetType &bitset() const { return bitset_; }

  constexpr EnumSet &operator&=(const EnumSet &that) {
    bitset_ &= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator&=(EnumSet &&that) {
    bitset_ &= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator|=(const EnumSet &that) {
    bitset_ |= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator|=(EnumSet &&that) {
    bitset_ |= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator^=(const EnumSet &that) {
    bitset_ ^= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator^=(EnumSet &&that) {
    bitset_ ^= that.bitset_;
    return *this;
  }

  constexpr EnumSet operator~() const {
    EnumSet result;
    result.bitset_ = ~bitset_;
    return result;
  }
  constexpr EnumSet operator&(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ &= that.bitset_;
    return result;
  }
  constexpr EnumSet operator&(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ &= that.bitset_;
    return result;
  }
  constexpr EnumSet operator|(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ |= that.bitset_;
    return result;
  }
  constexpr EnumSet operator|(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ |= that.bitset_;
    return result;
  }
  constexpr EnumSet operator^(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ ^= that.bitset_;
    return result;
  }
  constexpr EnumSet operator^(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ ^= that.bitset_;
    return result;
  }

  constexpr EnumSet operator==(const EnumSet &that) const {
    return bitset_ == that.bitset_;
  }
  constexpr EnumSet operator==(EnumSet &&that) const {
    return bitset_ == that.bitset_;
  }
  constexpr EnumSet operator!=(const EnumSet &that) const {
    return bitset_ != that.bitset_;
  }
  constexpr EnumSet operator!=(EnumSet &&that) const {
    return bitset_ != that.bitset_;
  }

  // N.B. std::bitset<> has size() for max_size(), but that's not the same
  // thing as std::set<>::size(), which is an element count.
  static constexpr std::size_t max_size() { return bits_; }
  constexpr bool test(enumerationType x) const {
    return bitset_.test(static_cast<std::size_t>(x));
  }
  constexpr bool all() const { return bitset_.all(); }
  constexpr bool any() const { return bitset_.any(); }
  constexpr bool none() const { return bitset_.none(); }

  // N.B. std::bitset<> has count() as an element count, while
  // std::set<>::count(x) returns 0 or 1 to indicate presence.
  constexpr std::size_t count() const { return bitset_.count(); }
  constexpr std::size_t count(enumerationType x) const {
    return test(x) ? 1 : 0;
  }

  constexpr EnumSet &set() {
    bitset_.set();
    return *this;
  }
  constexpr EnumSet &set(enumerationType x, bool value = true) {
    bitset_.set(static_cast<std::size_t>(x), value);
    return *this;
  }
  constexpr EnumSet &reset() {
    bitset_.reset();
    return *this;
  }
  constexpr EnumSet &reset(enumerationType x, bool value = true) {
    bitset_.reset(static_cast<std::size_t>(x), value);
    return *this;
  }
  constexpr EnumSet &flip() {
    bitset_.flip();
    return *this;
  }
  constexpr EnumSet &flip(enumerationType x) {
    bitset_.flip(static_cast<std::size_t>(x));
    return *this;
  }

  constexpr bool empty() const { return none(); }
  void clear() { reset(); }
  void insert(enumerationType x) { set(x); }
  void insert(enumerationType &&x) { set(x); }
  void emplace(enumerationType &&x) { set(x); }
  void erase(enumerationType x) { reset(x); }
  void erase(enumerationType &&x) { reset(x); }

private:
  bitsetType bitset_;
};
}  // namespace semantics
}  // namespace Fortran

template<typename ENUM, ENUM maxval>
struct std::hash<Fortran::semantics::EnumSet<ENUM, maxval>> {
  std::size_t operator()(const Fortran::semantics::EnumSet<ENUM, maxval> &x) const {
    return std::hash(x.bitset());
  }
};
#endif  // FORTRAN_ENUM_SET_H_
