#pragma once

#include <memory>
#include <cstdint>


template<typename T = std::uint64_t, typename U = unsigned __int128>
struct Dict {
  std::uint64_t ndv_;

  std::uint64_t mask_;

  std::unique_ptr<T []> payload_;

  bool has_zero_;

  static T Hash(T x) {
    constexpr U kSeed = static_cast<U>(
        (static_cast<unsigned __int128>(0x7ebd4829c30942cf) << 64) |
        0xafaaf09f73e01172);
    return (kSeed * static_cast<decltype(kSeed)>(x)) >> (CHAR_BIT * sizeof(T));
  }

  explicit Dict(T mask = 0b1111)
      : ndv_(0),
        mask_(mask),
        payload_(new T[mask_ + 1]()),
        has_zero_(false) {
    assert(0 == (mask & (mask + 1)));
  }

  uint64_t SizeInBytes() const {
    return sizeof(*this) + sizeof(payload_[0]) * (mask_ + 1);
  }

  void Swap(Dict& that) {
    using std::swap;
    swap(ndv_, that.ndv_);
    swap(mask_, that.mask_);
    swap(payload_, that.payload_);
    swap(has_zero_, that.has_zero_);
  }

  void Upsize() {
    Dict that(mask_ * 2 + 1);
    that.has_zero_ = has_zero_;
    for (T i = 0; i <= mask_; ++i) {
      if (payload_[i]) that.Insert(payload_[i]);
    }
    Swap(that);
  }

  bool Insert(T x) {
    if (4 * ndv_ > 3 * mask_) Upsize();
    if (0 == x) {
      has_zero_ = true;
      return true;
    }
    for (T i = Hash(x) & mask_; true; i = ((i + 1) & mask_)) {
      if (0 == payload_[i]) {
        payload_[i] = x;
        ++ndv_;
        return true;
      } else if (x == payload_[i]) {
        return false;
      }
    }
  }

  bool Contains(T x) const {
    if (0 == x) {
      return has_zero_;
    }
    for (T i = Hash(x) & mask_; true; i = ((i + 1) & mask_)) {
      if (0 == payload_[i]) {
        return false;
      } else if (x == payload_[i]) {
        return true;
      }
    }
  }
};

template<typename T = std::uint64_t, typename U = unsigned __int128>
struct SizedDict {
  std::uint64_t ndv_;

  U capacity_;

  std::unique_ptr<T []> payload_;

  bool has_zero_;

  T Hash(T x) const {
    constexpr U kSeed = static_cast<U>(
        (static_cast<unsigned __int128>(0x7ebd4829c30942cf) << 64) |
        0xafaaf09f73e01172);
    const U partial =
        (kSeed * static_cast<decltype(kSeed)>(x)) >> (CHAR_BIT * sizeof(T));
    return (capacity_ * partial) >> (CHAR_BIT * sizeof(T));
  }

  explicit SizedDict(T max_ndv)
      : ndv_(0),
        capacity_(max_ndv / (0.5)),
        payload_(new T[capacity_ + 1]()),
        has_zero_(false) {}

  uint64_t SizeInBytes() const {
    return sizeof(*this) + sizeof(payload_[0]) * capacity_;
  }

  bool Insert(T x) {
    if (0 == x) {
      has_zero_ = true;
      return true;
    }
    for (T i = Hash(x); true; i = ((i + 1 >= capacity_) ? 0 : (i + 1))) {
      if (0 == payload_[i]) {
        payload_[i] = x;
        ++ndv_;
        return true;
      } else if (x == payload_[i]) {
        return false;
      }
    }
  }

  bool Contains(T x) const {
    if (0 == x) {
      return has_zero_;
    }
    for (T i = Hash(x); true; i = ((i + 1 >= capacity_) ? 0 : (i + 1))) {
      if (0 == payload_[i]) {
        return false;
      } else if (x == payload_[i]) {
        return true;
      }
    }
  }
};

struct SimdSizedDict {
  using uint64_t = std::uint64_t;
  using uint32_t = std::uint32_t;

  uint64_t ndv_;

  uint64_t capacity_;

  std::unique_ptr<uint32_t []> payload_;

  bool has_zero_;

  uint32_t Hash(uint32_t x) const {
    constexpr uint64_t kSeed = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(0x7ebd4829c30942cf) << 64) |
        0xafaaf09f73e01172);
    const uint64_t partial =
        (kSeed * static_cast<decltype(kSeed)>(x)) >> (CHAR_BIT * sizeof(uint32_t));
    return (capacity_ * partial) >> (CHAR_BIT * sizeof(uint32_t));
  }

  explicit SimdSizedDict(uint32_t max_ndv)
      : ndv_(0),
        capacity_(max_ndv / (0.5) + (sizeof(__m512i) / sizeof(uint32_t))),
        payload_(new uint32_t[capacity_]()),
        has_zero_(false) {}

  uint64_t SizeInBytes() const {
    return sizeof(*this) + sizeof(payload_[0]) * capacity_;
  }

  bool Insert(uint32_t x) {
    if (0 == x) {
      has_zero_ = true;
      return true;
    }
    for (uint32_t i = Hash(x); true; i = ((i + 1 >= capacity_) ? 0 : (i + 1))) {
      if (0 == payload_[i]) {
        payload_[i] = x;
        ++ndv_;
        return true;
      } else if (x == payload_[i]) {
        return false;
      }
    }
  }

  bool Contains(uint32_t x) const {
    if (0 == x) {
      return has_zero_;
    }
    const __m512i xs = _mm512_set1_epi32(x);
    const __m512i zeros = _mm512_set1_epi32(0);
    for (uint32_t i = Hash(x); true;
         i = ((i + sizeof(__m512i) / sizeof(uint32_t) >= capacity_)
                  ? 0
                  : (i + sizeof(__m512i) / sizeof(uint32_t)))) {
      const __m512i data =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&payload_[i]));
      if (_mm512_cmpeq_epu32_mask(data, xs)) return true;
      if (_mm512_cmpeq_epu32_mask(data, zeros)) return false;
      // if (0 == _mm512_reduce_min_epu32(data)) return false;
      // _mm512_reduce_mul_epi32
    }
  }
};
