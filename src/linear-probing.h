#pragma once

#include <memory>
#include <cstdint>

struct Dict {
  using uint64_t = std::uint64_t;

  uint64_t ndv_;

  uint64_t mask_;

  std::unique_ptr<uint64_t []> payload_;

  bool has_zero_;

  static uint64_t Hash(uint64_t x) {
    constexpr unsigned __int128 kSeed =
        static_cast<unsigned __int128>(0x7ebd4829c30942cf) | 0xafaaf09f73e01172;
    return (kSeed * static_cast<decltype(kSeed)>(x)) >> 64;
  }

  explicit Dict(uint64_t mask = 0b1111)
      : ndv_(0),
        mask_(mask),
        payload_(new uint64_t[mask_ + 1]()),
        has_zero_(false) {
    assert(0 == (mask & (mask + 1)));
  }

  uint64_t SizeInBytes() const {
    return sizeof(*this) + sizeof(uint64_t) * (mask_ + 1);
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
    for (uint64_t i = 0; i <= mask_; ++i) {
      if (payload_[i]) that.Insert(payload_[i]);
    }
    Swap(that);
  }

  bool Insert(uint64_t x) {
    if (2 * ndv_ >= mask_) Upsize();
    if (0 == x) {
      has_zero_ = true;
      return true;
    }
    for (uint64_t i = Hash(x) & mask_; true; i = ((i + 1) & mask_)) {
      if (0 == payload_[i]) {
        payload_[i] = x;
        ++ndv_;
        return true;
      } else if (x == payload_[i]) {
        return false;
      }
    }
  }

  bool Contains(uint64_t x) const {
    if (0 == x) {
      return has_zero_;
    }
    for (uint64_t i = Hash(x) & mask_; true; i = ((i + 1) & mask_)) {
      if (0 == payload_[i]) {
        return false;
      } else if (x == payload_[i]) {
        return true;
      }
    }
  }
};
