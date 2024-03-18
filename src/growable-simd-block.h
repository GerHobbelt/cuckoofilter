// This follows "How to Approximate A Set Without Knowing Its Size In Advance",
// by Pagh et al. This is a version of their warmup construction that supports
// approximate membership query on sets in which the size is not known when the
// structure is initialized. Note that it uses the SimdBlockBloom filter from
// this repository, rather than the Raman and Rao construction mentioned in
// Pagh, et al.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "hashutil.h"
#include "simd-block.h"

using uint64_t = ::std::uint64_t;

template <typename HashFamily = ::cuckoofilter::TwoIndependentMultiplyShift>
class GrowSimdBlockFilter {
  using Filter = SimdBlockFilter<HashFamily>;

  std::vector<std::unique_ptr<Filter>> that_;

  double epsilon_;

  uint64_t initial_bytes_;

  int64_t ttl_;

 public:
  explicit GrowSimdBlockFilter(uint64_t initial_bytes, double epsilon)
      : that_(),
        epsilon_(epsilon * 6 / std::pow(3.1415, 2)),
        initial_bytes_(
            1 << static_cast<int>(std::floor(std::log2(initial_bytes)))),
        ttl_(CHAR_BIT * initial_bytes_ * (-1.0 / 8) *
             std::log(1.0 - std::pow(epsilon_, 1.0 / 8))) {
    that_.emplace_back(new Filter(std::floor(std::log2(initial_bytes_))));
  }

  void AddUnique(uint64_t key) {
    if (ttl_ <= 0) {
      const double bytes = initial_bytes_ << that_.size();
      ttl_ = CHAR_BIT * bytes * (-1.0 / 8) *
             std::log(1.0 - std::pow(epsilon_ / std::pow(that_.size() + 1, 2),
                                     1.0 / 8));
      that_.emplace_back(new Filter(log2(bytes)));
    }
    that_.back()->Add(key);
    --ttl_;
  }

  bool AddAny(uint64_t key) {
    if (Find(key)) return false;
    AddUnique(key);
    return true;
  }

  bool Find(uint64_t key) const noexcept {
    for (const auto &f : that_) {
      if (f->Find(key)) return true;
    }
    return false;
  }

  uint64_t SizeInBytes() const {
    uint64_t result = sizeof(*this) + that_.capacity() * sizeof(that_[0]);
    for (const auto &f : that_) result += f->SizeInBytes();
    return result;
  }
};
