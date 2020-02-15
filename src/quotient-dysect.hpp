#pragma once

#include "slot-array.hpp"

struct QuotientDysect {
  std::unique_ptr<std::unique_ptr<SlotArray[]>[] > payload_;

  int k_, v_, d_, w_, s_, log_little_;

  uint64_t SpaceUsed() const {
    uint64_t result = sizeof(*this);
    for (int p = 0; p < d_; ++p) {
      result += sizeof(payload_[p]);
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        result += payload_[p][q].SpaceUsed();
      }
    }
    return result;
  }

  QuotientDysect(int k, int v, int d, int w, int s)
    : payload_(nullptr), k_(k), v_(v), d_(d), w_(w), s_(s), log_little_(s_) {
    assert(k_ > 0);
    assert (v_ >= 0);
    assert (d_ >= 2);
    assert (w_ >= 0);
    assert (s_ >= 0);
    payload_.reset(new std::unique_ptr<SlotArray[]>[d_]);
    for (int p = 0; p < d_; ++p) {
      payload_[p].reset(new SlotArray[UINT64_C(1) << w_]);
      const int slot_length = s_ + v_ + std::max(0, k_ - w_ - log_little_);
      for (uint64_t q = 0; q < (UINT64_C(1) << w); ++q) {
        payload_[p][q] = SlotArray(slot_length, UINT64_C(1) << log_little_);
      }
    }
    assert (0 == FilledSlots());
    assert (Capacity() == (UINT64_C(1) << (log_little_ + w_)) * d);
  }

  uint64_t Capacity() const {
    uint64_t result = 0;
    for (int p = 0; p < d_; ++p) {
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        result += payload_[p][q].Capacity();
      }
    }
    return result;
  }

  uint64_t FilledSlots() const {
    uint64_t result = 0;
    for (int p = 0; p < d_; ++p) {
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        for (uint64_t r = 0; r < payload_[p][q].Capacity(); ++r) {
          result += (payload_[p][q][r] != 0);
        }
      }
    }
    assert (result < Capacity());
    return result;
  }

  uint64_t operator()(int p, uint64_t q, uint64_t r) const {
    return payload_[p][q][r];
  }
};
