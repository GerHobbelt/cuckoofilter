#pragma once

#include "slot-array.hpp"

struct QuotientDysect {
  struct Partition {
    std::unique_ptr<SlotArray[]> sub_tables_;
    SlotArray sub_table_big_;
  };

  std::unique_ptr<Partition[]> payload_;

  int k_, v_, d_, w_, s_, log_little_;

  uint64_t SpaceUsed() const {
    uint64_t result = sizeof(payload_);
    for (int p = 0; p < d_; ++p) {
      result += sizeof(payload_[p]);
      result += SlotArray::SpaceUsed(UINT64_C(1) << w_, 1);
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        const int ell = log_little_ + payload_[p].sub_table_big_(1, q);
        const int slot_length = s_ + v_ + std::max(0, k_ - w_ - ell);
        result += SlotArray::SpaceUsed(UINT64_C(1) << ell, slot_length);
      }
    }
    //result += (UINT64_C(1) << w_) * d_ * sizeof(void *)
    return result;
  }

  QuotientDysect(int k, int v, int d, int w, int s)
    : payload_(nullptr), k_(k), v_(v), d_(d), w_(w), s_(s), log_little_(s_) {
    assert(k_ > 0);
    assert (v_ >= 0);
    assert (d_ >= 2);
    assert (w_ >= 0);
    assert (s_ >= 0);
    payload_.reset(new Partition[d_]);
    for (int p = 0; p < d_; ++p) {
      payload_[p].sub_tables_.reset(new SlotArray[UINT64_C(1) << w_]);
      payload_[p].sub_table_big_ = SlotArray(UINT64_C(1) << w_, 1);
      const int slot_length = s_ + v_ + std::max(0, k_ - w_ - log_little_);
      for (uint64_t q = 0; q < (UINT64_C(1) << w); ++q) {
        payload_[p].sub_tables_[q] =
          SlotArray(UINT64_C(1) << log_little_, slot_length);
      }
    }
    assert (0 == FilledSlots());
    assert (Capacity() == (UINT64_C(1) << (log_little_ + w_)) * d);
  }

  uint64_t Capacity() const {
    uint64_t result = 0;
    for (int p = 0; p < d_; ++p) {
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        const int ell = log_little_ + payload_[p].sub_table_big_(1, q);
        result += UINT64_C(1) << ell;
      }
    }
    return result;
  }

  uint64_t FilledSlots() const {
    uint64_t result = 0;
    for (int p = 0; p < d_; ++p) {
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        const int ell = log_little_ + payload_[p].sub_table_big_(1, q);
        const int slot_length = s_ + v_ + std::max(0, k_ - w_ - ell);
        for (uint64_t r = 0; r < (UINT64_C(1) << ell); ++r) {
          uint64_t slot_val = payload_[p].sub_tables_[q](slot_length, r);
          slot_val = slot_val & (((UINT64_C(1) << v_) - 1) << s_);
          result += (slot_val != 0);
        }
      }
    }
    assert (result < Capacity());
    return result;
  }

  uint64_t operator()(int p, uint64_t q, uint64_t r) const {
    const int ell = log_little_ + payload_[p].sub_table_big_(1, q);
    const int slot_length = s_ + v_ + std::max(0, k_ - w_ - ell);
    return payload_[p].sub_tables_[q](slot_length, r);
  }
};
