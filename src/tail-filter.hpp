#pragma once

#include <cmath>

#include "quotient-dysect.hpp"

struct TailFilter {

  uint64_t ndv_;

  int epoch_;

  int lgm_;

  int lgme_;

  QuotientDysect qd_;

  SlotArray bitset_;

  static uint64_t MultiplyHash(uint64_t x) {
    unsigned __int128 y = 0x6d064b7e7e084d32;
    y = y << 64;
    y = y | 0xaad87d809dd4d431;
    return (static_cast<unsigned __int128>(x) * y) >> 64;
  }

  TailFilter(int lgm, double epsilon)
      : ndv_(0),
        epoch_(0),
        lgm_(lgm),
        lgme_(lgm_ + std::ilogb(1.0 / epsilon)),
        qd_(lgm_ + epoch_, 1 + lgme_ - lgm_, 2, 3, 1, lgm_ - 1 - 3,
            MultiplyHash),
        bitset_() {
    assert(lgm > 0);
    assert(0 < epsilon);
    assert(epsilon < 1);
  }

  bool Lookup(uint64_t hash) const {
    if (bitset_.Capacity() > 0) {
      const int shift_out =
          __builtin_clzll(static_cast<unsigned long long>(bitset_.Capacity()));
      if (1 == bitset_[hash >> shift_out]) return true;
    }
    const uint64_t key = hash >> (CHAR_BIT * sizeof(hash) - lgm_ - epoch_);
    const uint64_t value =
        (hash >> (CHAR_BIT * sizeof(hash) - epoch_ - 1 - lgme_)) &
        ((UINT64_C(1) << (lgme_ - lgm_)) - 1);
    for (auto i = qd_.Find(key); i != qd_.End(); ++i) {
      const int tz = __builtin_ctzll(static_cast<unsigned long long>(*i));
      const uint64_t prefix = (*i) >> (tz + 1);
      const uint64_t keyfix = value >> tz;
      if (prefix == keyfix) return true;
    }
    return false;
  }

  void Upsize() {
    QuotientDysect qd2(lgm_ + epoch_ + 1, 1 + lgme_ - lgm_, 2, 3, 1,
                       epoch_ + lgm_ - 1 - 3, MultiplyHash);
    SlotArray sa = (bitset_.Capacity() > 0)
                       ? SlotArray(1, bitset_.Capacity() * 2)
                       : SlotArray();
    for (uint64_t i = 0; i < bitset_.Capacity(); ++i) {
      sa[2 * i] = sa[2 * i + 1] = bitset_[i];
    }
    for(auto i = qd_.Begin(); i!= qd_.End(); ++i) {
      auto kv = i.GetOriginal();
      if (kv.value == (UINT64_C(1) << (lgme_ - lgm_))) {
        if (sa.Capacity() == 0) {
          sa = SlotArray(1, UINT64_C(1) << (lgm_ + epoch_));
        }
        sa[kv.key] = 1;
      } else {
        const uint64_t insert_key = kv.key * 2 + (kv.value >> (lgme_ - lgm_));
        const uint64_t insert_value =
            (kv.value * 2) & ((1 << (1 + lgme_ - lgm_)) - 1);
        qd2.Insert(insert_key, insert_value);
      }
    }
    qd_.Swap(qd2);
    bitset_.Swap(sa);
    ++epoch_;
  }

  bool Insert(uint64_t hash) {
    if (Lookup(hash)) return false;
    const uint64_t key = hash >> (CHAR_BIT * sizeof(hash) - lgm_ - epoch_);
    const uint64_t value =
        (hash >> (CHAR_BIT * sizeof(hash) - epoch_ - 1 - lgme_)) &
        ((UINT64_C(1) << (lgme_ - lgm_)) - 1);
    qd_.Insert(key, (2 * value) | 1);
    assert(Lookup(hash));
    ++ndv_;
    return true;
  }
};
