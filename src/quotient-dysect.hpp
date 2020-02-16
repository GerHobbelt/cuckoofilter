#pragma once

#include <functional>


#include "slot-array.hpp"

struct QuotientDysect {
  std::unique_ptr<std::unique_ptr<SlotArray[]>[] > payload_;

  int k_, v_, d_, w_, s_, log_little_;

  std::unique_ptr<std::function<uint64_t(uint64_t)>[]> hash_functions_;

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

  QuotientDysect(
      int k, int v, int d, int w, int s,
      std::unique_ptr<std::function<uint64_t(uint64_t)>[]> hash_functions)
      : payload_(nullptr),
        k_(k),
        v_(v),
        d_(d),
        w_(w),
        s_(s),
        log_little_(s_),
        hash_functions_(hash_functions.release()) {
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

  uint64_t Hash(int arena, uint64_t key) const {
    return hash_functions_[arena - 1](key);
  }

  struct Iterator {
    const QuotientDysect *that_;

    int p_;

    uint64_t q_, r_;

    bool operator==(const Iterator &other) {
      return that_ == other.that_ && p_ == other.p_ && q_ == other.q_ &&
             r_ == other.r_;
    }

    struct KeyValuePair {
      uint64_t key = 0, value = 0;
    };

    KeyValuePair Get() const {
      KeyValuePair result;
      const SlotArray &t = that_->payload_[p_][q_];
      const uint64_t pow_ell = that_->payload_[p_][q_].Capacity();
      const int ell =
          that_->log_little_ + (pow_ell == (1 << that_->log_little_));
      // const int width = that_->payload_[p_][q_].Width();
      const uint64_t r = (r_ - (t[r_] & ((1 << that_->s_) - 1))) & (pow_ell - 1);
      result.key = (q_ << ell) | r;
      result.key = result.key >> std::max(0, ell + that_->w_ - that_->k_);
      result.key = (result.key << std::max(0, that_->k_ - ell - that_->w_)) |
                   (t[r_] >> (that_->v_ + that_->s_));
      result.value = (t[r_] >> that_->s_) & ((1 << that_->v_) - 1);
      return result;
    }

    Iterator &operator++() {
      do {
        if (p_ >= that_->d_) return *this;
        ++r_;
        if (r_ >= that_->payload_[p_][q_].Capacity()) {
          r_ = 0;
          ++q_;
          if (q_ >= (1 << that_->w_)) {
            q_ = 0;
            ++p_;
            if (p_ >= that_->d_) return *this;
          }
        }
      } while (0 == that_->payload_[p_][q_][r_]);
      return *this;
    }

    int Arena() const { return p_; }
  };


  Iterator Begin() const {
    Iterator result{this, 0, 0, 0};
    if (not payload_[0][0][0]) ++result;
    return result;
  }

  Iterator End() const { return Iterator{this, d_, 0, 0}; }

  struct ResultSetIterator {
    QuotientDysect * that_;

    Iterator i_;

    uint64_t key_, current_key_;

    uint64_t operator*() const { return i_.Get().value; }

    ResultSetIterator operator++() {
      do {
        const int arena = i_.Arena();
        ++i_;
        if (i_ == that_->End()) return *this;
        if (i_.Arena() > arena) {
          current_key_ = that_->Hash(i_.Arena(), key_);
          AdvanceWithinArena();
        }
      } while (i_.Get().key != current_key_);
      return *this;
    }

    void AdvanceWithinArena() {
      i_.q_ = current_key_ >> (that_->k_ - that_->w_);
      const uint64_t pow_ell = that_->payload_[i_.p_][i_.q_].Capacity();
      const int ell =
          that_->log_little_ + (pow_ell == (1 << that_->log_little_));
      i_.r_ = (current_key_ & ((1 << that_->k_) - 1)) >>
               std::max(0, ell + that_->w_ - that_->k_);
      i_.r_ = i_.r_ << std::max(0, that_->k_ - ell - that_->w_);
    }

    ResultSetIterator(QuotientDysect *that, uint64_t key)
        : that_(that), i_{that_, 0, 0, 0}, key_(key), current_key_(key) {
      AdvanceWithinArena();
      if (i_.Get().key != current_key_) ++i_;
    }
  };

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
