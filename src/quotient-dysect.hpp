#pragma once

#include <functional>

#include "slot-array.hpp"



struct QuotientDysect {
  std::unique_ptr<std::unique_ptr<SlotArray[]>[] > payload_;

  int k_, v_, d_, w_, s_, log_little_;

  using HashFunction = std::function<uint64_t(uint64_t)>;

  using HashBijection = std::pair<HashFunction, HashFunction>;

  std::unique_ptr<HashBijection[]> hash_bijections_;

  template<typename F>
  static HashBijection Feistelize(F g, int key_length) {
    HashFunction f = g;
    constexpr int kRounds = 4;
    auto forward = [f, key_length](uint64_t x) {
      int newk = (key_length + 1) / 2;
      uint64_t result = x;
      for (int i = 0; i < kRounds; ++i) {
        uint64_t temp = result >> newk;
        temp = temp | (((result & ((1 << newk) - 1)) ^
                        (f(result >> newk) & ((1 << newk) - 1)))
                       << newk);
        result = temp;
      }
      return result;
    };
    auto backward = [f, key_length](uint64_t x) {
      int newk = (key_length + 1) / 2;
      uint64_t result = x;
      for (int i = 0; i < kRounds; ++i) {
        uint64_t temp = (result & ((1 << newk) - 1)) << newk;
        temp = temp | ((result >> newk) ^
                       (f(result & ((1 << newk) - 1)) & ((1 << newk) - 1)));
        result = temp;
      }
      return result;
    };
    return HashBijection{forward, backward};
  }

  uint64_t SpaceUsed() const {
    uint64_t result = sizeof(*this);
    for (int p = 0; p < d_; ++p) {
      result += sizeof(payload_[p]);
      for (uint64_t q = 0; q < (UINT64_C(1) << w_); ++q) {
        result += payload_[p][q].SpaceUsed();
      }
    }
    assert(result >= (d_ * (UINT64_C(1) << w_) * (UINT64_C(1) << log_little_) *
                          (s_ + v_) +
                      CHAR_BIT - 1) /
                         CHAR_BIT);
    return result;
  }

  template <typename... Ts>
  QuotientDysect(int k, int v, int d, int w, int s, int log_little,
                 Ts... hash_functions)
      : payload_(nullptr),
        k_(k),
        v_(v),
        d_(d),
        w_(w),
        s_(s),
        log_little_(log_little),
        hash_bijections_(new HashBijection[sizeof...(Ts)]{
            Feistelize(hash_functions, k_)...}) {
    assert(k_ > 0);
    assert (v_ >= 0);
    assert (d_ >= 2);
    assert (w_ >= 0);
    assert (s_ >= 0);
    assert (hash_bijections_.get() != nullptr);
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
    return hash_bijections_[arena - 1].first(key);
  }

  uint64_t HashInverse(int arena, uint64_t key) const {
    return hash_bijections_[arena - 1].second(key);
  }

  void Insert(uint64_t key, uint64_t value) {
    assert (key < (UINT64_C(1) << k_));
    assert(value < (UINT64_C(1) << v_));
    uint64_t current_key = key;
    int p = 0;
    uint64_t iterations = 0;
    while (true) {
      ++iterations;
      if (iterations > 500) Upsize();

      const uint64_t q = current_key >> (k_ - w_);
      if (SetLocal(payload_[p][q], key, value)) return;


      const uint64_t pow_ell = payload_[p][q].Capacity();
      const int ell = log_little_ + (pow_ell > (UINT64_C(1) << log_little_));
      uint64_t r = (current_key >> std::max(0, k_ - w_ - ell)) & (pow_ell - 1);
      r = r << std::max(0, w_ + ell - k_);

      uint64_t i = rand() % ((1 << std::max(0, w_ + ell - k_)) + (1 << s_) - 1);

      KeyValuePair kv = Get(p, q, (r + i) & (pow_ell - 1));
      payload_[p][q][(r + i) & (pow_ell - 1)] = 0;
      const bool ok = SetLocal(payload_[p][q], current_key, value);
      assert(ok);
      current_key = (p > 0) ? HashInverse(p, kv.key) : kv.key;
      value = kv.value;
      p = (p+1) % d_;
    }
  }

  struct KeyValuePair {
    uint64_t key = 0, value = 0;
  };

  bool SetLocal(SlotArray &sa, uint64_t key, uint64_t value) {
    //const uint64_t q = key >> (k_ - w_);
    const uint64_t pow_ell = sa.Capacity();
    const int ell = log_little_ + (pow_ell > (UINT64_C(1) << log_little_));
    uint64_t r = (key >> std::max(0, k_ - w_ - ell)) & (pow_ell - 1);
    r = r << std::max(0, w_ + ell - k_);
    uint64_t val =
        (value << s_) |
        ((key & ((1 << std::max(0, k_ - w_ - ell)) - 1)) << (s_ + v_));
    for (uint64_t i = r; i < (r + (1 << std::max(0, w_ + ell - k_))); ++i) {
      if (0 == sa[i & (pow_ell - 1)]) {
        sa[i & (pow_ell - 1)] = val;
        return true;
      }
    }

    for (uint64_t i = 1; i < (UINT64_C(1) << s_); ++i) {
      uint64_t r_with =
          (r + (1 << std::max(0, w_ + ell - k_)) - 1 + i) & (pow_ell - 1);
      if (0 == sa[r_with]) {
        sa[r_with] = val | i;
        return true;
      }
    }
    return false;
  }

  KeyValuePair Get(int p, uint64_t q, uint64_t r) const {
    KeyValuePair result;
    const SlotArray &t = payload_[p][q];
    const uint64_t pow_ell = payload_[p][q].Capacity();
    const int ell = log_little_ + (pow_ell > (UINT64_C(1) << log_little_));
    const uint64_t r_adjusted = (r - (t[r] & ((1 << s_) - 1))) & (pow_ell - 1);
    result.key = (q << ell) | r_adjusted;
    result.key = result.key >> std::max(0, ell + w_ - k_);
    result.key =
        (result.key << std::max(0, k_ - ell - w_)) | (t[r] >> (v_ + s_));
    result.value = (t[r] >> s_) & ((1 << v_) - 1);
    return result;
  }

  struct Iterator {
    const QuotientDysect *that_;

    int p_;

    uint64_t q_, r_;

    bool operator==(const Iterator &other) const {
      return that_ == other.that_ && p_ == other.p_ && q_ == other.q_ &&
             r_ == other.r_;
    }

    bool operator!=(const Iterator &other) { return not(*this == other); }

    KeyValuePair Get() const {
      return that_->Get(p_,q_,r_);
    }

    Iterator &operator++() {
      do {
        if (p_ >= that_->d_) return *this;
        ++r_;
        if (r_ >= that_->payload_[p_][q_].Capacity()) {
          r_ = 0;
          ++q_;
          if (q_ >= (UINT64_C(1) << that_->w_)) {
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
    const QuotientDysect *that_;

    Iterator i_;

    uint64_t key_, current_key_;

    bool operator==(const Iterator &j) const { return i_ == j; }

    bool operator!=(const Iterator &j) const { return not(i_ == j); }

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
        that_->log_little_ + (pow_ell > (UINT64_C(1) << that_->log_little_));
      i_.r_ = current_key_ & ((1 << (that_->k_ - that_->w_)) - 1);
      i_.r_ = i_.r_ >> std::max(0, that_->k_ - ell - that_->w_);
      assert(i_.r_ < that_->payload_[i_.p_][i_.q_].Capacity());
      i_.r_ = i_.r_ << std::max(0, that_->w_ + ell - that_->k_);
      assert(i_.r_ < that_->payload_[i_.p_][i_.q_].Capacity());
    }

    ResultSetIterator(const QuotientDysect *that, uint64_t key)
        : that_(that), i_{that_, 0, 0, 0}, key_(key), current_key_(key) {
      AdvanceWithinArena();
      if (i_.Get().key != current_key_) ++i_;
    }
  };

  ResultSetIterator Find(uint64_t key) const {
    return ResultSetIterator(this, key);
  }

  bool Upsize() {
    for (int p = 0; p < d_; ++p) {
      for (int q = 0; q < (1 << w_); ++ q) {
        if (payload_[p][q].Capacity() == (UINT64_C(1) << log_little_)) {
          SlotArray replacement(
              s_ + v_ + std::max(0, k_ - log_little_ - w_ - 1),
              UINT64_C(2) << log_little_);
          for (uint64_t r = 0; r < (UINT64_C(1) << log_little_); ++r) {
            if (payload_[p][q][r] == 0) continue;
            KeyValuePair kv = Get(p, q, r);
            if (not SetLocal(replacement, kv.key, kv.value)) return false;
          }
          payload_[p][q].Swap(replacement);
          if ((p + 1 == d_) && (q + 1 == (1 << w_))) ++log_little_;
          return true;
        }
      }
    }
    return false;
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
