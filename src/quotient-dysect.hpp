#pragma once

#include <functional>
#include <iostream>

#include "slot-array.hpp"



struct QuotientDysect {
  uint64_t ndv_, capacity_;

  std::unique_ptr<std::unique_ptr<SlotArray[]>[] > payload_;

  int k_, v_, d_, w_, s_, log_little_;

  using HashFunction = std::function<uint64_t(uint64_t)>;

  using HashBijection = std::pair<HashFunction, HashFunction>;

  std::unique_ptr<HashBijection[]> hash_bijections_;

  void Swap(QuotientDysect &that) {
    using std::swap;
    swap(payload_, that.payload_);
    swap(k_, that.k_);
    swap(v_, that.v_);
    swap(d_,that.d_);
    swap(w_, that.w_);
    swap(s_, that.s_);
    swap(log_little_, that.log_little_);
    swap(hash_bijections_, that.hash_bijections_);
  }

  template<typename F>
  static HashBijection Feistelize(F g, int key_length) {
    HashFunction f = g;
    constexpr int kRounds = 4;
    auto forward = [f, key_length](uint64_t x) {
      int smallk = key_length / 2;
      int bigk = key_length - smallk;
      uint64_t result = x;
      for (int i = 0; i < kRounds; ++i) {
        uint64_t ab = result >> bigk;
        uint64_t cde = result & ((UINT64_C(1) << bigk) - 1);
        uint64_t cd = cde >> (bigk - smallk);
        uint64_t e = cde & ((UINT64_C(1) << (bigk - smallk)) - 1);
        uint64_t eab = (e << smallk) | ab;
        uint64_t temp = (cd ^ f(ab)) & ((UINT64_C(1) << smallk) - 1);
        result = (temp << bigk) | eab;
      }
      return result;
    };
    auto backward = [f, key_length](uint64_t x) {
      int smallk = key_length / 2;
      int bigk = key_length - smallk;
      uint64_t result = x;
      for (int i = 0; i < kRounds; ++i) {
        uint64_t eab = result & ((UINT64_C(1) << bigk) - 1);
        uint64_t temp = result >> bigk;
        uint64_t e = eab >> smallk;
        uint64_t ab = eab & ((UINT64_C(1) << smallk) - 1);
        uint64_t cd = (f(ab) & ((UINT64_C(1) << smallk) - 1)) ^ temp;
        result = (ab << bigk) | (cd << (bigk - smallk)) | e;
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
      : ndv_(0),
        capacity_(0),
        payload_(nullptr),
        k_(k),
        v_(v),
        d_(d),
        w_(w),
        s_(s),
        log_little_(log_little),
        hash_bijections_(new HashBijection[sizeof...(Ts)]{
            Feistelize(hash_functions, k_)...}) {
    if (not((k_ > 0) && (v_ >= 0) && (d_ >= 2) && (w_ >= 0) && (s_ >= 0) &&
            (log_little_ >= 0) && (hash_bijections_.get() != nullptr) &&
            (sizeof...(Ts) + 1 == d_) && (k_ <= 128) && (v_ <= 128) &&
            (w_ <= 32) && (log_little_ <= 32))) {
      throw std::range_error("bad arguments");
    }
    payload_.reset(new std::unique_ptr<SlotArray[]>[d_]);
    for (int p = 0; p < d_; ++p) {
      payload_[p].reset(new SlotArray[UINT64_C(1) << w_]);
      const int slot_length = s_ + v_ + std::max(0, k_ - w_ - log_little_);
      for (uint64_t q = 0; q < (UINT64_C(1) << w); ++q) {
        payload_[p][q] = SlotArray(slot_length, UINT64_C(1) << log_little_);
        capacity_ += UINT64_C(1) << log_little_;
      }
    }
    assert (0 == FilledSlots());
    assert (Capacity() == (UINT64_C(1) << (log_little_ + w_)) * d);
  }

  uint64_t Hash(int arena, uint64_t key) const {
    assert(key == hash_bijections_[arena - 1].second(
                      hash_bijections_[arena - 1].first(key)));
    return hash_bijections_[arena - 1].first(key);
  }

  uint64_t HashInverse(int arena, uint64_t key) const {
    return hash_bijections_[arena - 1].second(key);
  }

  bool FindExact(uint64_t key, uint64_t value) {
    for (auto i = Find(key); not i.AtEnd(); ++i) {
      if (*i == value) return true;
    }
    return false;
  }

  void Insert(uint64_t key, uint64_t value) {
    //std::cout << "Insert\t" << key << '\t' << value << std::endl;
    if ((1.0 * capacity_)/ndv_ < 1.05) {
      const bool ok = Upsize();
      assert(ok);
    }
    assert (key < (UINT64_C(1) << k_));
    assert(value < (UINT64_C(1) << v_));
    uint64_t current_key = key;
    int p = 0;
    uint64_t iterations = 0;
    while (true) {
      ++iterations;
      if (iterations > ndv_) {
        // std::cout << "slots per item " << (1.0 * capacity_ / ndv_) << std::endl;
        // std::cout << "Upsize\t" << capacity_ << '\t' << log_little_
        //           << std::endl;

        // TODO: we can pick which slot to upsize, if we want.
        const bool ok = Upsize();
        assert(ok);
        iterations = 0;
      }

      const uint64_t q = current_key >> (k_ - w_);
      if (SetLocal(payload_[p][q], current_key, value, false)) {
        ++ndv_;
        assert(FindExact(key, value));
        return;
      }

      const uint64_t pow_ell = payload_[p][q].Capacity();
      const int ell = log_little_ + (pow_ell > (UINT64_C(1) << log_little_));
      assert(pow_ell == (UINT64_C(1) << ell));
      uint64_t r = (current_key >> std::max(0, k_ - w_ - ell)) & (pow_ell - 1);
      r = r << std::max(0, w_ + ell - k_);

      uint64_t i =
          rand() % std::min(pow_ell, (1 << std::max(0, w_ + ell - k_)) +
                                         (1 << s_) - UINT64_C(1));

      KeyValuePair kv = GetRaw(p, q, (r + i) & (pow_ell - 1));
      //std::cout << p << '\t' << q << '\t' << r << '\t' << i << '\t' << kv.key << '\t' << kv.value << std::endl;
      payload_[p][q][(r + i) & (pow_ell - 1)] = 0;
      uint64_t slot_val =
          (value << s_) |
          ((current_key & ((1 << std::max(0, k_ - w_ - ell)) - 1)) << (s_ + v_));
      if (i < (UINT64_C(1) << std::max(0, w_ + ell - k_))) {
        payload_[p][q][(r + i) & (pow_ell - 1)] = slot_val;
      } else {
        payload_[p][q][(r + i) & (pow_ell - 1)] = (slot_val | (i - (1 << std::max(0, w_ + ell - k_)) + 1));
      }
      // const bool ok = SetLocal(payload_[p][q], current_key, value, true);
      // assert(ok);
      assert(FindExact(key, value));
      current_key = kv.key;
      key = (p > 0) ? HashInverse(p, current_key) : current_key;
      value = kv.value;
      p = (p+1) % d_;
      current_key = (p > 0) ? Hash(p, key) : key;
    }
  }

  struct KeyValuePair {
    uint64_t key = 0, value = 0;
  };

  uint64_t SetLocal(SlotArray &sa, uint64_t key, uint64_t value, bool force) {
    // const uint64_t q = key >> (k_ - w_);
    const uint64_t pow_ell = sa.Capacity();
    const int ell = log_little_ + (pow_ell > (UINT64_C(1) << log_little_));
    assert(pow_ell == (UINT64_C(1) << ell));

  CUCKOO:
    uint64_t r = key & ((1 << (k_ - w_)) - 1);
    r = r >> std::max(0, k_ - ell - w_);
    assert(r < pow_ell);
    r = r << std::max(0, w_ + ell - k_);
    assert(r < pow_ell);

    // uint64_t r = (key >> std::max(0, k_ - w_ - ell)) & (pow_ell - 1);
    // r = r << std::max(0, w_ + ell - k_);

    uint64_t val =
        (value << s_) |
        ((key & ((1 << std::max(0, k_ - w_ - ell)) - 1)) << (s_ + v_));
    for (uint64_t i = r; i < (r + (1 << std::max(0, w_ + ell - k_))); ++i) {
      if (0 == sa[i & (pow_ell - 1)]) {
        sa[i & (pow_ell - 1)] = val;
        return (i & (pow_ell - 1)) + 1;
      }
    }

    for (uint64_t i = 1; i < std::min(sa.Capacity(), UINT64_C(1) << s_); ++i) {
      uint64_t r_with =
          (r + (1 << std::max(0, w_ + ell - k_)) - 1) & (pow_ell - 1);
      r_with = (r_with + i) & (pow_ell - 1);
      if (0 == sa[r_with]) {
        sa[r_with] = val | i;
        return r_with + 1;
      }
      if (force && (i > (sa[r_with] & ((1 << s_) - 1)))) {
        value = (sa[r_with] >> s_) & ((1 << v_) - 1);
        key = r_with - (sa[r_with] & ((1 << s_) - 1));
        key = key >> std::max(0, w_ + ell - k_);
        key = key << std::max(0, k_ - ell - w_);
        key = key | (sa[r_with] >> (v_ + s_));
        sa[r_with] = val | i;
        goto CUCKOO;
      }
    }
    return 0;
  }

  KeyValuePair GetRaw(int p, uint64_t q, uint64_t r) const {
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

  KeyValuePair GetOriginal(int p, uint64_t q, uint64_t r) const {
    KeyValuePair result = GetRaw(p, q, r);
    if (p > 0) result.key = HashInverse(p, result.key);
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

    KeyValuePair GetRaw() const { return that_->GetRaw(p_, q_, r_); }

    KeyValuePair GetOriginal() const { return that_->GetOriginal(p_, q_, r_); }

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

    int p_;
    uint64_t q_, r_;

    uint64_t offset_, end_offset_;

    uint64_t key_, current_key_;

    bool AtEnd() {
      return p_ == that_->d_;
    }

    KeyValuePair GetBothRaw() const {
      return that_->GetRaw(
          p_, q_, (r_ + offset_) & (that_->payload_[p_][q_].Capacity() - 1));
    }

    uint64_t operator*() const { return GetBothRaw().value; }

    ResultSetIterator& operator++() {
      if (AtEnd()) return *this;
      if (offset_ + 1 < end_offset_) {
        ++offset_;
        while ((GetBothRaw().key != current_key_) &&
               (offset_ + 1 < end_offset_)) {
          ++offset_;
        }
        if ((GetBothRaw().key == current_key_) && (GetBothRaw().value != 0)) return *this;
      }
      do {
        ++p_;
        if (AtEnd()) return *this;
        current_key_ = that_->Hash(p_, key_);
        AdvanceWithinArena();
      } while (offset_ == end_offset_);
      return *this;
    }

    void AdvanceWithinArena() {
      offset_ = 0;
      q_ = current_key_ >> (that_->k_ - that_->w_);
      const uint64_t pow_ell = that_->payload_[p_][q_].Capacity();
      const int ell =
        that_->log_little_ + (pow_ell > (UINT64_C(1) << that_->log_little_));
      r_ = current_key_ & ((1 << (that_->k_ - that_->w_)) - 1);
      r_ = r_ >> std::max(0, that_->k_ - ell - that_->w_);
      assert(r_ < that_->payload_[p_][q_].Capacity());
      r_ = r_ << std::max(0, that_->w_ + ell - that_->k_);
      assert(r_ < that_->payload_[p_][q_].Capacity());
      end_offset_ = (1 << std::max(0, that_->w_ + ell - that_->k_)) +
                    (UINT64_C(1) << that_->s_) - 1;
      end_offset_ = std::min(end_offset_, that_->payload_[p_][q_].Capacity());
      while ((offset_ < end_offset_) && ((GetBothRaw().value == 0) ||
                                         (GetBothRaw().key != current_key_))) {
        ++offset_;
      }
    }

    ResultSetIterator(const QuotientDysect *that, uint64_t key)
        : that_(that),
          p_(0),
          q_(0),
          r_(0),
          offset_(0),
          end_offset_(0),
          key_(key),
          current_key_(key) {
      AdvanceWithinArena();
      while (offset_ == end_offset_) {
        ++p_;
        if (AtEnd()) return;
        current_key_ = that_->Hash(p_, key_);
        AdvanceWithinArena();
      }
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
          int ndv_before = 0, ndv_after = 0;
          for (uint64_t r = 0; r < (UINT64_C(1) << log_little_); ++r) {
            if (payload_[p][q][r] == 0) continue;
            ++ndv_before;
            KeyValuePair kv = GetRaw(p, q, r);
            const uint64_t ok = SetLocal(replacement, kv.key, kv.value, true);
            // std::cout << "relocate " << r << " to " << ok << " within "
            //           << payload_[p][q].Capacity() << " / "
            //           << replacement.Capacity() << std::endl;
            //assert(ok);
            if (not ok) {  //
              // TODO: shouldn't this be an error?
            }
            ndv_after += (ok != 0);
          }
          if (ndv_before != ndv_after) {
            std::cout << "uhoh " << p << " " << q << " " << ndv_before << " "
                      << ndv_after << std::endl;
          }
          payload_[p][q].Swap(replacement);
          capacity_ += UINT64_C(1) << log_little_;
          if ((p + 1 == d_) && (q + 1 == (1 << w_))) {
            ++log_little_;
            // std::cout << "increased log_little_ to " << log_little_
            //           << std::endl;
          }
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
