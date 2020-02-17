#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <memory>

using std::uint64_t;

struct SlotArray {
  // TODO: use unsigned __int128 for when slots are bigger than CHAR_BIT * sizeof(uint64_t) - CHAR_BIT
  int width_;
  uint64_t capacity_;
  std::unique_ptr<char[]> payload_;

  void Swap(SlotArray &that) {
    using std::swap;
    swap(width_, that.width_);
    swap(capacity_, that.capacity_);
    swap(payload_, that.payload_);
  }

  uint64_t Capacity() const { return capacity_; }

  uint64_t PayloadSpaceUsed(){
    return (((capacity_ * width_ + sizeof(uint64_t) * CHAR_BIT - 1) /
             (sizeof(uint64_t) * CHAR_BIT)) +
            1) *
           sizeof(uint64_t);
  }

  uint64_t SpaceUsed() {
    return PayloadSpaceUsed() + sizeof(*this);
  }

  explicit SlotArray() : payload_(nullptr) {}

  explicit SlotArray(int width, uint64_t capacity)
      : width_(width), capacity_(capacity), payload_(nullptr) {
    assert(width_ > 0);
    payload_.reset(new char[PayloadSpaceUsed()]());
  }

  explicit SlotArray(uint64_t, int) = delete;

  template <typename T>
  struct ReferenceBase {
    T *that_;
    uint64_t index_;

    operator uint64_t() const { return that_->Get(index_); }

    ReferenceBase &operator=(uint64_t value) {
      that_->Set(index_, value);
      return *this;
    }
  };

  using Reference = ReferenceBase<SlotArray>;
  using ConstReference = ReferenceBase<const SlotArray>;

  Reference operator[](uint64_t index) {
    assert(index < capacity_);
    return Reference{this, index};
  }

  ConstReference operator[](uint64_t index) const {
    assert(index < capacity_);
    return ConstReference{this, index};
  }

  uint64_t Get(uint64_t index) const {
    uint64_t result;
    std::memcpy(&result, &payload_[(index * width_) / CHAR_BIT],
                sizeof(result));
    result = result >>
             ((index * width_) - (CHAR_BIT * ((index * width_) / CHAR_BIT)));
    result = result & ((UINT64_C(1) << width_) - 1);
    return result;
  }

  void Set(uint64_t index, uint64_t value) {
    assert(value < (UINT64_C(1) << width_));
    const int offset =
        ((index * width_) - (CHAR_BIT * ((index * width_) / CHAR_BIT)));
    const uint64_t mask = ((UINT64_C(1) << width_) - 1) << offset;
    uint64_t before;
    std::memcpy(&before, &payload_[(index * width_) / CHAR_BIT], sizeof(before));
    before = before & ~mask;
    before = before | (value << offset);
    std::memcpy(&payload_[(index * width_) / CHAR_BIT], &before, sizeof(before));
  }
};
