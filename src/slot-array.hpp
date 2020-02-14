#pragma once

#include <cassert>
#include <climits>
#include <cstring>
#include <memory>

using std::uint64_t;

struct SlotArray {
  // TODO: use unsigned __int128 for when slots are bigger than CHAR_BIT * sizeof(uint64_t) - CHAR_BIT + 1
  std::unique_ptr<char[]> payload_;

  static uint64_t SpaceUsed(uint64_t capacity, int width) {
    return (((capacity * width + sizeof(uint64_t) - 1) / sizeof(uint64_t)) *
                sizeof(uint64_t) +
            CHAR_BIT - 1) /
           CHAR_BIT;
  }
  static uint64_t SpaceUsed(int, uint64_t) = delete;

  explicit SlotArray() : payload_(nullptr) {}

  explicit SlotArray(uint64_t capacity, int width) : payload_(nullptr) {
    payload_.reset(new char[SpaceUsed(capacity, width)]());
  }

  explicit SlotArray(int, uint64_t) = delete;

  uint64_t operator()(int width, uint64_t index) const {
    uint64_t result;
    std::memcpy(&result, &payload_[(index * width) / CHAR_BIT], sizeof(result));
    result =
        result >> ((index * width) - (CHAR_BIT * ((index * width) / CHAR_BIT)));
    result = result & ((UINT64_C(1) << width) - 1);
    return result;
  }

  uint64_t operator()(int width, uint64_t index) {
    return static_cast<const SlotArray &>(*this)(width, index);
  }

  uint64_t operator()(uint64_t, int) = delete;
  uint64_t operator()(uint64_t, int) const = delete;

  void Set(int width, uint64_t index, uint64_t value) {
    assert(value < (UINT64_C(1) << width));
    const int offset =
        ((index * width) - (CHAR_BIT * ((index * width) / CHAR_BIT)));
    const uint64_t mask = ((UINT64_C(1) << width) - 1) << offset;
    uint64_t before;
    std::memcpy(&before, &payload_[(index * width) / CHAR_BIT], sizeof(before));
    before = before & ~mask;
    before = before | (value << offset);
    std::memcpy(&payload_[(index * width) / CHAR_BIT], &before, sizeof(before));
  }

  void Set(int, uint64_t, int, uint64_t) = delete;
};
