// This file represents a degenerate version of crate dictionaries as
// in "Fully-Dynamic Space-Efficient Dictionaries and Filters with
// Constant Number of Memory Accesses",
// https://arxiv.org/abs/1911.05060
//
// In particular, it works with a single crate, doesn't support the
// spare structure, expects no buckets will overflow, and doesn't
// support deletion. Each Pocket Dictionary (which is isomorphic to the
// Elias-Fano structure) represents up to 51 fingerprints in the range [0, 50 *
// 256).

#include <assert.h>
#include <iostream>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <unordered_set>

#include "immintrin.h"

#include "linear-probing.h"

// returns the position (starting from 0) of the jth set bit of x.
inline uint64_t select64(uint64_t x, int64_t j) {
  assert(j >= 0);
  assert(j < 64);
  const uint64_t y = _pdep_u64(UINT64_C(1) << j, x);
  return _tzcnt_u64(y);
}

// similar to select64, but now j can be as low as -1. Useful for branchless
// selection
inline uint64_t select64_alt(uint64_t x, int64_t j) {
  assert(j >= -1);
  assert(j < 64);
  const uint64_t y = _pdep_u64((UINT64_C(1) << (j + 1)) >> 1, x);
  return 63 & _tzcnt_u64(y);
}

// returns the position (starting from 0) of the jth set bit of x.
inline uint64_t select128(unsigned __int128 x, int64_t j) {
  const int64_t pop = _mm_popcnt_u64(x);
  if (j < pop) return select64(x, j);
  return 64 + select64(x >> 64, j - pop);
}

// returns the position (starting from 0) of the jth set bit of x. Takes as
// input the popcnt of the low 64 bits of x.
inline uint64_t select128withPop64(unsigned __int128 x, int64_t j,
                                   int64_t pop) {
  if (j < pop) return select64(x, j);
  return 64 + select64(x >> 64, j - pop);
}

// Useful for gdb, which doesn't want to call _mm_popcnt_u64 directly
int popcount64(uint64_t x) { return _mm_popcnt_u64(x); }

int popcount128(unsigned __int128 x) {
  const uint64_t hi = x >> 64;
  const uint64_t lo = x;
  return popcount64(lo) + popcount64(hi);
}

// find an 8-bit value in a pocket dictionary with quotients in [0,50) and 51
// values
inline bool pd_find_50(int64_t quot, uint8_t rem, const __m512i *pd) {
  assert(0 == (reinterpret_cast<uintptr_t>(pd) % 64));
  assert(quot < 50);
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;
  // [begin,end) are the zeros in the header that correspond to the fingerprints
  // with quotient quot.
  const int64_t pop = _mm_popcnt_u64(header);
  const uint64_t begin =
      (quot ? (select128withPop64(header, quot - 1, pop) + 1) : 0) - quot;
  const uint64_t end = select128withPop64(header, quot, pop) - quot;
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);
  // round up to remove the header
  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// Like pd_find_50, but manually inlines the calls to select128 to reduce
// unneeded computation.
inline bool pd_find_50_alt(int64_t quot, uint8_t rem, const __m512i *pd) {
  assert(0 == (reinterpret_cast<uintptr_t>(pd) % 64));
  assert(quot < 50);
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;
  // [begin,end) are the zeros in the header that correspond to the fingerprints
  // with quotient quot.

  uint64_t begin, end;
  if (0 == quot) {
    begin = 0;
    end = select64(header, 0);
  } else {
    const int64_t pop = _mm_popcnt_u64(header);
    if (quot - 1 >= pop) {
      begin = 64 + select64(header >> 64, quot - 1 - pop) + 1 - quot;
      end = 64 + select64(header >> 64, quot - pop) - quot;
    } else {
      begin = select64(header, quot - 1) + 1 - quot;
      if (quot >= pop) {
        end = 64 + select64(header >> 64, quot - pop) - quot;
      } else {
        end = select64(header, quot) - quot;
      }
    }
  }
  assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
  assert(end == select128(header, quot) - quot);
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);
  // round up to remove the header
  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// Like pd_find_50_alt but further simplifies the computation of begin and end.
inline bool pd_find_50_alt2(int64_t quot, uint8_t rem, const __m512i *pd) {
  assert(0 == (reinterpret_cast<uintptr_t>(pd) % 64));
  assert(quot < 50);
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;
  // [begin,end) are the zeros in the header that correspond to the fingerprints
  // with quotient quot.

  uint64_t begin = 0;
  if (quot > 0) {
    const int64_t pop = _mm_popcnt_u64(header);
    if (quot - 1 < pop) {
      begin = select64(header, quot - 1) + 1 - quot;
    } else {
      begin = 64 + select64(header >> 64, quot - 1 - pop) + 1 - quot;
    }
  }
  const uint64_t end = begin + _tzcnt_u64(header >> (begin + quot));
  assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
  assert(end == select128(header, quot) - quot);
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);
  // round up to remove the header
  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// Like pd_find_50_alt2, but attempts to make the computation of begin and end
// use cmov to make it less branchy
inline bool pd_find_50_alt3(int64_t quot, uint8_t rem, const __m512i *pd) {
  assert(0 == (reinterpret_cast<uintptr_t>(pd) % 64));
  assert(quot < 50);
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;

  uint64_t begin = 0;
  if (quot > 0) {
    const int64_t pop = _mm_popcnt_u64(header);
    const auto q1 = quot - 1;
    begin = (q1 < pop) ? select64(header, q1) : (64 + select64(header >> 64, q1-pop));
    begin -= q1;
  }
  const uint64_t end = begin + _tzcnt_u64(header >> (begin + quot));
  assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
  assert(end == select128(header, quot) - quot);
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);

  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// Like pd_find_50_alt3, but uses shift on 128-bit values to remove another conditional.
inline bool pd_find_50_alt4(int64_t quot, uint8_t rem, const __m512i *pd) {
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;

  uint64_t begin = 0;
  if (quot > 0) {
    auto p = _mm_popcnt_u64(header & ((UINT64_C(1) << (quot - 1)) - 1));
    begin = select64(header >> (quot - 1), quot - 1 - p);
  }
  const uint64_t end = begin + _tzcnt_u64(header >> (begin + quot));
  assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
  assert(end == select128(header, quot) - quot);
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);

  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// Like pd_find_50_alt4, but hopefully mostly branchless. One known
// cmov is still present in the right shift by (begin + quot), which
// could be >= 64
//
// While this might be slower than some other alternatives, it is a
// step towards vectorization.
//
// This would be simpler if the format made the header start with a 1. That
// would require modifying the insert procedure.
inline bool pd_find_50_alt5(int64_t quot, uint8_t rem, const __m512i *pd) {
  unsigned __int128 header = 0;
  memcpy(&header, pd, sizeof(header));
  constexpr unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;

  uint64_t begin = 0;

  uint64_t mask = ((UINT64_C(1) << quot) - 1) >> 1;
  auto p = _mm_popcnt_u64(header & mask);
  // & quot with 63 to let remove testb instruction.
  begin = select64_alt((header << 1) >> (quot & 63), quot - 1 - p);

  const uint64_t end = begin + _tzcnt_u64(header >> (begin + quot));
  assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
  assert(end == select128(header, quot) - quot);
  assert(begin <= end);
  assert(end <= 51);
  const __m512i target = _mm512_set1_epi8(rem);
  uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);

  constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kHeaderBytes < sizeof(header));
  v = v >> kHeaderBytes;
  return (v & ((UINT64_C(1) << end) - 1)) >> begin;
}

// inline uint8_t pd_find_50_x8(__m512i quot, const uint8_t rem[8],
//                              const __m512i *base, __m256i pd_idxs) {
//   uint8_t result = 0;
//   pd_idxs =* sizeof(base[0]);
//   auto pd_idxs_back = pd_idxs + sizeof(uint64_t);
//   __m512i fake_header[2];
//   fake_header[0] = _mm512_i32gather_epi64(pd_idxs, base, 1);
//   fake_header[1] = _mm512_i32gather_epi64(pd_idxs_back, base, 1);

//   constexpr uint64_t kLeftoverMaskLo = (UINT64_C(1) << (50 + 51 - 64)) - 1;
//   constexpr __m512i kLeftoverMask = {
//       kLeftoverMaskLo, kLeftoverMaskLo, kLeftoverMaskLo, kLeftoverMaskLo,
//       kLeftoverMaskLo, kLeftoverMaskLo, kLeftoverMaskLo, kLeftoverMaskLo};
//   fake_header[0] &= kLeftoverMask;

//   constexpr __m512i ones = {1,1,1,1,1,1,1,1};
//   auto mask = _mm512_srli_epi64(
//       _mm512_sub_epi64(_mm512_sllv_epi64(ones, quot), ones), 1);
//   auto pops = _mm512_popcnt_epi64(fake_header[1] & mask);

//   __m512i real_header[2];
//   real_header[0] = _mm512_unpackhi_epi64(fake_header[1], fake_header[0]);
//   real_header[1] = _mm512_unpacklo_epi64(fake_header[1], fake_header[0]);

//   // STOPPED HERE
  
//   uint64_t begin = 0;


//   // & quot with 63 to let remove testb instruction.
//   begin = select64_alt((header << 1) >> (quot & 63), quot - 1 - p);

//   const uint64_t end = begin + _tzcnt_u64(header >> (begin + quot));
//   assert(begin == (quot ? (select128(header, quot - 1) + 1) : 0) - quot);
//   assert(end == select128(header, quot) - quot);
//   assert(begin <= end);
//   assert(end <= 51);
//   const __m512i target = _mm512_set1_epi8(rem);
//   uint64_t v = _mm512_cmpeq_epu8_mask(target, *pd);

//   constexpr unsigned kHeaderBytes = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
//   assert(kHeaderBytes < sizeof(header));
//   v = v >> kHeaderBytes;
//   return (v & ((UINT64_C(1) << end) - 1)) >> begin;
// }

int pd_popcount(const __m512i *pd) {
  uint64_t header_end;
  memcpy(&header_end, reinterpret_cast<const uint64_t *>(pd) + 1,
         sizeof(header_end));
  constexpr uint64_t kLeftoverMask = (UINT64_C(1) << (50 + 51 - 64)) - 1;
  header_end = header_end & kLeftoverMask;
  const int result = 128 - 51 - _lzcnt_u64(header_end) + 1;
  return result;
}

bool pd_full(const __m512i *pd) {
  uint64_t header_end;
  memcpy(&header_end, reinterpret_cast<const uint64_t *>(pd) + 1,
         sizeof(header_end));
  return 1 & (header_end >> (50 + 51 - 64 - 1));
}

// insert a pair of a quotient (mod 50) and an 8-bit remainder in a pocket
// dictionary. Returns false if the dictionary is full.
inline bool pd_add_50(int64_t quot, uint8_t rem, __m512i *pd) {
  assert(0 == (reinterpret_cast<uintptr_t>(pd) % 64));
  assert(quot < 50);
  // The header has size 50 + 51
  unsigned __int128 header = 0;
  // We need to copy (50+51) bits, but we copy slightly more and mask out the
  // ones we don't care about.
  //
  // memcpy is the only defined punning operation
  const unsigned kBytes2copy = (50 + 51 + CHAR_BIT - 1) / CHAR_BIT;
  assert(kBytes2copy < sizeof(header));
  memcpy(&header, pd, kBytes2copy);
  // Number of bits to keep. Requires little-endianness
  // const unsigned __int128 kLeftover = sizeof(header) * CHAR_BIT - 50 - 51;
  const unsigned __int128 kLeftoverMask =
      (((unsigned __int128)1) << (50 + 51)) - 1;
  header = header & kLeftoverMask;
  assert(50 == popcount128(header));
  const unsigned fill = select128(header, 50 - 1) - (50 - 1);
  assert((fill <= 14) || (fill == pd_popcount(pd)));
  assert((fill == 51) == pd_full(pd));
  if (fill == 51) return false;
  // [begin,end) are the zeros in the header that correspond to the fingerprints
  // with quotient quot.
  const uint64_t begin = quot ? (select128(header, quot - 1) + 1) : 0;
  const uint64_t end = select128(header, quot);
  assert(begin <= end);
  assert(end <= 50 + 51);
  unsigned __int128 new_header =
      header & ((((unsigned __int128)1) << begin) - 1);
  new_header |= ((header >> end) << (end + 1));
  assert(popcount128(new_header) == 50);
  assert(select128(new_header, 50 - 1) - (50 - 1) == fill + 1);
  memcpy(pd, &new_header, kBytes2copy);
  const uint64_t begin_fingerprint = begin - quot;
  const uint64_t end_fingerprint = end - quot;
  assert(begin_fingerprint <= end_fingerprint);
  assert(end_fingerprint <= 51);
  uint64_t i = begin_fingerprint;
  for (; i < end_fingerprint; ++i) {
    if (rem <= ((const uint8_t *)pd)[kBytes2copy + i]) break;
  }
  assert((i == end_fingerprint) ||
         (rem <= ((const uint8_t *)pd)[kBytes2copy + i]));
  memmove(&((uint8_t *)pd)[kBytes2copy + i + 1],
          &((const uint8_t *)pd)[kBytes2copy + i],
          sizeof(*pd) - (kBytes2copy + i + 1));
  ((uint8_t *)pd)[kBytes2copy + i] = rem;

  assert(pd_find_50(quot, rem, pd) == pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50_alt2(quot, rem, pd) == pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50_alt3(quot, rem, pd) == pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50_alt4(quot, rem, pd) == pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50_alt5(quot, rem, pd) == pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50(quot, rem, pd));
  assert(pd_find_50_alt(quot, rem, pd));
  assert(pd_find_50_alt2(quot, rem, pd));
  assert(pd_find_50_alt3(quot, rem, pd));
  assert(pd_find_50_alt4(quot, rem, pd));
  assert(pd_find_50_alt5(quot, rem, pd));
  return true;
}

#include <algorithm>

template <bool FINDER(int64_t quot, uint8_t rem, const __m512i *pd)>
struct GenericCrate {
  uint64_t bucket_count_;
  __m512i *buckets_;

  SimdSizedDict spare_;
  // 3.3e-3 for fill 40
  // 4.6e-3 for fill 41
  // 7e-3 for fill 42
  // 0.01 for fill 43
  // 0.013 for fill 44
  // 0.0165 for fill 45
  // 0.021 for fill 46
  // 0.026 for fill 46

  //SizedDict<uint32_t, uint64_t> spare_;
  //Dict<uint32_t, uint64_t> spare_;

  uint64_t SizeInBytes() const {
    return sizeof(*buckets_) * bucket_count_ + spare_.SizeInBytes();
  }
  GenericCrate(size_t add_count)
      : spare_(0.013 * add_count)
  {
    bucket_count_ = add_count / 44;
    buckets_ = new __m512i[bucket_count_];
    std::fill(buckets_, buckets_ + bucket_count_,
              __m512i{(INT64_C(1) << 50) - 1, 0, 0, 0, 0, 0, 0, 0});
  }
  ~GenericCrate() {
    std::cout << std::endl
              << "bucket bytes\t" << sizeof(*buckets_) * bucket_count_ << std::endl
              << "spare_ bytes\t" << spare_.SizeInBytes() << std::endl
              << "spare_ NDV\t" << spare_.ndv_ << std::endl;
    delete[] buckets_;
  }
  bool Add(uint64_t key) {
    bool result = pd_add_50(
        ((key >> 40) * 50) >> 24, key >> 32,
        &buckets_[(static_cast<uint64_t>(static_cast<uint32_t>(key)) *
                   bucket_count_) >>
                  32]);
    if (not result) result = spare_.Insert(key);
    return result;
  }
  bool Contain(uint64_t key) const {
    return FINDER(((key >> 40) * 50) >> 24, key >> 32,
                  &buckets_[(static_cast<uint64_t>(static_cast<uint32_t>(key)) *
                             bucket_count_) >>
                            32]);
  }
  uint64_t Contain64(const uint64_t keys[64]) const {
    uint32_t indexes[64];
    for (int i = 0; i < 64; ++i) {
      indexes[i] = (static_cast<uint64_t>(static_cast<uint32_t>(keys[i])) *
                    bucket_count_) >>
                   32;
    }
    for (int i = 0; i < 64; ++i) {
      __builtin_prefetch(&buckets_[indexes[i]], 0 /* read only */,
                         0 /*non-temporal */);
    }
    uint64_t result = 0;
    uint64_t try_backup = 0;
    for (int i = 0; i < 64; ++i) {
      uint64_t found = FINDER(((keys[i] >> 40) * 50) >> 24, keys[i] >> 32,
                              &buckets_[indexes[i]]);
      if (0 == found && pd_full(&buckets_[indexes[i]])) {
        if (0 == keys[i]) {
          result |= static_cast<uint64_t>(spare_.has_zero_) << i;
        } else {
          try_backup |= (UINT64_C(1) << i);
          indexes[i] = spare_.Hash(keys[i]);
          __builtin_prefetch(&spare_.payload_512_[indexes[i]],
                             0 /* read only */, 0 /*non-temporal */);
        }
      }
      result |= found << i;
    }
    while (try_backup > 0) {
      auto i = _tzcnt_u64(try_backup);
      result |=
          (static_cast<bool>(spare_.ContainsKeyWithHash(keys[i], indexes[i]))
           << i);
      try_backup = try_backup & (try_backup - 1);
    }

    return result;
  }

  uint64_t Contain64_alt(const uint64_t keys[64]) const {
    uint32_t indexes[64];
    for (int i = 0; i < 64; ++i) {
      indexes[i] = (static_cast<uint64_t>(static_cast<uint32_t>(keys[i])) *
                    bucket_count_) >>
                   32;
      __builtin_prefetch(&buckets_[indexes[i]], 0 /* read only */,
                         0 /*non-temporal */);
    }
    uint64_t result = 0;
    for (int i = 0; i < 64; ++i) {
      result |=
          (static_cast<uint64_t>(FINDER(((keys[i] >> 40) * 50) >> 24,
                                        keys[i] >> 32, &buckets_[indexes[i]]))
           << i);
    }
    return result;
  }

  unsigned __int128 Contain128(const uint64_t keys[128]) const {
    uint32_t indexes[128];
    for (int i = 0; i < 128; ++i) {
      indexes[i] = (static_cast<uint64_t>(static_cast<uint32_t>(keys[i])) *
                    bucket_count_) >>
                   32;
    }
    for (int i = 0; i < 128; ++i) {
      __builtin_prefetch(&buckets_[indexes[i]], 0 /* read only */,
                         0 /*non-temporal */);
    }
    unsigned __int128 result = 0;
    for (int i = 0; i < 128; ++i) {
      const bool tmp = FINDER(((keys[i] >> 40) * 50) >> 24, keys[i] >> 32,
                              &buckets_[indexes[i]]);

      result |= (static_cast<unsigned __int128>(tmp) << i);
      assert (tmp == Contain(keys[i]));
    }
    return result;
  }

};
