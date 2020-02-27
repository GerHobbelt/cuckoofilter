#pragma once

#include <assert.h>
#include <algorithm>
#include <sstream>

#include "bitsutil.h"
#include "debug.h"
#include "hashutil.h"
#include "packedtable.h"
#include "printutil.h"
#include "singletable.h"

namespace cuckoofilter {

// A cuckoo filter class exposes a Bloomier filter interface,
// providing methods of Add, Delete, Contain. It takes three
// template parameters:
//   ItemType:  the type of item you want to insert
//   bits_per_item: how many bits each item is hashed into
//   TableType: the storage of table, SingleTable by default, and
// PackedTable to enable semi-sorting
template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType = SingleTable,
          typename HashFamily = TwoIndependentMultiplyShift>
class CuckooBatchPrefetch {
  // Storage of items
  TableType<bits_per_item> *table_;

  // Number of items stored
  size_t num_items_;

  typedef struct {
    size_t index;
    uint32_t tag;
    bool used;
  } VictimCache;

  VictimCache victim_;

  HashFamily hasher_;

  inline size_t IndexHash(uint32_t hv) const {
    // table_->num_buckets is always a power of two, so modulo can be replaced
    // with
    // bitwise-and:
    return hv & (table_->NumBuckets() - 1);
  }

  inline uint32_t TagHash(uint32_t hv) const {
    uint32_t tag;
    tag = hv & ((1ULL << bits_per_item) - 1);
    tag += (tag == 0);
    return tag;
  }

  inline void GenerateIndexTagHash(const ItemType& item, size_t* index,
                                   uint32_t* tag) const {
    const uint64_t hash = hasher_(item);
    *index = IndexHash(hash >> 32);
    *tag = TagHash(hash);
  }

  inline size_t AltIndex(const size_t index, const uint32_t tag) const {
    // NOTE(binfan): originally we use:
    // index ^ HashUtil::BobHash((const void*) (&tag), 4)) & table_->INDEXMASK;
    // now doing a quick-n-dirty way:
    // 0x5bd1e995 is the hash constant from MurmurHash2
    return IndexHash((uint32_t)(index ^ (tag * 0x5bd1e995)));
  }

  Status AddImpl(const size_t i, const uint32_t tag);

  // load factor is the fraction of occupancy
  double LoadFactor() const { return 1.0 * Size() / table_->SizeInTags(); }

  double BitsPerItem() const { return 8.0 * table_->SizeInBytes() / Size(); }

 public:
  explicit CuckooBatchPrefetch(const size_t max_num_keys) : num_items_(0), victim_(), hasher_() {
    size_t assoc = 4;
    size_t num_buckets = upperpower2(std::max<uint64_t>(1, max_num_keys / assoc));
    double frac = (double)max_num_keys / num_buckets / assoc;
    if (frac > 0.96) {
      num_buckets <<= 1;
    }
    victim_.used = false;
    table_ = new TableType<bits_per_item>(num_buckets);
  }

  ~CuckooBatchPrefetch() { delete table_; }

  // Add an item to the filter.
  Status Add(const ItemType &item);

  // Report if the item is inserted, with false positive rate.
  Status Contain(const ItemType &item) const;

  struct PrefetchSetupResult {
    size_t i1, i2;
    uint32_t tag;
    bool found;
  };

  PrefetchSetupResult PrefetchSetup(const ItemType &item) const;
  void Prefetch(size_t i) const;
  Status PrefetchContain(size_t i, uint32_t tag) const;

  uint64_t Contain64(const ItemType items[64]) const {
    uint64_t result = 0;
    PrefetchSetupResult setups[64];
    for (int i = 0; i < 64; ++i) {
      setups[i] = PrefetchSetup(items[i]);
    }
    for (int i = 0; i < 64; ++i) {
      if (setups[i].found) {
        result |= (UINT64_C(1) << i);
      } else {
        Prefetch(setups[i].i1);
      }
    }
    for (int i = 0; i < 64; ++i) {
      if (not setups[i].found) {
        if (Ok == PrefetchContain(setups[i].i1, setups[i].tag)) {
          result |= (UINT64_C(1) << i);
        } else {
          Prefetch(setups[i].i2);
        }
      }
    }
    for (int i = 0; i < 64; ++i) {
      if (not setups[i].found) {
        if (Ok == PrefetchContain(setups[i].i2, setups[i].tag)) {
          result |= (UINT64_C(1) << i);
        }
      }
    }
    return result;
  }

  uint64_t Contain64_aggressive(const ItemType items[64]) const {
    uint64_t result = 0;
    PrefetchSetupResult setups[64];
    for (int i = 0; i < 64; ++i) {
      setups[i] = PrefetchSetup(items[i]);
    }
    for (int i = 0; i < 64; ++i) {
      if (setups[i].found) {
        result |= (UINT64_C(1) << i);
      } else {
        Prefetch(setups[i].i1);
        Prefetch(setups[i].i2);
      }
    }
    for (int i = 0; i < 64; ++i) {
      if (not setups[i].found) {
        if ((Ok == PrefetchContain(setups[i].i1, setups[i].tag)) ||
            (Ok == PrefetchContain(setups[i].i2, setups[i].tag))) {
          result |= (UINT64_C(1) << i);
        }
      }
    }
    return result;
  }

  // Delete an key from the filter
  Status Delete(const ItemType &item);

  /* methods for providing stats  */
  // summary infomation
  std::string Info() const;

  // number of current inserted items;
  size_t Size() const { return num_items_; }

  // size of the filter in bytes.
  size_t SizeInBytes() const { return table_->SizeInBytes(); }
};

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
Status CuckooBatchPrefetch<ItemType, bits_per_item, TableType, HashFamily>::Add(
    const ItemType &item) {
  size_t i;
  uint32_t tag;

  if (victim_.used) {
    return NotEnoughSpace;
  }

  GenerateIndexTagHash(item, &i, &tag);
  return AddImpl(i, tag);
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
Status CuckooBatchPrefetch<ItemType, bits_per_item, TableType, HashFamily>::AddImpl(
    const size_t i, const uint32_t tag) {
  size_t curindex = i;
  uint32_t curtag = tag;
  uint32_t oldtag;

  for (uint32_t count = 0; count < kMaxCuckooCount; count++) {
    bool kickout = count > 0;
    oldtag = 0;
    if (table_->InsertTagToBucket(curindex, curtag, kickout, oldtag)) {
      num_items_++;
      return Ok;
    }
    if (kickout) {
      curtag = oldtag;
    }
    curindex = AltIndex(curindex, curtag);
  }

  victim_.index = curindex;
  victim_.tag = curtag;
  victim_.used = true;
  return Ok;
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
Status CuckooBatchPrefetch<ItemType, bits_per_item, TableType, HashFamily>::Contain(
    const ItemType &key) const {
  bool found = false;
  size_t i1, i2;
  uint32_t tag;

  GenerateIndexTagHash(key, &i1, &tag);
  i2 = AltIndex(i1, tag);

  assert(i1 == AltIndex(i2, tag));

  found = victim_.used && (tag == victim_.tag) &&
          (i1 == victim_.index || i2 == victim_.index);

  if (found || table_->FindTagInBuckets(i1, i2, tag)) {
    return Ok;
  } else {
    return NotFound;
  }
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
typename CuckooBatchPrefetch<ItemType, bits_per_item, TableType,
                    HashFamily>::PrefetchSetupResult
CuckooBatchPrefetch<ItemType, bits_per_item, TableType,
                    HashFamily>::PrefetchSetup(const ItemType &key) const {
  PrefetchSetupResult result;

  GenerateIndexTagHash(key, &result.i1, &result.tag);
  result.i2 = AltIndex(result.i1, result.tag);

  assert(result.i1 == AltIndex(result.i2, result.tag));

  result.found = victim_.used && (result.tag == victim_.tag) &&
          (result.i1 == victim_.index || result.i2 == victim_.index);

  return result;
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
void CuckooBatchPrefetch<ItemType, bits_per_item, TableType,
                         HashFamily>::Prefetch(size_t i) const {
  __builtin_prefetch(&table_->buckets_[i], 0 /* read only */,
                     0 /*non-temporal */);
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
Status CuckooBatchPrefetch<ItemType, bits_per_item, TableType,
                         HashFamily>::PrefetchContain(size_t i,
                                                      uint32_t tag) const {
  return table_->FindTagInBucket(i,tag) ? Ok : NotFound;
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
Status CuckooBatchPrefetch<ItemType, bits_per_item, TableType, HashFamily>::Delete(
    const ItemType &key) {
  size_t i1, i2;
  uint32_t tag;

  GenerateIndexTagHash(key, &i1, &tag);
  i2 = AltIndex(i1, tag);

  if (table_->DeleteTagFromBucket(i1, tag)) {
    num_items_--;
    goto TryEliminateVictim;
  } else if (table_->DeleteTagFromBucket(i2, tag)) {
    num_items_--;
    goto TryEliminateVictim;
  } else if (victim_.used && tag == victim_.tag &&
             (i1 == victim_.index || i2 == victim_.index)) {
    // num_items_--;
    victim_.used = false;
    return Ok;
  } else {
    return NotFound;
  }
TryEliminateVictim:
  if (victim_.used) {
    victim_.used = false;
    size_t i = victim_.index;
    uint32_t tag = victim_.tag;
    AddImpl(i, tag);
  }
  return Ok;
}

template <typename ItemType, size_t bits_per_item,
          template <size_t> class TableType, typename HashFamily>
std::string CuckooBatchPrefetch<ItemType, bits_per_item, TableType, HashFamily>::Info() const {
  std::stringstream ss;
  ss << "CuckooBatchPrefetch Status:\n"
     << "\t\t" << table_->Info() << "\n"
     << "\t\tKeys stored: " << Size() << "\n"
     << "\t\tLoad factor: " << LoadFactor() << "\n"
     << "\t\tHashtable size: " << (table_->SizeInBytes() >> 10) << " KB\n";
  if (Size() > 0) {
    ss << "\t\tbit/key:   " << BitsPerItem() << "\n";
  } else {
    ss << "\t\tbit/key:   N/A\n";
  }
  return ss.str();
}
}  // namespace cuckoofilter
