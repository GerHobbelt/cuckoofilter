#ifndef XOR_FILTER_XOR_FILTER_H_
#define XOR_FILTER_XOR_FILTER_H_

#include <assert.h>
#include <algorithm>

#include "debug.h"
#include "hashutil.h"
#include "printutil.h"

using namespace std;
using namespace cuckoofilter;

namespace xorfilter {
// status returned by a xor filter operation
enum Status {
  Ok = 0,
  NotFound = 1,
  NotEnoughSpace = 2,
  NotSupported = 3,
};

inline uint64_t hash64(uint64_t x) {
    // TODO need to check if this is "fair" (cuckoo filter uses TwoIndependentMultiplyShift)
    x = x * 0xbf58476d1ce4e5b9L;
    x = x ^ (x >> 31);
    return x;

    // mix64
    // x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9L;
    // x = (x ^ (x >> 27)) * 0x94d049bb133111ebL;
    // x = x ^ (x >> 31);
    // return x;
}

inline uint32_t fingerprint(uint64_t hash) {
    return (uint32_t) (hash & ((1 << 8) - 1));
}

inline uint32_t reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t) (((uint64_t) hash * n) >> 32);
}

size_t getHash(uint64_t key, int hashIndex, int index, int blockLength) {
    uint64_t hash = hash64(key + hashIndex);
    uint32_t r;
    switch(index) {
    case 0:
        r = (uint32_t) (hash);
        break;
    case 1:
        r = (uint32_t) (hash >> 16);
        break;
    default:
        r = (uint32_t) (hash >> 32);
        break;
    }
    r = reduce(r, blockLength);
    r = r + index * blockLength;
    return (size_t) r;
}

size_t getHashFromHash(uint64_t hash, int index, int blockLength) {
    uint32_t r;
    switch(index) {
    case 0:
        r = (uint32_t) (hash);
        break;
    case 1:
        r = (uint32_t) (hash >> 16);
        break;
    default:
        r = (uint32_t) (hash >> 32);
        break;
    }
    r = reduce(r, blockLength);
    r = r + index * blockLength;
    return (size_t) r;
}

template <typename ItemType, size_t bits_per_item,
          typename HashFamily = TwoIndependentMultiplyShift>
class XorFilter {

  size_t size;
  size_t arrayLength;
  size_t blockLength;
  uint32_t hashIndex;
  uint8_t *fingerprints;

  HashFamily hasher;

  double BitsPerItem() const { return 8.0; }

 public:
  explicit XorFilter(const size_t size) : hasher() {
    this->size = size;
    this->arrayLength = 3 + 1.23 * size;
    this->blockLength = arrayLength / 3;
    fingerprints = new uint8_t[arrayLength];
  }

  ~XorFilter() { delete[] fingerprints; }

  Status AddAll(const vector<ItemType> data, const size_t start, const size_t end);

  // Report if the item is inserted, with false positive rate.
  Status Contain(const ItemType &item) const;

  /* methods for providing stats  */
  // summary infomation
  std::string Info() const;

  // number of current inserted items;
  size_t Size() const { return size; }

  // size of the filter in bytes.
  size_t SizeInBytes() const { return arrayLength; }
};

template <typename ItemType, size_t bits_per_item,
          typename HashFamily>
Status XorFilter<ItemType, bits_per_item, HashFamily>::AddAll(
    const vector<ItemType> keys, const size_t start, const size_t end) {
    int m = arrayLength;
    uint64_t* reverseOrder = new uint64_t[size];
    uint8_t* reverseH = new uint8_t[size];
    size_t reverseOrderPos;
    int hashIndex = 0;
    uint64_t* t2 = new uint64_t[m];
    uint8_t* t2count = new uint8_t[m];
    while (true) {
        memset(t2count, 0, sizeof(uint8_t[m]));
        memset(t2, 0, sizeof(uint64_t[m]));
        for(size_t i = start; i < end; i++) {
            uint64_t k = keys[i];
            uint64_t hash = hash64(k + hashIndex);
            int h0 = reduce((int) (hash), blockLength);
            int h1 = reduce((int) (hash >> 16), blockLength) + blockLength;
            int h2 = reduce((int) (hash >> 32), blockLength) + 2 * blockLength;
            t2count[h0]++;
            t2[h0] ^= hash;
            t2count[h1]++;
            t2[h1] ^= hash;
            t2count[h2]++;
            t2[h2] ^= hash;
        }
        reverseOrderPos = 0;
        int* alone = new int[arrayLength];
        int alonePos = 0;
        reverseOrderPos = 0;
        for(size_t nextAloneCheck = 0; nextAloneCheck < arrayLength;) {
            while (nextAloneCheck < arrayLength) {
                if (t2count[nextAloneCheck] == 1) {
                    alone[alonePos++] = nextAloneCheck;
                    // break;
                }
                nextAloneCheck++;
            }
            while (alonePos > 0) {
                int i = alone[--alonePos];
                if (t2count[i] == 0) {
                    continue;
                }
                long hash = t2[i];
                uint8_t found = -1;
                for (int hi = 0; hi < 3; hi++) {
                    int h = getHashFromHash(hash, hi, blockLength);
                    int newCount = --t2count[h];
                    if (newCount == 0) {
                        found = (uint8_t) hi;
                    } else {
                        if (newCount == 1) {
                            alone[alonePos++] = h;
                        }
                        t2[h] ^= hash;
                    }
                }
                reverseOrder[reverseOrderPos] = hash;
                reverseH[reverseOrderPos] = found;
                reverseOrderPos++;
            }
        }
        delete [] alone;
        if (reverseOrderPos == size) {
            break;
        }

std::cout << "WARNING: hashIndex " << hashIndex << "\n";

        hashIndex++;
    }

    this->hashIndex = hashIndex;

    for (int i = reverseOrderPos - 1; i >= 0; i--) {
        // the key we insert next
        uint64_t hash = reverseOrder[i];
        int found = reverseH[i];
        // which entry in the table we can change
        int change = -1;
        // we set table[change] to the fingerprint of the key,
        // unless the other two entries are already occupied
        uint8_t xor2 = fingerprint(hash);
        for (int hi = 0; hi < 3; hi++) {
            size_t h = getHashFromHash(hash, hi, blockLength);
            if (found == hi) {
                change = h;
            } else {
                // this is different from BDZ: using xor to calculate the
                // fingerprint
                xor2 ^= fingerprints[h];
            }
        }
        fingerprints[change] = (uint8_t) xor2;
    }

    delete [] t2;
    delete [] t2count;
    delete [] reverseOrder;
    delete [] reverseH;

    return Ok;
}

template <typename ItemType, size_t bits_per_item,
          typename HashFamily>
Status XorFilter<ItemType, bits_per_item, HashFamily>::Contain(
    const ItemType &key) const {
    uint64_t hash = hash64(key + hashIndex);
    uint8_t f = fingerprint(hash);
    uint32_t r0 = (uint32_t) hash;
    uint32_t r1 = (uint32_t) (hash >> 16);
    uint32_t r2 = (uint32_t) (hash >> 32);
    uint32_t h0 = reduce(r0, blockLength);
    uint32_t h1 = reduce(r1, blockLength) + blockLength;
    uint32_t h2 = reduce(r2, blockLength) + 2 * blockLength;
    f ^= fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2];
    return (f & 0xff) == 0 ? Ok : NotFound;
}

template <typename ItemType, size_t bits_per_item,
          typename HashFamily>
std::string XorFilter<ItemType, bits_per_item, HashFamily>::Info() const {
  std::stringstream ss;
  ss << "XorFilter Status:\n"
     << "\t\tKeys stored: " << Size() << "\n";
  if (Size() > 0) {
    ss << "\t\tbit/key:   " << BitsPerItem() << "\n";
  } else {
    ss << "\t\tbit/key:   N/A\n";
  }
  return ss.str();
}
}  // namespace xorfilter
#endif  // XOR_FILTER_XOR_FILTER_H_
