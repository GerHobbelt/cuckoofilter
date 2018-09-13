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

inline uint64_t rotl64(uint64_t n, unsigned int c) {
    // assumes width is a power of 2
    const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
    // assert ( (c<=mask) &&"rotate by type width or more");
    c &= mask;
    return (n << c) | ( n >> ((-c) & mask));
}

__attribute__((always_inline))
inline uint32_t reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t) (((uint64_t) hash * n) >> 32);
}

size_t getHashFromHash(uint64_t hash, int index, int blockLength) {
    uint32_t r;
    switch(index) {
    case 0:
        r = (uint32_t) (hash);
        break;
    case 1:
        r = (uint32_t) rotl64(hash, 21);
        break;
    default:
        r = (uint32_t) rotl64(hash, 42);
        break;
    }
    r = reduce(r, blockLength);
    r = r + index * blockLength;
    return (size_t) r;
}

template <typename ItemType, typename FingerprintType,
          typename HashFamily = TwoIndependentMultiplyShift>
class XorFilter {

  size_t size;
  size_t arrayLength;
  size_t blockLength;
  FingerprintType *fingerprints;

  HashFamily* hasher;

  inline FingerprintType fingerprint(const uint64_t hash) const {
    return (FingerprintType) hash;
  }

 public:
  explicit XorFilter(const size_t size) {
    hasher = new HashFamily();
    this->size = size;
    this->arrayLength = 3 + 1.23 * size;
    this->blockLength = arrayLength / 3;
    fingerprints = new FingerprintType[arrayLength]();
    std::fill_n(fingerprints, arrayLength, 0);
  }

  ~XorFilter() {
    delete[] fingerprints;
    delete hasher;
  }

  Status AddAll(const vector<ItemType> data, const size_t start, const size_t end);

  // Report if the item is inserted, with false positive rate.
  Status Contain(const ItemType &item) const;

  /* methods for providing stats  */
  // summary infomation
  std::string Info() const;

  // number of current inserted items;
  size_t Size() const { return size; }

  // size of the filter in bytes.
  size_t SizeInBytes() const { return arrayLength * sizeof(FingerprintType); }
};


#define BETTERLOCALITY
#ifdef BETTERLOCALITY
struct t2val {
  uint64_t t2;
  uint64_t t2count;
};

typedef struct t2val t2val_t;

#endif

template <typename ItemType, typename FingerprintType,
          typename HashFamily>
Status XorFilter<ItemType, FingerprintType, HashFamily>::AddAll(
    const vector<ItemType> keys, const size_t start, const size_t end) {

    int m = arrayLength;
    uint64_t* reverseOrder = new uint64_t[size];
    uint8_t* reverseH = new uint8_t[size];
    size_t reverseOrderPos;
    int hashIndex = 0;
#ifdef BETTERLOCALITY
    t2val_t * t2vals = new t2val_t[m];
#else
    uint64_t* t2 = new uint64_t[m];
    uint8_t* t2count = new uint8_t[m];
#endif
    while (true) {
#ifndef BETTERLOCALITY
        memset(t2count, 0, sizeof(uint8_t[m]));
        memset(t2, 0, sizeof(uint64_t[m]));
#else
        memset(t2vals, 0, sizeof(t2val_t[m]));
#endif
        for(size_t i = start; i < end; i++) {
            uint64_t k = keys[i];
            uint64_t hash = (*hasher)(k);
            int h0 = reduce((int) (hash), blockLength);
            int h1 = reduce((int) rotl64(hash, 21), blockLength) + blockLength;
            int h2 = reduce((int) rotl64(hash, 42), blockLength) + 2 * blockLength;
#ifndef BETTERLOCALITY
            t2count[h0]++;
            t2[h0] ^= hash;
            t2count[h1]++;
            t2[h1] ^= hash;
            t2count[h2]++;
            t2[h2] ^= hash;
#else
            t2vals[h0].t2count++;
            t2vals[h0].t2 ^= hash;
            t2vals[h1].t2count++;
            t2vals[h1].t2 ^= hash;
            t2vals[h2].t2count++;
            t2vals[h2].t2 ^= hash;
#endif
        }
        reverseOrderPos = 0;
        int* alone = new int[arrayLength];
        int alonePos = 0;
        reverseOrderPos = 0;
        for(size_t nextAloneCheck = 0; nextAloneCheck < arrayLength;) {
            while (nextAloneCheck < arrayLength) {
#ifndef BETTERLOCALITY
                if (t2count[nextAloneCheck] == 1) {
#else
                if (t2vals[nextAloneCheck].t2count == 1) {
#endif
                    alone[alonePos++] = nextAloneCheck;
                    // break;
                }
                nextAloneCheck++;
            }
            while (alonePos > 0) {
                int i = alone[--alonePos];
#ifdef BETTERLOCALITY
                if (t2vals[i].t2count == 0) {
                    continue;
                }
                long hash = t2vals[i].t2;
#else
                if (t2count[i] == 0) {
                    continue;
                }
                long hash = t2[i];
#endif
                uint8_t found = -1;
                for (int hi = 0; hi < 3; hi++) {
                    int h = getHashFromHash(hash, hi, blockLength);
#ifdef BETTERLOCALITY
                    int newCount =  -- t2vals[h].t2count;
#else
                    int newCount = --t2count[h];
#endif
                    if (newCount == 0) {
                        found = (uint8_t) hi;
                    } else {
                        if (newCount == 1) {
                            alone[alonePos++] = h;
                        }
#ifdef BETTERLOCALITY
                        t2vals[h].t2 ^= hash;
#else
                        t2[h] ^= hash;
#endif
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
        if (hashIndex >= 0) {
           // size_t outputlimit = 5; // we don't want to spam
            std::cout << (end - start) << " keys; arrayLength " << arrayLength
                << " blockLength " << blockLength
                << " reverseOrderPos " << reverseOrderPos << "\n";
           // int pos = 0;
           /* for (size_t i = 0; pos < 1000 && i < arrayLength; i++) {
                if (t2count[i] > 1) {
                    if(outputlimit > 0) {
                       std::cout << "  count[" << i << "] = " << (int) t2count[i] << "\n";
                       outputlimit --;
                     }
                }
            }
           for(size_t i = start; i < end; i++) {
                uint64_t k = keys[i];
                uint64_t hash = (*hasher)(k);
                int h0 = reduce((int) (hash), blockLength);
                int h1 = reduce((int) rotl64(hash, 21), blockLength) + blockLength;
                int h2 = reduce((int) rotl64(hash, 42), blockLength) + 2 * blockLength;
                if (t2count[h0] > 1 || t2count[h1] > 1 || t2count[h2] > 1) {
                    if(outputlimit > 0) {
                      std::cout << "  key " << k << " hash=" << hash << " h0=" << h0 << " h1=" << h1 << " h2=" << h2 << "\n";
                      outputlimit --;
                    }
                }
            }*/

            // for(size_t i = start; i < end; i++) {
            //     uint64_t k = keys[i];
            //     std::cout << k << "\n";
            // }
            // std::cout << "end\n";
        }

        hashIndex++;

        // use a new random numbers
        delete hasher;
        hasher = new HashFamily();

    }

    for (int i = reverseOrderPos - 1; i >= 0; i--) {
        // the hash of the key we insert next
        uint64_t hash = reverseOrder[i];
        int found = reverseH[i];
        // which entry in the table we can change
        int change = -1;
        // we set table[change] to the fingerprint of the key,
        // unless the other two entries are already occupied
        FingerprintType xor2 = fingerprint(hash);
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
        fingerprints[change] = xor2;
    }
#ifdef BETTERLOCALITY
    delete [] t2vals;
#else
    delete [] t2;
    delete [] t2count;
#endif
    delete [] reverseOrder;
    delete [] reverseH;
    return Ok;
}

template <typename ItemType, typename FingerprintType,
          typename HashFamily>
Status XorFilter<ItemType, FingerprintType, HashFamily>::Contain(
    const ItemType &key) const {
    uint64_t hash = (*hasher)(key);
    FingerprintType f = fingerprint(hash);
    uint32_t r0 = (uint32_t) hash;
    uint32_t r1 = (uint32_t) rotl64(hash, 21);
    uint32_t r2 = (uint32_t) rotl64(hash, 42);
    uint32_t h0 = reduce(r0, blockLength);
    uint32_t h1 = reduce(r1, blockLength) + blockLength;
    uint32_t h2 = reduce(r2, blockLength) + 2 * blockLength;
    f ^= fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2];
    return f == 0 ? Ok : NotFound;
}

template <typename ItemType, typename FingerprintType,
          typename HashFamily>
std::string XorFilter<ItemType, FingerprintType, HashFamily>::Info() const {
  std::stringstream ss;
  ss << "XorFilter Status:\n"
     << "\t\tKeys stored: " << Size() << "\n";
  return ss.str();
}
}  // namespace xorfilter
#endif  // XOR_FILTER_XOR_FILTER_H_
