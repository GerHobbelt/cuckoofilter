#include "quotient-dysect.hpp"
#include "tail-filter.hpp"

#include <iostream>

using namespace std;

uint64_t MultiplyHash(uint64_t x) {
  unsigned __int128 y = 0x6d064b7e7e084d32;
  y = y << 64;
  y = y| 0xaad87d809dd4d431;
  return (static_cast<unsigned __int128>(x) * y) >> 64;
}


// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint64_t pcg64(pcg32_random_t* rng) {
  uint64_t result = pcg32_random_r(rng);
  result = (result << 32) | pcg32_random_r(rng);
  return result;
}

int main() {
  TailFilter tf(10, 1.0 / 256);

  pcg32_random_t rnd = {1, 1};
  const uint64_t ndv = 1'000'000;
  unique_ptr<uint64_t[]> hashes(new uint64_t[ndv]);
  for (uint64_t i = 0; i < ndv; ++i) {
    hashes[i] = pcg64(&rnd);
    if (not tf.Insert(hashes[i])) cout << i << endl;
    for (uint64_t j = 0; j <= i; ++j) {
      assert(tf.Lookup(hashes[j]));
    }
  }

  int keylength = 22;

  // auto f = new QuotientDysect::HashBijection[1];
  // f[0] = QuotientDysect::Feistelize(MultiplyHash, keylength);
  // QuotientDysect mm_old(keylength, 3, 2, 3, 1, 1,
  //                   unique_ptr<QuotientDysect::HashBijection[]>(f));

  QuotientDysect mm(keylength, 3, 2, 3, 1, 1, MultiplyHash);
  cout << mm.SpaceUsed() << endl
       << mm.Capacity() << endl
       << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl
       << mm.FilledSlots() << endl
       << boolalpha << (mm.Begin() == mm.End()) << endl;

  //   const auto original_capacity = mm.Capacity();
  // while (mm.Capacity() <= (1 << 2) * original_capacity) {
  //   auto ok = mm.Upsize();
  //   assert(ok);
  //   cout << mm.Capacity() << endl
  //        << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl;
  // }

  // cout << "------------------------" << endl;
  mm.Insert(1, 2);
  mm.Insert(1, 3);
  mm.Insert(1, 4);
  for (auto i = mm.Begin(); i != mm.End(); ++i) {
    auto f = i.GetOriginal();
    cout << f.key
         //<< ' '
         << f.value << endl;
  }
  for (int i = 0; i < 10'000'000; ++i) {
    auto key = rand() & ((1 << keylength) - 1);
    mm.Insert(key, 1);
    assert(mm.Find(key) != mm.End());
    if (0 == (i & (i-1))) {
      cout << i << endl
           << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl;
    }
  }
}
