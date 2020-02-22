#include "quotient-dysect.hpp"
#include "tail-filter.hpp"

#include <iostream>
#include <vector>

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
  TailFilter tf(20, 1.0 / 512);

  pcg32_random_t rnd = {1, 1};

  const uint64_t absent_ndv = 1'000'000;
  unique_ptr<uint64_t[]> absent(new uint64_t[absent_ndv]);
  for (uint64_t i = 0; i < absent_ndv; ++i) {
    absent[i] = pcg64(&rnd);
  }

  const uint64_t ndv = 128'000'000;
  unique_ptr<uint64_t[]> hashes(new uint64_t[ndv]);

  for (uint64_t i = 0; i < ndv; ++i) {
    if ((i > 0) && (0 == (i & (i - 1)))) {
      cout << "n " << i << " or " << tf.QdNdv() << endl
           << "bits per item " << ((1.0 * tf.SpaceUsed() * CHAR_BIT) / i) << endl
           << "bits per slot " << ((1.0 * tf.SpaceUsed() * CHAR_BIT) / tf.QuotientCapacity())
           << endl
           << "slots per item " << ((1.0 * tf.QuotientCapacity()) / i) << endl;
      if (i >= 1024) {
        uint64_t present = 0;
        for(uint64_t j = 0; j < absent_ndv; ++j) {
          present += tf.Lookup(absent[j]);
        }
        cout << "fpp\t" << ((1.0 * present) / absent_ndv) << endl
             << "optimal static bits per item\t"
             << -log2((1.0 * present) / absent_ndv) << endl
             << "classical BF bits per item\t"
             << log2(exp(1)) * -log2((1.0 * present) / absent_ndv) << endl;
      }
    }
    hashes[i] = pcg64(&rnd);
    //const bool more_ndv =
    if (i == 259) {
      cout << "here\n";
    }
    tf.Insert(hashes[i]);

    //    if (not more_ndv) cout << i << endl;

    // for (uint64_t j = 0; j <= i; ++j) {
    //   assert(tf.Lookup(hashes[j]));
    // }

    if (i >= 949101) {
      assert(tf.Lookup(hashes[949101]));
    }
  }

  cout << "n " << ndv << " or " << tf.QdNdv() << endl
       << "bits per item " << ((1.0 * tf.SpaceUsed() * CHAR_BIT) / ndv) << endl
       << "bits per slot "
       << ((1.0 * tf.SpaceUsed() * CHAR_BIT) / tf.QuotientCapacity()) << endl
       << "slots per item " << ((1.0 * tf.QuotientCapacity()) / ndv) << endl;
  if (ndv >= 1024) {
    uint64_t present = 0;
    for (uint64_t j = 0; j < absent_ndv; ++j) {
      present += tf.Lookup(absent[j]);
    }
    cout << "fpp\t" << ((1.0 * present) / absent_ndv) << endl
         << "optimal static bits per item\t"
         << -log2((1.0 * present) / absent_ndv) << endl
             << "classical BF bits per item\t"
         << log2(exp(1)) * -log2((1.0 * present) / absent_ndv) << endl;
  }

  for (uint64_t i = 0; i < ndv; ++i) {
    assert(tf.Lookup(hashes[i]));
  }

  return 0;
  int keylength = 22;

  // auto f = new QuotientDysect::HashBijection[1];
  // f[0] = QuotientDysect::Feistelize(MultiplyHash, keylength);
  // QuotientDysect mm_old(keylength, 3, 2, 3, 1, 1,
  //                   unique_ptr<QuotientDysect::HashBijection[]>(f));

  QuotientDysect mm(keylength, 3 /*val*/, 2/*d*/, 3/*w*/, 2/*s*/, 1, MultiplyHash);
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

  vector<int> inserted;
  for (int i = 0; i < 40'000'000; ++i) {
    auto key = pcg64(&rnd) & ((1 << keylength) - 1);
    key = key & ((1 << mm.k_) - 1);
    inserted.push_back(key);
    // cout << "iteration " << i << endl;
    mm.Insert(key, 1);
    assert(not mm.Find(key).AtEnd());
    assert(mm.FindExact(key,1));
    if (0 == (i & (i-1))) {
      cout << i << endl
           << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl;
    }
  }
  for (unsigned i = 0; i < inserted.size(); ++i) {
    if (not mm.FindExact(inserted[i], 1)) {
      cout << i << endl;
      assert(false);
    }
  }
}
