#include "quotient-dysect.hpp"

#include <iostream>

using namespace std;

uint64_t MultiplyHash(uint64_t x) {
  unsigned __int128 y = 0x6d064b7e7e084d32;
  y = y << 64;
  y = y| 0xaad87d809dd4d431;
  return (static_cast<unsigned __int128>(x) * y) >> 64;
}

int main() {
  auto f = new QuotientDysect::HashBijection[1];
  int keylength = 6;
  f[0] = QuotientDysect::Feistelize(MultiplyHash, keylength);
  QuotientDysect mm(keylength, 6, 2, 3, 1, 10,
                    unique_ptr<QuotientDysect::HashBijection[]>(f));
  const auto original_capacity = mm.Capacity();
  cout << mm.SpaceUsed() << endl
       << mm.Capacity() << endl
       << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl
       << mm.FilledSlots() << endl
       << boolalpha << (mm.Begin() == mm.End()) << endl;
  while (mm.Capacity() <= (1 << 2) * original_capacity) {
    auto ok = mm.Upsize();
    assert(ok);
    cout << mm.Capacity() << endl
         << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl;
  }
  cout << "------------------------" << endl;
  mm.Insert(1, 2);
  mm.Insert(1, 3);
  mm.Insert(1, 4);
  for (auto i = mm.Begin(); i != mm.End(); ++i) {
    auto f = i.Get();
    cout << f.key << ' ' << f.value << endl;
  }
}
