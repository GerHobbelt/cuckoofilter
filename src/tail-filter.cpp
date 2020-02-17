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
  auto f = new function<uint64_t(uint64_t)>[1];
  f[0] = MultiplyHash;
  QuotientDysect mm(6, 6, 2, 5, 1,
                    unique_ptr<function<uint64_t(uint64_t)>[]>(f));
  const auto original_capacity = mm.Capacity();
  cout << mm.SpaceUsed() << endl
       << mm.Capacity() << endl
       << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl
       << mm.FilledSlots() << endl
       << boolalpha << (mm.Begin() == mm.End()) << endl;
  while (mm.Capacity() <= 8 * original_capacity) {
    auto ok = mm.Upsize();
    assert(ok);
    cout << mm.Capacity() << endl;
  }
}
