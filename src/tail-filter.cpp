#include "quotient-dysect.hpp"

#include <iostream>

using namespace std;

int main() {
  QuotientDysect mm(6, 6, 2, 5, 1);
  cout << mm.SpaceUsed() << endl
       << mm.Capacity() << endl
       << (1.0 * mm.SpaceUsed() * CHAR_BIT) / mm.Capacity() << endl
       << mm.FilledSlots() << endl;
}
