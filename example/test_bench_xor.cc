#include "cuckoofilter.h"

#include <assert.h>
#include <math.h>

#include <iostream>
#include <vector>

#include "../benchmarks/random.h"
#include "xorfilter.h"

using cuckoofilter::CuckooFilter;
using xorfilter::XorFilter;

using namespace std;

int main(int argc, char **argv) {
  size_t total_items = 10000000;

  XorFilter<uint64_t, 8> filter(total_items);


  std::cout << "xor filter created\n";
  
  const vector<uint64_t> data = GenerateRandom64(total_items * 2);

  std::cout << "random ok\n";

  filter.AddAll(data, 0, total_items);
  std::cout << "inserted " << total_items << "\n";

  // Check if previously inserted items are in the filter, expected
  // true for all items
  for (size_t i = 0; i < total_items; i++) {
    assert(filter.Contain(data[i]) == xorfilter::Ok);
  }

  // Check non-existing items, a few false positives expected
  size_t total_queries = 0;
  size_t false_queries = 0;
  for (size_t i = total_items; i < 2 * total_items; i++) {
    if (filter.Contain(data[i]) == xorfilter::Ok) {
      false_queries++;
    }
    total_queries++;
  }

  // Output the measured false positive rate
  std::cout << "false positive rate is "
            << 100.0 * false_queries / total_queries << "%\n";

  return 0;
}
