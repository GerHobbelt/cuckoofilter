// This benchmark reports on the bulk insert and bulk query rates. It is invoked as:
//
//     ./bulk-insert-and-query.exe 158000
//
// That invocation will test each probabilistic membership container type with 158000
// randomly generated items. It tests bulk Add() from empty to full and Contain() on
// filters with varying rates of expected success. For instance, at 75%, three out of
// every four values passed to Contain() were earlier Add()ed.
//
// Example usage:
//
// for alg in `seq 0 1 14`; do for num in `seq 2 2 20`; do ./bulk-insert-and-query.exe ${num}000000 ${alg}; done; done > results.txt

#include <climits>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <vector>

#include "cuckoofilter.h"
#include "cuckoofilter_stable.h"
#include "xorfilter.h"
#include "xorfilter_plus.h"
#include "bloom.h"
#include "gcs.h"
#include "gqf_cpp.h"
#include "random.h"
#include "simd-block.h"
#include "timing.h"

using namespace std;

using namespace cuckoofilter;
using namespace xorfilter;
using namespace xorfilter_plus;
using namespace bloomfilter;
using namespace gcsfilter;
using namespace gqfilter;

// The number of items sampled when determining the lookup performance
const size_t SAMPLE_SIZE = 1000 * 1000;

// The statistics gathered for each table type:
struct Statistics {
  size_t add_count;
  double adds_per_nano;
  map<int, double> finds_per_nano; // The key is the percent of queries that were expected
                                   // to be positive
  double false_positive_probabilty;
  double bits_per_item;
};

// Output for the first row of the table of results. type_width is the maximum number of
// characters of the description of any table type, and find_percent_count is the number
// of different lookup statistics gathered for each table. This function assumes the
// lookup expected positive probabiilties are evenly distributed, with the first being 0%
// and the last 100%.
string StatisticsTableHeader(int type_width, int find_percent_count) {
  ostringstream os;

  os << string(type_width, ' ');
  os << setw(12) << right << "million";
  for (int i = 0; i < find_percent_count; ++i) {
    os << setw(8) << "find";
  }
  os << setw(8) << "" << setw(11) << "" << setw(11)
     << "optimal" << setw(8) << "wasted" << setw(8) << "million" << endl;

  os << string(type_width, ' ');
  os << setw(12) << right << "adds/sec";
  for (int i = 0; i < find_percent_count; ++i) {
    os << setw(7)
       << static_cast<int>(100 * i / static_cast<double>(find_percent_count - 1)) << '%';
  }
  os << setw(9) << "ε" << setw(11) << "bits/item" << setw(11)
     << "bits/item" << setw(8) << "space" << setw(8) << "keys";
  return os.str();
}

// Overloading the usual operator<< as used in "std::cout << foo", but for Statistics
template <class CharT, class Traits>
basic_ostream<CharT, Traits>& operator<<(
    basic_ostream<CharT, Traits>& os, const Statistics& stats) {
  constexpr double NANOS_PER_MILLION = 1000;
  os << fixed << setprecision(2) << setw(12) << right
     << stats.adds_per_nano * NANOS_PER_MILLION;
  for (const auto& fps : stats.finds_per_nano) {
    os << setw(8) << fps.second * NANOS_PER_MILLION;
  }
  const auto minbits = log2(1 / stats.false_positive_probabilty);
  os << setw(7) << setprecision(3) << stats.false_positive_probabilty * 100 << '%'
     << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << minbits
     << setw(7) << setprecision(1) << 100 * (stats.bits_per_item / minbits - 1) << '%'
     << setw(8) << setprecision(1) << (stats.add_count / 1000000.);

  return os;
}

template<typename Table>
struct FilterAPI {};

template <typename ItemType, size_t bits_per_item, template <size_t> class TableType>
struct FilterAPI<CuckooFilter<ItemType, bits_per_item, TableType>> {
  using Table = CuckooFilter<ItemType, bits_per_item, TableType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table * table) {
    if (0 != table->Add(key)) {
      throw logic_error("The filter is too small to hold all of the elements");
    }
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item, template <size_t> class TableType>
struct FilterAPI<CuckooFilterStable<ItemType, bits_per_item, TableType>> {
  using Table = CuckooFilterStable<ItemType, bits_per_item, TableType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table * table) {
    if (0 != table->Add(key)) {
      throw logic_error("The filter is too small to hold all of the elements");
    }
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <>
struct FilterAPI<SimdBlockFilter<>> {
  using Table = SimdBlockFilter<>;
  static Table ConstructFromAddCount(size_t add_count) {
    Table ans(ceil(log2(add_count * 8.0 / CHAR_BIT)));
    return ans;
  }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<uint64_t> keys, const size_t start, const size_t end, Table* table) {
  }
  static bool Contain(uint64_t key, const Table * table) {
    return table->Find(key);
  }
};

template <typename ItemType, typename FingerprintType>
struct FilterAPI<XorFilter<ItemType, FingerprintType>> {
  using Table = XorFilter<ItemType, FingerprintType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, typename FingerprintType>
struct FilterAPI<XorFilterPlus<ItemType, FingerprintType>> {
  using Table = XorFilterPlus<ItemType, FingerprintType>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item>
struct FilterAPI<GcsFilter<ItemType, bits_per_item>> {
  using Table = GcsFilter<ItemType, bits_per_item>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
    table->AddAll(keys, start, end);
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item>
struct FilterAPI<GQFilter<ItemType, bits_per_item>> {
  using Table = GQFilter<ItemType, bits_per_item>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename ItemType, size_t bits_per_item>
struct FilterAPI<BloomFilter<ItemType, bits_per_item>> {
  using Table = BloomFilter<ItemType, bits_per_item>;
  static Table ConstructFromAddCount(size_t add_count) { return Table(add_count); }
  static void Add(uint64_t key, Table* table) {
    table->Add(key);
  }
  static void AddAll(const vector<ItemType> keys, const size_t start, const size_t end, Table* table) {
  }
  static bool Contain(uint64_t key, const Table * table) {
    return (0 == table->Contain(key));
  }
};

template <typename Table>
Statistics FilterBenchmark(
    size_t add_count, const vector<uint64_t>& to_add, const vector<uint64_t>& to_lookup) {
  if (add_count > to_add.size()) {
    throw out_of_range("to_add must contain at least add_count values");
  }

  if (SAMPLE_SIZE > to_lookup.size()) {
    throw out_of_range("to_lookup must contain at least SAMPLE_SIZE values");
  }

  Table filter = FilterAPI<Table>::ConstructFromAddCount(add_count);
  Statistics result;

  // Add values until failure or until we run out of values to add:
  auto start_time = NowNanos();

  for (size_t added = 0; added < add_count; ++added) {
    FilterAPI<Table>::Add(to_add[added], &filter);
  }
  // for the XorFilter
  FilterAPI<Table>::AddAll(to_add, 0, add_count, &filter);

  result.add_count = add_count;
  result.adds_per_nano = add_count / static_cast<double>(NowNanos() - start_time);
  result.bits_per_item = static_cast<double>(CHAR_BIT * filter.SizeInBytes()) / add_count;

  size_t found_count = 0;
  for (const double found_probability : {0.0, 0.25, 0.50, 0.75, 1.00}) {
    const auto to_lookup_mixed = MixIn(&to_lookup[0], &to_lookup[SAMPLE_SIZE], &to_add[0],
        &to_add[add_count], found_probability);
    const auto start_time = NowNanos();
    for (const auto v : to_lookup_mixed) {
      found_count += FilterAPI<Table>::Contain(v, &filter);
    }
    const auto lookup_time = NowNanos() - start_time;
    result.finds_per_nano[100 * found_probability] =
        SAMPLE_SIZE / static_cast<double>(lookup_time);
    if (0.0 == found_probability) {
      result.false_positive_probabilty =
          found_count / static_cast<double>(to_lookup_mixed.size());
    }
  }
  return result;
}

int main(int argc, char * argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " $NUMBER" << endl;
    return 1;
  }
  stringstream input_string(argv[1]);
  size_t add_count;
  input_string >> add_count;
  if (input_string.fail()) {
    cerr << "Invalid number: " << argv[1];
    return 2;
  }
  int algorithmId = -1;
  if (argc > 2) {
      stringstream input_string_2(argv[2]);
      input_string_2 >> algorithmId;
      if (input_string_2.fail()) {
          cerr << "Invalid number: " << argv[2];
          return 2;
      }
  }

  const vector<uint64_t> to_add = GenerateRandom64(add_count);
  const vector<uint64_t> to_lookup = GenerateRandom64(SAMPLE_SIZE);

  constexpr int NAME_WIDTH = 16;

  cout << StatisticsTableHeader(NAME_WIDTH, 5) << endl;

  if (algorithmId == 0 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          XorFilter<uint64_t, uint8_t>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Xor8" << cf << endl;
  }

  if (algorithmId == 1 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          XorFilter<uint64_t, uint16_t>>(
        add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Xor16" << cf << endl;
  }

  if (algorithmId == 2 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          XorFilterPlus<uint64_t, uint8_t>>(
        add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Xor+8" << cf << endl;
  }

  if (algorithmId == 3 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          XorFilterPlus<uint64_t, uint16_t>>(
        add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Xor+16" << cf << endl;
  }

  if (algorithmId == 4 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          BloomFilter<uint64_t, 10 /* bits per item */>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Bloom" << cf << endl;
  }

  if (algorithmId == 5 || algorithmId < 0) {
      auto cf = FilterBenchmark<SimdBlockFilter<>>(add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "SimdBlock8" << cf << endl;
  }

  if (algorithmId == 6 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          GcsFilter<uint64_t, 8>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "GCS" << cf << endl;
  }

  if (algorithmId == 7 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilterStable<uint64_t, 12 /* bits per item */, SingleTable /* not semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "CuckooStable12" << cf << endl;
  }

  if (algorithmId == 8 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          GQFilter<uint64_t, 8>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "CQF" << cf << endl;
  }

  if (algorithmId == 9 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 8 /* bits per item */, SingleTable /* not semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Cuckoo8" << cf << endl;
  }

  if (algorithmId == 10 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 12 /* bits per item */, SingleTable /* not semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Cuckoo12" << cf << endl;
  }

  if (algorithmId == 11 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 16 /* bits per item */, SingleTable /* not semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "Cuckoo16" << cf << endl;
  }

  if (algorithmId == 12 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 9 /* bits per item */, PackedTable /* semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "SemiSort9" << cf << endl;
  }

  if (algorithmId == 13 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 13 /* bits per item */, PackedTable /* semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "SemiSort13" << cf << endl;
  }

  if (algorithmId == 14 || algorithmId < 0) {
      auto cf = FilterBenchmark<
          CuckooFilter<uint64_t, 17 /* bits per item */, PackedTable /* semi-sorted*/>>(
          add_count, to_add, to_lookup);
      cout << setw(NAME_WIDTH) << "SemiSort17" << cf << endl;
  }

}
