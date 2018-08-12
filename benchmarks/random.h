// Generating random data

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>


::std::vector<::std::uint64_t> GenerateRandom64(::std::size_t count) {
  ::std::vector<::std::uint64_t> result(count);
  ::std::random_device random;
  // To generate random keys to lookup, this uses ::std::random_device which is slower but
  // stronger than some other pseudo-random alternatives. The reason is that some of these
  // alternatives (like libstdc++'s ::std::default_random, which is a linear congruential
  // generator) behave non-randomly under some hash families like Dietzfelbinger's
  // multiply-shift.
  auto genrand = [&random]() {
    return random() + (static_cast<::std::uint64_t>(random()) << 32);
  };
  ::std::generate(result.begin(), result.end(), ::std::ref(genrand));
  return result;
}

::std::vector<::std::uint64_t> GenerateRandom64Fast(::std::size_t count, uint64_t start) {
  ::std::vector<::std::uint64_t> result(count);
  uint64_t index = start;
  auto genrand = [&index]() {
    // mix64
    uint64_t x = index++;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9L;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebL;
    x = x ^ (x >> 31);
    return x;
  };
  ::std::generate(result.begin(), result.end(), ::std::ref(genrand));
  return result;
}

// Using two pointer ranges for sequences x and y, create a vector clone of x but for
// y_probability y's mixed in.
template <typename T>
::std::vector<T> MixIn(const T* x_begin, const T* x_end, const T* y_begin, const T* y_end,
    double y_probability) {
  const size_t x_size = x_end - x_begin, y_size = y_end - y_begin;
  if (y_size > (1ull << 32)) throw ::std::length_error("y is too long");
  ::std::vector<T> result(x_begin, x_end);
  ::std::random_device random;
  auto genrand = [&random, y_size]() {
    return (static_cast<size_t>(random()) * y_size) >> 32;
  };
  for (size_t i = 0; i < y_probability * x_size; ++i) {
    result[i] = *(y_begin + genrand());
  }
  ::std::shuffle(result.begin(), result.end(), random);
  return result;
}

template <typename T>
::std::vector<T> MixInFast(const T* x_begin, const T* x_end, const T* y_begin, const T* y_end,
    double y_probability, uint64_t start) {
  const size_t x_size = x_end - x_begin, y_size = y_end - y_begin;
  if (y_size > (1ull << 32)) throw ::std::length_error("y is too long");
  ::std::vector<T> result(x_begin, x_end);
  uint64_t index = start;
  auto genrand = [&index, y_size]() {
    // mix64
    uint64_t x = index++;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9L;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebL;
    x = x ^ (x >> 31);
    return static_cast<size_t>((((uint32_t) x) * (uint64_t) y_size) >> 32);
  };
  for (size_t i = 0; i < y_probability * x_size; ++i) {
    result[i] = *(y_begin + genrand());
  }
  ::std::shuffle(result.begin(), result.end(), std::default_random_engine(start));
  return result;
}
