#ifndef CGL_RANDOMUTIL_H
#define CGL_RANDOMUTIL_H

#include <random>
#include <chrono>
#include "device-utils.h"
#include <thrust/random.h>
// #define XORSHIFT_RAND

namespace CGL {
static std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31, 0x9908b0df,
                                    11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
                                    1812433253> minstd_engine;

static double rmax = 1.0 / (minstd_engine.max() - minstd_engine.min());

class RandomGenerator {
 public:

  CUDA_CALLABLE CUDA_FORCEINLINE
  RandomGenerator(uint64_t seed) : state(seed) {}

  CUDA_CALLABLE CUDA_FORCEINLINE
  uint64_t next() {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double get() {
    return static_cast<double>(next()) / static_cast<double>(UINT64_MAX);
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double2 get_uv() {
    return make_double2(get(), get());
  }

 private:
  uint64_t state;
};

/**
 * Returns a number distributed uniformly over [0, 1].
 */
inline double random_uniform() {
  return clamp(double(minstd_engine() - minstd_engine.min()) * rmax, 0.0000001, 0.99999999);
}

/**
 * Returns true with probability p and false with probability 1 - p.
 */
inline bool coin_flip(double p) {
  return random_uniform() < p;
}

} // namespace CGL

#endif  // CGL_RANDOMUTIL_H
