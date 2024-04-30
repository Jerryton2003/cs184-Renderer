#ifndef CGL_SAMPLER_H
#define CGL_SAMPLER_H

#include "CGL/vector2D.h"
#include "CGL/vector3D.h"
#include "CGL/misc.h"
#include "util/random_util.h"
#include <cuda_runtime.h>
#include <util/gpu-arrays.h>
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#define M_1_PI 0.31830988618379067154 /* 1/pi */

namespace CGL {
CUDA_DEVICE CUDA_INLINE
double3 cosHemisphereSample(const double2 &uv) {
  // cosine weighted hemisphere sample
  double r = sqrt(uv.x);
  double theta = 2 * PI * uv.y;
  return double3(r * cos(theta), r * sin(theta), sqrt(1 - uv.x));
}

CUDA_DEVICE CUDA_INLINE
double3 uniformHemisphereSample(const double2& uv) {
  double z = 1 - 2 * uv.x;
  double r = sqrt(fmax(0.0, 1 - z * z));
  double phi = 2 * PI * uv.y;
  return make_double3(r * cos(phi), r * sin(phi), z);
}

#define M_1_4PI 0.07957747154594767
CUDA_DEVICE CUDA_FORCEINLINE
double uniformHemispherePdf() {
  return M_1_4PI;
}

CUDA_DEVICE CUDA_INLINE
double cosHemispherePdf(const double3 &d) {
  return d.z * M_1_PI;
}

/**
 * TODO (extra credit) :
 * Jittered sampler implementations
 */
struct DiscreteDistribution {
  DiscreteDistribution() = default;

  explicit DiscreteDistribution(int n) {
    probabilities = std::make_unique<DeviceArray<double>>(n);
    alias = std::make_unique<DeviceArray<int>>(n);
    accept = std::make_unique<DeviceArray<double>>(n);
  }

  explicit DiscreteDistribution(const std::vector<double> &weights) {
    buildFromWeights(weights);
  }

  void buildFromWeights(const std::vector<double> &weights) const {
    std::vector<double> host_probabilities = weights;
    // normalize host_probabilities
    double sum = 0;
    for (double p : host_probabilities)
      sum += p;
    for (double &p : host_probabilities)
      p /= sum;
    auto n = host_probabilities.size();
    std::vector<int> host_alias(n);
    std::vector<double> scaledProbabilities = host_probabilities;
    for (int i = 0; i < n; i++)
      scaledProbabilities[i] *= n;
    std::vector<int> small, large;
    for (int i = 0; i < n; ++i) {
      if (scaledProbabilities[i] < 1.0)
        small.push_back(i);
      else
        large.push_back(i);
    }

    while (!small.empty() && !large.empty()) {
      int less = small.back();
      small.pop_back();
      int more = large.back();
      large.pop_back();

      host_alias[less] = more;
      scaledProbabilities[more] = scaledProbabilities[more] + scaledProbabilities[less] - 1.0;

      if (scaledProbabilities[more] < 1.0)
        small.push_back(more);
      else
        large.push_back(more);
    }

    while (!large.empty()) {
      int curr = large.back();
      large.pop_back();
      scaledProbabilities[curr] = 1.0;
    }

    while (!small.empty()) {
      int curr = small.back();
      small.pop_back();
      scaledProbabilities[curr] = 1.0;
    }

    accept->copyFrom(scaledProbabilities);
    probabilities->copyFrom(host_probabilities);
    // print scaledProbabilities and host_probabilities
    for (int i = 0; i < n; i++) {
      printf("scaledProbabilities[%d] = %f, host_probabilities[%d] = %f\n", i, scaledProbabilities[i], i, host_probabilities[i]);
    }
    alias->copyFrom(host_alias);
  }

  std::unique_ptr<DeviceArray<double>> probabilities;
  std::unique_ptr<DeviceArray<int>> alias;
  std::unique_ptr<DeviceArray<double>> accept;
};
} // namespace CGL

#endif //CGL_SAMPLER_H
