//
// Created by creeper on 4/28/24.
//

#ifndef ASSIGNMENT3_SRC_SCENE_MICROFACET_CUH_
#define ASSIGNMENT3_SRC_SCENE_MICROFACET_CUH_
#include "../util/gpu-arrays.h"
#include "../util/device-utils.h"
#include "../util/device-vec-ops.h"
#include "../../CGL/include/CGL/misc.h"
#include <thrust/swap.h>
namespace CGL {
CUDA_DEVICE CUDA_FORCEINLINE double computeFresnel(const double3& incident, const double3& normal, double ior) {
  double cos_theta_i = dot(normal, incident);
  double eta_i = 1.0;
  double eta_t = ior;
  if (cos_theta_i > 0.0)
    thrust::swap(eta_i, eta_t);
  double sin_theta_i = sqrt(fmax(0.0, 1.0 - cos_theta_i * cos_theta_i));
  double sin_theta_t = eta_i / eta_t * sin_theta_i;
  if (sin_theta_t >= 1.0) {
    return 1.0;
  } else {
    double cos_theta_t = std::sqrt(fmax(0.0, 1.0 - sin_theta_t * sin_theta_t));
    double r_parallel = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) /
                       ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    double r_perpendicular = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) /
                            ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
  }
}

CUDA_DEVICE CUDA_FORCEINLINE double computeCookTorranceBRDF(const double3& normal, const double3& view_dir,
                                            const double3& light_dir, double roughness, double ior) {
  double cos_theta_h = fmax(0.0, dot(normal, normalize(view_dir + light_dir)));
  double denom = (4.0 * fmax(0.001, (dot(normal, view_dir) * dot(normal, light_dir))));
  double roughness_sqr = roughness * roughness;
  double roughness_sqr_cos_theta_h = roughness_sqr * cos_theta_h * cos_theta_h;
  double d = (roughness_sqr_cos_theta_h + (1.0 - cos_theta_h * cos_theta_h)) /
            (M_PI * denom * denom);
  double f = computeFresnel(view_dir, normal, ior);
  return d * f / (4.0 * dot(normal, view_dir) * dot(normal, light_dir));
}

CUDA_DEVICE CUDA_FORCEINLINE double computeTrowbridgeHeitz(const double3& normal, const double3& half_vector,
                                            double roughness) {
  double cos_theta_h = fmax(0.0, dot(normal, half_vector));
  double tan_theta_h_sqr = fmax(0.0, 1.0 - cos_theta_h * cos_theta_h) / (cos_theta_h * cos_theta_h);
  double alpha_sqr = roughness * roughness;
  double alpha_sqr_cos_theta_h_sqr = alpha_sqr * cos_theta_h * cos_theta_h;
  double denom = M_PI * alpha_sqr_cos_theta_h_sqr * alpha_sqr_cos_theta_h_sqr;
  return alpha_sqr_cos_theta_h_sqr / denom;
}

CUDA_DEVICE CUDA_FORCEINLINE double computeTrowbridgeHeitzG1(const double3& v, const double3& m, const double3& n,
                                             double alpha) {
  double cos_theta_v = fmax(0.0, dot(v, n));
  if (cos_theta_v <= 0.0) return 0.0;
  double tan_theta_v = sqrt(fmax(0.0, 1.0 - cos_theta_v * cos_theta_v)) / cos_theta_v;
  if (tan_theta_v == 0.0) return 1.0;
  double a = 1.0 / (alpha * tan_theta_v);
  if (a >= 1.6f) return 1.0;
  double a_sqr = a * a;
  return (3.535f * a + 2.181f * a_sqr) / (1.0 + 2.276f * a + 2.577f * a_sqr);
}

CUDA_DEVICE CUDA_FORCEINLINE double computeTrowbridgeHeitzG2(const double3& wi, const double3& wo, const double3& m,
                                             const double3& n, double alpha) {
  double g1_wi = computeTrowbridgeHeitzG1(wi, m, n, alpha);
  double g1_wo = computeTrowbridgeHeitzG1(wo, m, n, alpha);
  return g1_wi * g1_wo;
}

CUDA_DEVICE CUDA_FORCEINLINE double3 sampleMicrofacetDistribution(const double3& normal, double roughness,
                                                const double2& rng_uv) {
  double phi = 2.0 * M_PI * rng_uv.x;
  double cos_theta = std::sqrt((1.0 - rng_uv.y) / (1.0 + (roughness * roughness - 1.0) * rng_uv.y));
  double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
  return double3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

CUDA_DEVICE CUDA_FORCEINLINE double computeMicrofacetPdf(const double3& normal, const double3& wi,
                                         double roughness) {
  double cos_theta = fmax(0.0, dot(normal, wi));
  double tan_theta_sqr = fmax(0.0, 1.0 - cos_theta * cos_theta) / (cos_theta * cos_theta);
  double alpha_sqr = roughness * roughness;
  double alpha_sqr_cos_theta_sqr = alpha_sqr * cos_theta * cos_theta;
  return (2.0 * PI * alpha_sqr_cos_theta_sqr * cos_theta) /
         (4.0 * dot(normal, wi) * sqrt(1.0 + tan_theta_sqr * alpha_sqr));
}
}
#endif //ASSIGNMENT3_SRC_SCENE_MICROFACET_CUH_
