#ifndef CGL_STATICSCENE_BSDF_H
#define CGL_STATICSCENE_BSDF_H

#include "CGL/CGL.h"
#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"

#include "pathtracer/sampler.h"
#include "util/image.h"

#include <algorithm>
#include <cmath>
#include "util/device-utils.h"

namespace CGL {

// Helper math functions. Assume all vectors are in unit hemisphere //
CUDA_CALLABLE CUDA_FORCEINLINE
double clamp (double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

CUDA_CALLABLE CUDA_FORCEINLINE
double cos_theta(const double3& w) {
  return w.z;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double sin_theta2(const double3& w) {
  return fmax(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}
/**
 * Interface for BSDFs.
 * BSDFs (Bidirectional Scattering Distribution Functions)
 * describe the ratio of incoming light scattered from
 * incident direction to outgoing direction.
 * Scene objects are initialized with a BSDF subclass, used
 * to represent the object's material and associated properties.
 */
class BSDF {
 public:

  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
  virtual double3 f(const double3 &wo, const double3 &wi) = 0;

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, samplea incident light
   * direction and store it in wi. Store the pdf of the sampled direction in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the sampled incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  virtual double3 sample_f(const double3 &wo, double3 *wi, double *pdf) = 0;

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy double3.
   * \return emission double3 of the surface material
   */
  virtual double3 get_emission() const = 0;

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  virtual bool is_delta() const = 0;

  virtual void render_debugger_node() {};

  /**
   * Reflection helper
   */
  virtual void reflect(const double3 &wo, double3 *wi);

  /**
   * Refraction helper
   */
  virtual bool refract(const double3 &wo, double3 *wi, double ior);

  const HDRImageBuffer *reflectanceMap;
  const HDRImageBuffer *normalMap;

}; // class BSDF

/**
 * Diffuse BSDF.
 */
class DiffuseBSDF : public BSDF {
 public:

  /**
   * DiffuseBSDFs are constructed with a double3 as input,
   * which is stored into the member variable `reflectance`.
   */
   DiffuseBSDF(const Vector3D& a) : reflectance(to_double3(a)) {}
  DiffuseBSDF(const double3 &a) : reflectance(a) {}

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf);

  [[nodiscard]] double3 get_emission() const { return double3(); }

  bool is_delta() const { return false; }

  /*
   * Reflectance is also commonly called the "albedo" of a surface,
   * which ranges from [0,1] in RGB, representing a range of
   * total absorption(0) vs. total reflection(1) per color channel.
   */
  double3 reflectance;
  /*
   * A sampler object that can be used to obtain
   * a random double3 sampled according to a
   * cosine-weighted hemisphere distribution.
   * See pathtracer/sampler.cpp.
   */
  CosineWeightedHemisphereSampler3D sampler;

}; // class DiffuseBSDF

/**
 * Microfacet BSDF.
 */

class MicrofacetBSDF : public BSDF {
 public:

  MicrofacetBSDF(const double3 &eta, const double3 &k, double alpha)
      : eta(eta), k(k), alpha(alpha) {}

  double getTheta(const double3 &w) {
    return acos(clamp(w.z, -1.0 + 1e-5, 1.0 - 1e-5));
  }

  double Lambda(const double3 &w) {
    double theta = getTheta(w);
    double a = 1.0 / (alpha * tan(theta));
    return 0.5 * (erf(a) - 1.0 + std::exp(-a * a) / (a * PI));
  }

  double3 F(const double3 &wi);

  double G(const double3 &wo, const double3 &wi);

  double D(const double3 &h);

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf);

  double3 get_emission() const { return double3(); }

  bool is_delta() const { return false; }

  double3 eta, k;
  double alpha;
  UniformGridSampler2D sampler;
  CosineWeightedHemisphereSampler3D cosineHemisphereSampler;
}; // class MicrofacetBSDF

/**
 * Mirror BSDF
 */
class MirrorBSDF : public BSDF {
 public:

  MirrorBSDF(const double3 &reflectance) : reflectance(reflectance) {}

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf);

  double3 get_emission() const { return double3(); }

  bool is_delta() const { return true; }


  double roughness;
  double3 reflectance;

}; // class MirrorBSDF*/

/**
 * Refraction BSDF.
 */
class RefractionBSDF : public BSDF {
 public:

  RefractionBSDF(const double3 &transmittance, double roughness, double ior)
      : transmittance(transmittance), roughness(roughness), ior(ior) {}

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf);

  double3 get_emission() const { return double3(); }

  bool is_delta() const { return true; }


  double ior;
  double roughness;
  double3 transmittance;

}; // class RefractionBSDF

/**
 * Dielectric BSDF.
 */
class GlassBSDF : public BSDF {
 public:

  GlassBSDF(const double3 &transmittance, const double3 &reflectance,
            double roughness, double ior) :
      transmittance(transmittance), reflectance(reflectance),
      roughness(roughness), ior(ior) {}

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf);

  double3 get_emission() const { return double3(); }

  bool is_delta() const { return true; }


  double ior;
  double roughness;
  double3 reflectance;
  double3 transmittance;

}; // class GlassBSDF

/**
 * Emission BSDF.
 */
class EmissionBSDF final : public BSDF {
 public:
  explicit EmissionBSDF(const Vector3D &radiance) : radiance(to_double3(radiance)) {}
  explicit EmissionBSDF(const double3 &radiance) : radiance(radiance) {}

  double3 f(const double3 &wo, const double3 &wi);

  double3 sample_f(const double3 &wo, double3 *wi, double *pdf) override;

  [[nodiscard]] double3 get_emission() const override { return radiance; }

  [[nodiscard]] bool is_delta() const override { return false; }


  double3 radiance;
  CosineWeightedHemisphereSampler3D sampler;

}; // class EmissionBSDF

}  // namespace CGL

#endif  // CGL_STATICSCENE_BSDF_H
