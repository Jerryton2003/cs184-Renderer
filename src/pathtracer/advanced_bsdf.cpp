#include "bsdf.h"

#include <algorithm>
#include <iostream>
#include <utility>

#include "application/visual_debugger.h"

using std::max;
using std::min;
using std::swap;

namespace CGL {

// Mirror BSDF //

double3 MirrorBSDF::f(const double3& wo, const double3& wi) {
  return double3();
}

double3 MirrorBSDF::sample_f(const double3& wo, double3* wi, double* pdf) {

  // TODO:
  // Implement MirrorBSDF
  
  return double3();
}

// Microfacet BSDF //

double MicrofacetBSDF::G(const double3& wo, const double3& wi) {
  return 1.0 / (1.0 + Lambda(wi) + Lambda(wo));
}

double MicrofacetBSDF::D(const double3& h) {
  // TODO: proj3-2, part 3
  // Compute Beckmann normal distribution function (NDF) here.
  // You will need the roughness alpha.
  
  return 1.0;
}

double3 MicrofacetBSDF::F(const double3& wi) {
  // TODO: proj3-2, part 3
  // Compute Fresnel term for reflection on dielectric-conductor interface.
  // You will need both eta and etaK, both of which are double3.

  double cosTheta = cos_theta(wi);
  
  return double3();
}

double3 MicrofacetBSDF::f(const double3& wo, const double3& wi) {
  // TODO: proj3-2, part 3
  // Implement microfacet model here.

  return double3();
}

double3 MicrofacetBSDF::sample_f(const double3& wo, double3* wi, double* pdf) {
  // TODO: proj3-2, part 3
  // *Importance* sample Beckmann normal distribution function (NDF) here.
  // Note: You should fill in the sampled direction *wi and the corresponding *pdf,
  //       and return the sampled BRDF value.



  *wi = cosineHemisphereSampler.get_sample(pdf);

  return MicrofacetBSDF::f(wo, *wi);
}

// Refraction BSDF //

double3 RefractionBSDF::f(const double3& wo, const double3& wi) {
  return double3();
}

double3 RefractionBSDF::sample_f(const double3& wo, double3* wi, double* pdf) {

  // TODO:
  // Implement RefractionBSDF
  
  
  return double3();
}

// Dielectric BSDF //

double3 GlassBSDF::f(const double3& wo, const double3& wi) {
  return double3();
}

double3 GlassBSDF::sample_f(const double3& wo, double3* wi, double* pdf) {

  // TODO:
  // Compute Fresnel coefficient and either reflect or refract based on it.

  // compute Fresnel coefficient and use it as the probability of reflection
  // - Fundamentals of Computer Graphics page 305


  return double3();
}

void BSDF::reflect(const double3& wo, double3* wi) {

  // TODO:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  


}

bool BSDF::refract(const double3& wo, double3* wi, double ior) {

  // TODO:
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.




  return true;

}

} // namespace CGL
