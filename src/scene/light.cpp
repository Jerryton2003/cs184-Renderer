#include "light.h"

#include <iostream>

#include "pathtracer/sampler.h"
#include "../util/device-vec-ops.h"
namespace CGL {
namespace SceneObjects {
// Directional Light //

DirectionalLight::DirectionalLight(const double3 &rad,
                                   const double3 &lightDir)
    : radiance(rad) {
  dirToLight = -normalize(lightDir);
}

double3 DirectionalLight::sample_L(const double3 &p, double3 *wi,
                                   double *distToLight, double *pdf) const {
  *wi = dirToLight;
  *distToLight = INF_D;
  *pdf = 1.0;
  return radiance;
}

// Infinite Hemisphere Light //

InfiniteHemisphereLight::InfiniteHemisphereLight(const double3 &rad)
    : radiance(rad) {
  sampleToWorld[0] = Vector3D(1, 0, 0);
  sampleToWorld[1] = Vector3D(0, 0, -1);
  sampleToWorld[2] = Vector3D(0, 1, 0);
}

double3 InfiniteHemisphereLight::sample_L(const double3 &p, double3 *wi,
                                          double *distToLight,
                                          double *pdf) const {
  double3 dir = sampler.get_sample();
  *wi = to_Mat3(sampleToWorld) * dir;
  *distToLight = INF_D;
  *pdf = 1.0 / (2.0 * PI);
  return radiance;
}

// Point Light //

PointLight::PointLight(const double3 &rad, const double3 &pos) :
    radiance(rad), position(pos) {}

double3 PointLight::sample_L(const double3 &p, double3 *wi,
                             double *distToLight,
                             double *pdf) const {
  double3 d = position - p;
  *wi = normalize(d);
  *distToLight = length(d);
  *pdf = 1.0;
  return radiance;
}


// Spot Light //

SpotLight::SpotLight(const double3 &rad, const double3 &pos,
                     const double3 &dir, double angle) {

}

double3 SpotLight::sample_L(const double3 &p, double3 *wi,
                            double *distToLight, double *pdf) const {
  return double3();
}


// Area Light //

AreaLight::AreaLight(const double3 &rad,
                     const double3 &pos, const double3 &dir,
                     const double3 &dim_x, const double3 &dim_y)
    : radiance(rad), position(pos), direction(dir),
      dim_x(dim_x), dim_y(dim_y), area(length(dim_x) * length(dim_y)) {}

double3 AreaLight::sample_L(const double3 &p, double3 *wi,
                            double *distToLight, double *pdf) const {

  double2 sample = sampler.get_sample() - make_double2(0.5f, 0.5f);
  double3 d = position + sample.x * dim_x + sample.y * dim_y - p;
  double cosTheta = dot(d, direction);
  double sqDist = length(d) * length(d);
  double dist = sqrt(sqDist);
  *wi = d / dist;
  *distToLight = dist;
  *pdf = sqDist / (area * fabs(cosTheta));
  return cosTheta < 0 ? radiance : double3();
};
}

} // namespace CGL
