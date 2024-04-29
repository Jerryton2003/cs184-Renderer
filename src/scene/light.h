#ifndef CGL_STATICSCENE_LIGHT_H
#define CGL_STATICSCENE_LIGHT_H

#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"
#include "pathtracer/sampler.h" // UniformHemisphereSampler3D, UniformGridSampler2D
#include "util/image.h"   // HDRImageBuffer

#include "scene.h"  // SceneLight
#include "object.h" // Mesh, SphereObject
#include "util/device-utils.h"
#include <format>
namespace CGL {
namespace SceneObjects {
// Directional Light //

class DirectionalLight : public SceneLight {
 public:
  DirectionalLight(const Vector3D &rad, const Vector3D &lightDir) : DirectionalLight(to_double3(rad),
                                                                                     to_double3(lightDir)) {}

  DirectionalLight(const double3 &rad, const double3 &lightDir);

  double3 sample_L(const double3 &p, double3 *wi, double *distToLight,
                   double *pdf) const;

  bool is_delta_light() const { return true; }

 private:
  double3 radiance;
  double3 dirToLight;

}; // class Directional Light

// Infinite Hemisphere Light //

class InfiniteHemisphereLight : public SceneLight {
 public:
  InfiniteHemisphereLight(const double3 &rad);

  double3 sample_L(const double3 &p, double3 *wi, double *distToLight,
                   double *pdf) const;

  bool is_delta_light() const { return false; }

  double3 radiance;
  Matrix3x3 sampleToWorld;
  UniformHemisphereSampler3D sampler;

}; // class InfiniteHemisphereLight


// Point Light //

class PointLight : public SceneLight {
 public:
  PointLight(const Vector3D &rad, const Vector3D &pos) : PointLight(to_double3(rad), to_double3(pos)) {}

  PointLight(const double3 &rad, const double3 &pos);

  double3 sample_L(const double3 &p, double3 *wi, double *distToLight,
                   double *pdf) const;

  bool is_delta_light() const { return true; }

  double3 radiance;
  double3 position;

}; // class PointLight

// Spot Light //

class SpotLight : public SceneLight {
 public:
  SpotLight(const Vector3D &rad, const Vector3D &pos, const Vector3D &dir, double angle) : SpotLight(to_double3(rad),
                                                                                                     to_double3(pos),
                                                                                                     to_double3(dir),
                                                                                                     angle) {}

  SpotLight(const double3 &rad, const double3 &pos,
            const double3 &dir, double angle);

  double3 sample_L(const double3 &p, double3 *wi, double *distToLight,
                   double *pdf) const;

  bool is_delta_light() const { return true; }

  double3 radiance;
  double3 position;
  double3 direction;
  double angle;

}; // class SpotLight

// Area Light //

class AreaLight : public SceneLight {
 public:
  AreaLight(const Vector3D &rad, const Vector3D &pos, const Vector3D &dir,
            const Vector3D &dim_x, const Vector3D &dim_y) : AreaLight(to_double3(rad), to_double3(pos), to_double3(dir),
                                                                      to_double3(dim_x), to_double3(dim_y)) {}

  AreaLight(const double3 &rad,
            const double3 &pos, const double3 &dir,
            const double3 &dim_x, const double3 &dim_y);

  double3 sample_L(const double3 &p, double3 *wi, double *distToLight,
                   double *pdf) const;

  bool is_delta_light() const { return false; }

  double3 radiance;
  double3 position;
  double3 direction;
  double3 dim_x;
  double3 dim_y;
  UniformGridSampler2D sampler;
  double area;

}; // class AreaLight
inline void buildLightAsObjects(Scene &scene) {
  for (SceneLight *light : scene.lights) {
    auto *areaLight = dynamic_cast<AreaLight *>(light);
    if (areaLight) {
      Mesh* mesh = new Mesh();
      mesh->bsdf = new EmissionBSDF(areaLight->radiance);
      mesh->num_vertices = 4;
      mesh->positions = new Vector3D[4];
      mesh->normals = new Vector3D[4];
      mesh->indices = {2, 0, 1, 0, 2, 3};
      mesh->positions[0] = to_Vector3D(areaLight->position - areaLight->dim_x * 0.5 - areaLight->dim_y * 0.5);
      mesh->positions[1] = to_Vector3D(areaLight->position + areaLight->dim_x * 0.5 - areaLight->dim_y * 0.5);
      mesh->positions[2] = to_Vector3D(areaLight->position + areaLight->dim_x * 0.5 + areaLight->dim_y * 0.5);
      mesh->positions[3] = to_Vector3D(areaLight->position - areaLight->dim_x * 0.5 + areaLight->dim_y * 0.5);
      mesh->normals[0] = mesh->normals[1] = mesh->normals[2] = mesh->normals[3] = to_Vector3D(cross(areaLight->dim_x, areaLight->dim_y)).unit();
      scene.objects.push_back(static_cast<SceneObject*>(mesh));
      std::cout << std::format("find one area light, convert to mesh\n");
    }
  }
}
// Sphere Light //
}
} // namespace CGL

#endif  // CGL_STATICSCENE_BSDF_H
