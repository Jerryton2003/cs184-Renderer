#ifndef CGL_SCENE_H
#define CGL_SCENE_H

#include "CGL/CGL.h"
#include <memory>
#include <vector>
#include "pathtracer/sampler.h"
#include "util/gpu-textures.h"

namespace CGL {
struct DiscreteDistribution;
struct DiffuseData {
  double3 albedo;

  DiffuseData(const double3 &albedo) : albedo(albedo) {
  }
};

struct SpecularData {
  double3 albedo;
  double roughness;

  SpecularData(const double3 &albedo, double roughness) : albedo(albedo), roughness(roughness) {
  }
};

struct DielectricData {
  DielectricData(const double3 &albedo, double roughness, double ior)
      : albedo(albedo), roughness(roughness), ior(ior) {
  }

  double3 albedo{};
  double roughness{};
  double ior{};
};

struct MirrorData {
  double3 albedo;
  double roughness;

  MirrorData(const double3 &albedo, double roughness) : albedo(albedo), roughness(roughness) {
  }
};

struct EmissiveData {
  double3 radiance;

  EmissiveData(const double3 &radiance) : radiance(radiance) {
  }
};

struct HenyeyGreensteinData {
  double g;

  HenyeyGreensteinData(double g) : g(g) {
  }
};

struct IsotropicData {
  double3 albedo{};
  IsotropicData() = default;
  IsotropicData(const double3 &albedo) : albedo(albedo) {
  }
};

struct LightSamplerAccessor {
  int nLights;
  ConstDeviceArrayAccessor<double> accept;
  ConstDeviceArrayAccessor<double> probabilities;
  ConstDeviceArrayAccessor<int> alias;
  ConstDeviceArrayAccessor<int> light_indices;
  ConstDeviceArrayAccessor<int> map_primitive_to_light;

  struct SampleRecord {
    int pr_idx;
    double pdf;
  };
  CUDA_DEVICE CUDA_FORCEINLINE
  SampleRecord sample(const double2 &uv) const {
    int idx = static_cast<int>(uv.x * nLights);
    double pdf = accept[idx];
    if (uv.y < pdf)
      return {light_indices[idx], probabilities[idx]};
    return {light_indices[alias[idx]], probabilities[alias[idx]]};
  }

  CUDA_DEVICE CUDA_FORCEINLINE
  double prob(int light_id) const {
    return probabilities[light_id];
  }

  CUDA_DEVICE CUDA_FORCEINLINE
  double probPrimitive(int primitive_id) const {
    return prob(map_primitive_to_light[primitive_id]);
  }
};

struct LightSampler {
  std::unique_ptr<DiscreteDistribution> light_dist;
  std::unique_ptr<DeviceArray<int>> light_indices;
  std::unique_ptr<DeviceArray<int>> map_light_to_primitive;

  [[nodiscard]] LightSamplerAccessor accessor() const {
    return {
        static_cast<int>(light_dist->alias->size()), light_dist->accept->constAccessor(),
        light_dist->probabilities->constAccessor(), light_dist->alias->constAccessor(),
        light_indices->constAccessor(), map_light_to_primitive->constAccessor()
    };
  }
};

struct MediumInterfaceData {
  int8_t internal_id; // in theory, this toy renderer cannot handle that many media
  int8_t external_id;
};
enum SurfaceInfo : uint8_t {
  Diffuse,
  Specular,
  Dielectric,
  Mirror,
  Emissive,
  MediumInterface,
  EnvMap,
  NumSurfaceInfos
};
enum PhaseFunctions : uint8_t {
  Isotropic,
  HenyeyGreenstein,
  NumPhaseFunctions
};
struct PhaseFunction {
  PhaseFunctions pf;
  int pf_id;
};
struct Mesh;
struct Medium;
/**
 * Represents a scene in a raytracer-friendly format. To speed up raytracing,
 * all data is already transformed to world space.
 */
struct Scene {
  struct MeshPool {
    std::vector<DeviceArray<double3>> vertices{};
    std::vector<DeviceArray<double3>> normals{};
    std::vector<DeviceArray<uint32_t>> indices{};
  } mesh_pool;
  std::vector<std::unique_ptr<CudaTexture<float4>>> vol_textures;
  std::unique_ptr<DeviceArray<DiffuseData>> diffuse_data;
  std::unique_ptr<DeviceArray<SpecularData>> specular_data;
  std::unique_ptr<DeviceArray<DielectricData>> dielectric_data;
  std::unique_ptr<DeviceArray<MirrorData>> mirror_data;
  std::unique_ptr<DeviceArray<EmissiveData>> emissive_data;
  std::unique_ptr<DeviceArray<MediumInterfaceData>> medium_interface_data;
  std::unique_ptr<DeviceArray<Mesh>> meshes;
  std::unique_ptr<DeviceArray<Medium>> media;
  std::unique_ptr<DeviceArray<PhaseFunction>> phase_functions;
  std::unique_ptr<DeviceArray<IsotropicData>> isotropic_data;
  std::unique_ptr<DeviceArray<HenyeyGreensteinData>> hg_data;
  std::unique_ptr<LightSampler> light_sampler;
};
} // namespace CGL

#endif //CGL_SCENE_H