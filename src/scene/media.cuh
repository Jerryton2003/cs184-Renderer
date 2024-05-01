//
// Created by creeper on 4/20/24.
//

#ifndef ASSIGNMENT3_SRC_SCENE_MEDIA_CUH_
#define ASSIGNMENT3_SRC_SCENE_MEDIA_CUH_

#include "../util/gpu-textures.h"
#include "../util/device-vec-ops.h"

namespace CGL {

struct HeterogeneousMedium {
  CudaTextureAccessor<float4> density{};
  CudaTextureAccessor<float4> albedo{};
  int3 resolution{};
  double3 orig{}, spacing{}, majorant{};

  CUDA_DEVICE CUDA_FORCEINLINE double3 getScattering(const double3 &p) const {
    double3 pos = (p - orig) / spacing;
    float4 val = density.sample(pos) * albedo.sample(pos);
    return make_double3(val.x, val.y, val.z);
  }

  CUDA_DEVICE CUDA_FORCEINLINE double3 getAbsorption(const double3 &p) const {
    double3 pos = (p - orig) / spacing;
    float4 val = density.sample(pos) * (1.f - albedo.sample(pos));
    return make_double3(val.x, val.y, val.z);
  }

  CUDA_DEVICE CUDA_FORCEINLINE double3 getMajorant() const {
    return majorant;
  }
};

struct HomogeneousMedium {
  double3 sigma_a{};
  double3 sigma_s{};

  HomogeneousMedium() = default;

  CUDA_CALLABLE HomogeneousMedium(const double3 &sigma_a, const double3 &sigma_s)
      : sigma_a(sigma_a), sigma_s(sigma_s) {}

  CUDA_DEVICE CUDA_FORCEINLINE double3 getScattering(const double3 &p) const {
    return sigma_s;
  }

  CUDA_DEVICE CUDA_FORCEINLINE double3 getAbsorption(const double3 &p) const {
    return sigma_a;
  }

  CUDA_DEVICE CUDA_FORCEINLINE double3 getMajorant() const {
    return sigma_a + sigma_s;
  }
};

struct Medium {
#define FOREACH_MEDIUM_TYPE(MEDIUM_TYPE) \
  MEDIUM_TYPE(HomogeneousMedium) \
  MEDIUM_TYPE(HeterogeneousMedium)

#define GENERATE_ENUM(ENUM) ENUM##_enum,
  enum {
    FOREACH_MEDIUM_TYPE(GENERATE_ENUM)
  } type{};
#undef GENERATE_ENUM
#define GENERATE_UNION(UNION) UNION UNION##_union;
  union {
    FOREACH_MEDIUM_TYPE(GENERATE_UNION)
  } data{};
#undef GENERATE_UNION
#define SWITCH_DESPATCH(replace) \
    do {                             \
      switch (type) {                \
        FOREACH_MEDIUM_TYPE(replace) \
      }                              \
      cuExit();                     \
    } while (0)
#define GENERATE_GET_SCATTERING(MEDIUM_TYPE) \
case MEDIUM_TYPE##_enum: return data.MEDIUM_TYPE##_union.getScattering(p);
  CUDA_DEVICE CUDA_FORCEINLINE double3 getScattering(const double3 &p) const {
    SWITCH_DESPATCH(GENERATE_GET_SCATTERING);
  }

#undef GENERATE_GET_SCATTERING
#define GENERATE_GET_ABSORPTION(MEDIUM_TYPE) \
case MEDIUM_TYPE##_enum: return data.MEDIUM_TYPE##_union.getAbsorption(p);
  CUDA_DEVICE CUDA_FORCEINLINE double3 getAbsorption(const double3 &p) const {
    SWITCH_DESPATCH(GENERATE_GET_ABSORPTION);
  }

#undef GENERATE_GET_ABSORPTION
#define GENERATE_GET_MAJORANT(MEDIUM_TYPE) \
case MEDIUM_TYPE##_enum: return data.MEDIUM_TYPE##_union.getMajorant();
  CUDA_DEVICE CUDA_FORCEINLINE double3 getMajorant() const {
    SWITCH_DESPATCH(GENERATE_GET_MAJORANT);
  }

#undef GENERATE_GET_MAJORANT
#define GENERATE_GET_MEDIUM(MEDIUM_TYPE) \
MEDIUM_TYPE& get##MEDIUM_TYPE() { type = MEDIUM_TYPE##_enum; return data.MEDIUM_TYPE##_union; } \
const MEDIUM_TYPE& get##MEDIUM_TYPE() const { return data.MEDIUM_TYPE##_union; }

  FOREACH_MEDIUM_TYPE(GENERATE_GET_MEDIUM)

#undef GENERATE_GET_MEDIUM
#undef SWITCH_DESPATCH
#undef FOREACH_MEDIUM_TYPE
};

}
#endif //ASSIGNMENT3_SRC_SCENE_MEDIA_CUH_
