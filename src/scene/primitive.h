#ifndef CGL_STATICSCENE_PRIMITIVE_H
#define CGL_STATICSCENE_PRIMITIVE_H

#include "scene/bbox.h"
#include "scene/shape.h"
#include "scene/scene.h"
#include "../util/gpu-arrays.h"

namespace CGL { namespace SceneObjects {

/**
 * The abstract base class primitive is the bridge between geometry processing
 * and the shading subsystem. As such, its interface contains methods related
 * to both.
 */
class Primitive {};

} // namespace SceneObjects

struct Surface {
 SurfaceInfo surface_info;
 int surf_id;
};

struct PrimitivePoolAccessor {
 int nPrs;
 ConstDeviceArrayAccessor<Shape> shapes;
 ConstDeviceArrayAccessor<Surface> materials;
 CUDA_DEVICE [[nodiscard]] CUDA_FORCEINLINE bool has_intersection(int i, const double3&r, const double3&o) const {
  return shapes[i].has_intersection(r, o);
 }
 CUDA_DEVICE CUDA_INLINE bool intersect(int i,
                                        const double3&o,
                                        const double3&d,
                                        double3&isect_n,
                                        double&t) const {
  bool intersected = shapes[i].intersect(o, d, isect_n, t);
  return intersected;
 }
};

struct PrimitivePool {
 int nPrs;
 std::unique_ptr<DeviceArray<Shape>> shapes{};
 std::unique_ptr<DeviceArray<Surface>> materials{};
 PrimitivePool(int n_pr, const std::vector<Shape>& shs, const std::vector<Surface>& mats)
     : nPrs(n_pr) {
  shapes = std::make_unique<DeviceArray<Shape>>(shs);
  materials = std::make_unique<DeviceArray<Surface>>(mats);
 }
 CUDA_HOST [[nodiscard]] CUDA_INLINE PrimitivePoolAccessor accessor() const {
  return {nPrs, shapes->constAccessor(), materials->constAccessor()};
 }
};
} // namespace CGL
#endif //CGL_STATICSCENE_PRIMITIVE_H