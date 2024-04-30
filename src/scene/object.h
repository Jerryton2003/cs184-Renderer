#ifndef CGL_STATICSCENE_OBJECT_H
#define CGL_STATICSCENE_OBJECT_H

#include "scene.h"

namespace CGL {
/**
 * A triangle mesh object.
 */
struct Mesh {
  Mesh() = default;

  CUDA_CALLABLE CUDA_FORCEINLINE
  Mesh(ConstDeviceArrayAccessor<double3> positions,
       ConstDeviceArrayAccessor<double3> normals,
       ConstDeviceArrayAccessor<uint32_t> indices)
      : positions(positions), normals(normals), indices(indices) {}

  CUDA_CALLABLE CUDA_FORCEINLINE
  Mesh &operator=(const Mesh &other) {
    positions = other.positions;
    normals = other.normals;
    indices = other.indices;
    return *this;
  }

  ConstDeviceArrayAccessor<double3> positions{}; ///< vertex positions
  ConstDeviceArrayAccessor<double3> normals{}; ///< vertex normal
  ConstDeviceArrayAccessor<uint32_t> indices{}; ///< triangles defined by indices
};
} // namespace CGL

#endif // CGL_STATICSCENE_OBJECT_H
