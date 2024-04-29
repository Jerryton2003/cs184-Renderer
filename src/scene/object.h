#ifndef CGL_STATICSCENE_OBJECT_H
#define CGL_STATICSCENE_OBJECT_H

#include "util/halfEdgeMesh.h"
#include "scene.h"

namespace CGL {
namespace SceneObjects {

/**
 * A triangle mesh object.
 */
class Mesh : public SceneObject {
 public:

  Mesh() = default;
  /**
   * Constructor.
   * Construct a static mesh for rendering from halfedge mesh used in editing.
   * Note that this converts the input halfedge mesh into a collection of
   * world-space triangle primitives.
   */
  Mesh(const HalfedgeMesh& mesh, BSDF* bsdf);

  /**
   * Get the BSDF of the surface material of the mesh.
   * \return BSDF of the surface material of the mesh
   */
  BSDF* get_bsdf() const;

  Vector3D *positions{};  ///< position array
  Vector3D *normals{};    ///< normal array
  vector<size_t> indices{};  ///< triangles defined by indices
  int num_vertices{};     ///< number of vertices
  BSDF* bsdf; ///< BSDF of surface material


};

/**
 * A sphere object.
 */
class SphereObject : public SceneObject {
 public:

  /**
  * Constructor.
  * Construct a static sphere for rendering from given parameters
  */
  SphereObject(const Vector3D o, double r, BSDF* bsdf);

  /**
   * Get the BSDF of the surface material of the sphere.
   * \return BSDF of the surface material of the sphere
   */
  BSDF* get_bsdf() const;

  Vector3D o; ///< origin
  double r;   ///< radius

  BSDF* bsdf; ///< BSDF of the sphere objects' surface material

}; // class SphereObject


} // namespace SceneObjects
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
  ConstDeviceArrayAccessor<double3> positions{}; ///< vertex positions
  ConstDeviceArrayAccessor<double3> normals{}; ///< vertex normal
  ConstDeviceArrayAccessor<uint32_t> indices{}; ///< triangles defined by indices
};
} // namespace CGL

#endif // CGL_STATICSCENE_OBJECT_H
