
#ifndef CGL_STATICSCENE_SPHERE_H
#define CGL_STATICSCENE_SPHERE_H

#include "object.h"
#include "primitive.h"

namespace CGL { namespace SceneObjects {

/**
 * A sphere from a sphere object.
 * To be consistent with the triangle interface, each sphere primitive is
 * encapsulated in a sphere object. The have exactly the same origin and
 * radius. The sphere primitive may refer back to the sphere object for
 * other information such as surface material.
 */
class Sphere : public Primitive {
 public:

  /**
   * Parameterized Constructor.
   * Construct a sphere with given origin & radius.
   */
  Sphere(const SphereObject* object, const Vector3D o, double r)
    : object(object), o(o), r(r), r2(r*r) { }

  Sphere() {}

  const SphereObject* object; ///< pointer to the sphere object

  Vector3D o; ///< origin of the sphere
  double r;   ///< radius
  double r2;  ///< radius squared

}; // class Sphere

} // namespace SceneObjects
} // namespace CGL

#endif // CGL_STATICSCENE_SPHERE_H
