//
// Created by creeper on 3/4/24.
//

#ifndef SHAPE_H
#define SHAPE_H

#include <variant>
#include <cstdint>
#include "../scene/object.h"
#include "../util/device-utils.h"
#include "../util/device-vec-ops.h"
#include <algorithm>
#include "bbox.h"

namespace CGL {
struct Sphere {
  double3 center;
  double radius;

  CUDA_DEVICE CUDA_FORCEINLINE BBox get_bbox() const {
    return {make_double3(center.x - radius, center.y - radius, center.z - radius),
            make_double3(center.x + radius, center.y + radius, center.z + radius)};
  }

  CUDA_DEVICE CUDA_FORCEINLINE bool has_intersection(const double3 &o, const double3 &d) const {
    double a = dot(d, d);
    double b = 2 * dot((o - center), d);
    double c = dot(o - center, o - center) - radius * radius;
    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
      return false;
    double t2 = (-b + sqrt(discriminant)) / (2 * a);
    double t1 = (-b - sqrt(discriminant)) / (2 * a);
    if (t2 >= 0)
      return true;
    if (t1 >= 0)
      return true;
    return false;
  }

  CUDA_DEVICE CUDA_INLINE bool intersect(const double3 &o,
                                         const double3 &d,
                                         double3 &isect_n,
                                         double &t) const {
    double a = dot(d, d);
    double b = 2 * dot((o - center), d);
    double c = dot(o - center, o - center) - radius * radius;
    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
      return false;
    double t1, t2;
    if (b >= 0.0) {
      t1 = (-b - sqrt(discriminant)) / (2 * a);
      t2 = 2 * c / (-b - sqrt(discriminant));
    } else {
      t1 = 2 * c / (-b + sqrt(discriminant));
      t2 = (-b + sqrt(discriminant)) / (2 * a);
    }
    assert(t1 <= t2);
    if (t1 >= 0)
      t = t1;
    else if (t2 >= 0)
      t = t2;
    else return false;
    double3 inter = o + t * d;
    isect_n = normalize(inter - center);
    return true;
  }

  CUDA_DEVICE CUDA_INLINE double3 sample(const double2 &uv, double *pdf, double3 *normal) const {
    double z = 1 - 2 * uv.x;
    double r = sqrt(max(0.0, 1 - z * z));
    double phi = 2 * PI * uv.y;
    *pdf = 1 / (4 * PI);
    *normal = make_double3(cos(phi), sin(phi), z);
    return center + radius * make_double3(r * cos(phi), r * sin(phi), z);
  }

  CUDA_DEVICE CUDA_FORCEINLINE double pdf(const double3 &pos) const {
    return 1 / (4 * PI);
  }
};

/**
 * A single triangle from a mesh.
 * To save space, it holds a pointer back to the data in the original mesh
 * rather than holding the data itself. This means that its lifetime is tied
 * to that of the original mesh. The primitive may refer back to the mesh
 * object for other information such as normal, texcoord, material.
 */
class Triangle {
 public:

  Triangle(Mesh *mesh_, int idx_offset_) : mesh(mesh_), idx_offset(idx_offset_) {
    checkDevicePtr(mesh_);
  };

  /**
   * Get the world space bounding box of the triangle.
   * \return world space bounding box of the triangle
   */
  CUDA_DEVICE CUDA_FORCEINLINE
  BBox get_bbox() const {
    const double3 &v0 = mesh->positions[mesh->indices[idx_offset]];
    const double3 &v1 = mesh->positions[mesh->indices[idx_offset + 1]];
    const double3 &v2 = mesh->positions[mesh->indices[idx_offset + 2]];
    return {make_double3(fmin(fmin(v0.x, v1.x), v2.x),
                         fmin(fmin(v0.y, v1.y), v2.y),
                         fmin(fmin(v0.z, v1.z), v2.z)),
            make_double3(fmax(fmax(v0.x, v1.x), v2.x),
                         fmax(fmax(v0.y, v1.y), v2.y),
                         fmax(fmax(v0.z, v1.z), v2.z))};
  }

  CUDA_DEVICE CUDA_INLINE
  bool has_intersection(const double3 &o, const double3 &d) const {
    const auto &p1 = mesh->positions[mesh->indices[idx_offset]];
    const auto &p2 = mesh->positions[mesh->indices[idx_offset + 1]];
    const auto &p3 = mesh->positions[mesh->indices[idx_offset + 2]];
    double3 e1 = p2 - p1;
    double3 e2 = p3 - p1;
    double3 s = o - p1;
    double3 s1 = cross(d, e2);
    double3 s2 = cross(s, e1);

    double divisor = dot(s1, e1);
    if (divisor == 0) return false;
    double invDivisor = 1.0 / divisor;
    double b1 = dot(s1, s) * invDivisor;
    double b2 = dot(s2, d) * invDivisor;
    double b3 = 1.0 - b1 - b2;

    double t = dot(s2, e2) * invDivisor;

    if (t >= 0.0 && b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0)
      return true;
    return false;
  }

  CUDA_DEVICE CUDA_INLINE
  bool intersect(const double3 &o, const double3 &d, double3 &isect_n, double &t) const {
    const auto &p1 = mesh->positions[mesh->indices[idx_offset]];
    const auto &p2 = mesh->positions[mesh->indices[idx_offset + 1]];
    const auto &p3 = mesh->positions[mesh->indices[idx_offset + 2]];
    const auto &n1 = mesh->normals[mesh->indices[idx_offset]];
    const auto &n2 = mesh->normals[mesh->indices[idx_offset + 1]];
    const auto &n3 = mesh->normals[mesh->indices[idx_offset + 2]];
    double3 e1 = p2 - p1;
    double3 e2 = p3 - p1;
    double3 s = o - p1;
    double3 s1 = cross(d, e2);
    double3 s2 = cross(s, e1);

    double divisor = dot(s1, e1);

    if (divisor == 0) return false;
    double invDivisor = 1.0 / divisor;
    double b1 = dot(s1, s) * invDivisor;
    double b2 = dot(s2, d) * invDivisor;
    double b3 = 1.0 - b1 - b2;

    t = dot(s2, e2) * invDivisor;
    if (t < 0.0 || b1 < 0.0 || b2 < 0.0 || b3 < 0.0)
      return false;
    isect_n = normalize(b3 * n1 + b1 * n2 + b2 * n3);
    return true;
  }

  CUDA_DEVICE CUDA_INLINE double3 sample(const double2 &uv, double *pdf, double3 *normal) const {
    const auto &p1 = mesh->positions[mesh->indices[idx_offset]];
    const auto &p2 = mesh->positions[mesh->indices[idx_offset + 1]];
    const auto &p3 = mesh->positions[mesh->indices[idx_offset + 2]];
    const auto &n1 = mesh->normals[mesh->indices[idx_offset]];
    const auto &n2 = mesh->normals[mesh->indices[idx_offset + 1]];
    const auto &n3 = mesh->normals[mesh->indices[idx_offset + 2]];
    double u = sqrt(uv.x);
    double v = uv.y;
    double b1 = 1.0 - u;
    double b2 = u * (1.0 - v);
    double b3 = u * v;
    *pdf = 2.0 / length(cross(p2 - p1, p3 - p1));
    *normal = normalize(b1 * n1 + b2 * n2 + b3 * n3);
    return b1 * p1 + b2 * p2 + b3 * p3;
  }

  CUDA_DEVICE CUDA_FORCEINLINE double pdf(const double3 &pos) const {
    const auto &p1 = mesh->positions[mesh->indices[idx_offset]];
    const auto &p2 = mesh->positions[mesh->indices[idx_offset + 1]];
    const auto &p3 = mesh->positions[mesh->indices[idx_offset + 2]];
    return 2.0 / length(cross(p2 - p1, p3 - p1));
  }

  Mesh *mesh;
  int idx_offset;
}; // class Triangle

struct Shape {
#define FOREACH_SHAPE_TYPE(replace)\
replace(Sphere) \
replace(Triangle)
#define ENUM_ITEM(name) name##_enum,
  enum : uint8_t {
    FOREACH_SHAPE_TYPE(ENUM_ITEM)
  } type{};
#undef ENUM_ITEM
#define SWITCH_DESPATCH(replace) \
  do { \
    switch(type) { \
      FOREACH_SHAPE_TYPE(replace) \
    } \
  } while(0)
#define UNION_ITEM(name) name name##_union;
  union {
    FOREACH_SHAPE_TYPE(UNION_ITEM)
  } item{};
#undef UNION_ITEM
#define SWITCH_CASE_CLAUSE(name) \
case name##_enum: return item.name##_union.get_bbox();

  CUDA_DEVICE CUDA_FORCEINLINE BBox get_bbox() const {
    SWITCH_DESPATCH(SWITCH_CASE_CLAUSE);

  }

#undef SWITCH_CASE_CLAUSE
#define SWITCH_CASE_CLAUSE(name) \
case name##_enum: return item.name##_union.has_intersection(o, d);

  CUDA_DEVICE CUDA_FORCEINLINE bool has_intersection(const double3 &o, const double3 &d) const {
    SWITCH_DESPATCH(SWITCH_CASE_CLAUSE);

  }

#undef SWITCH_CASE_CLAUSE
#define SWITCH_CASE_CLAUSE(name) \
case name##_enum: return item.name##_union.intersect(o, d, isect_n, t);

  CUDA_DEVICE CUDA_FORCEINLINE bool intersect(const double3 &o,
                                              const double3 &d,
                                              double3 &isect_n,
                                              double &t) const {
    SWITCH_DESPATCH(SWITCH_CASE_CLAUSE);
  }

#undef SWITCH_CASE_CLAUSE

#define GET_ITEM(name) \
CUDA_CALLABLE CUDA_FORCEINLINE const name& get##name() const { return item.name##_union; } \
CUDA_CALLABLE CUDA_FORCEINLINE name& get##name() { type = name##_enum; return item.name##_union;}

  FOREACH_SHAPE_TYPE(GET_ITEM)

#undef GET_ITEM

#define SWITCH_CASE_CLAUSE(name) \
case name##_enum: return item.name##_union.sample(uv, pdf, normal);

  CUDA_DEVICE CUDA_FORCEINLINE double3 sample(const double2 &uv, double *pdf, double3 *normal) const {
    SWITCH_DESPATCH(SWITCH_CASE_CLAUSE);

  }

#undef SWITCH_CASE_CLAUSE

#define SWITCH_CASE_CLAUSE(name) \
case name##_enum: return item.name##_union.pdf(pos);

  CUDA_DEVICE CUDA_FORCEINLINE double pdf(const double3 &pos) const {
    SWITCH_DESPATCH(SWITCH_CASE_CLAUSE);

  }

#undef SWITCH_DESPATCH
};
}
#endif //SHAPE_H
