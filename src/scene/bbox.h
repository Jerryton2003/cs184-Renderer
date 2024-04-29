#ifndef CGL_BBOX_H
#define CGL_BBOX_H

#include <utility>
#include <algorithm>

#include "CGL/CGL.h"
#include "../util/device-utils.h"
#include "../util/device-vec-ops.h"
#include <thrust/swap.h>

namespace CGL {
/**
 * Axis-aligned bounding box.
 * An AABB is given by two positions in space, the min and the max. An addition
 * component, the extent of the bounding box is stored as it is useful in a lot
 * of the operations on bounding boxes.
 */
struct BBox {
  double3 hi{-1e9, -1e9, -1e9}; ///< min corner of the bounding box
  double3 lo{1e9, 1e9, 1e9}; ///< max corner of the bounding box

  CUDA_CALLABLE CUDA_INLINE
  BBox() = default;

  /**
   * Constructor.
   * Creates a bounding box that includes a single point.
   */
  CUDA_CALLABLE CUDA_INLINE
  BBox(const double3 &p) : lo(p), hi(p) {
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  BBox(const BBox &bbox) : lo(bbox.lo), hi(bbox.hi) {
  }

  /**
   * Constructor.
   * Creates a bounding box with given bounds.
   * \param min the min corner
   * \param max the max corner
   */
  CUDA_CALLABLE CUDA_INLINE
  BBox(const double3 &min, const double3 &max) : lo(min), hi(max) {
  }

  CUDA_CALLABLE CUDA_INLINE
  BBox(const double minX,
       const double minY,
       const double minZ,
       const double maxX,
       const double maxY,
       const double maxZ) {
    lo = make_double3(minX, minY, minZ);
    hi = make_double3(maxX, maxY, maxZ);
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  BBox &operator=(const BBox &bbox) = default;

  CUDA_CALLABLE CUDA_FORCEINLINE
  BBox &operator=(BBox &&bbox) noexcept {
    lo = bbox.lo;
    hi = bbox.hi;
    return *this;
  }

  CUDA_CALLABLE CUDA_INLINE
  void expand(const BBox &bbox) {
#ifdef __CUDA_ARCH__
    lo.x = fmin(lo.x, bbox.lo.x);
    lo.y = fmin(lo.y, bbox.lo.y);
    lo.z = fmin(lo.z, bbox.lo.z);
    hi.x = fmax(hi.x, bbox.hi.x);
    hi.y = fmax(hi.y, bbox.hi.y);
    hi.z = fmax(hi.z, bbox.hi.z);
#else
    lo.x = std::min(lo.x, bbox.lo.x);
    lo.y = std::min(lo.y, bbox.lo.y);
    lo.z = std::min(lo.z, bbox.lo.z);
    hi.x = std::max(hi.x, bbox.hi.x);
    hi.y = std::max(hi.y, bbox.hi.y);
    hi.z = std::max(hi.z, bbox.hi.z);
#endif
  }

  /**
   * Expand the bounding box to include a new point in space.
   * If the given point is already inside *this*, nothing happens.
   * Otherwise *this* is expanded to a minimum volume that contains the given
   * point.
   * \param p the point to be included
   */
  CUDA_CALLABLE CUDA_INLINE
  void expand(const double3 &p) {
#ifdef __CUDA_ARCH__
    lo.x = fmin(lo.x, p.x);
    lo.y = fmin(lo.y, p.y);
    lo.z = fmin(lo.z, p.z);
    hi.x = fmax(hi.x, p.x);
    hi.y = fmax(hi.y, p.y);
    hi.z = fmax(hi.z, p.z);
#else
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x);
    hi.y = std::max(hi.y, p.y);
    hi.z = std::max(hi.z, p.z);
#endif
  }

  void expand(const Vector3D &p) {
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x);
    hi.y = std::max(hi.y, p.y);
    hi.z = std::max(hi.z, p.z);
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double3 centroid() const {
    return (lo + hi) / 2;
  }

  /**
   * Compute the surface area of the bounding box.
   * \return surface area of the bounding box.
   */
  CUDA_CALLABLE CUDA_INLINE
  double surface_area() const {
    if (empty()) return 0.0;
    double3 d = hi - lo;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
  }

  /**
   * Check if bounding box is empty.
   * Bounding box that has no size is considered empty. Note that since
   * bounding box are used for objects with positive volumes, a bounding
   * box of zero size (empty, or contains a single vertex) are considered
   * empty.
   */
  CUDA_CALLABLE CUDA_INLINE
  bool empty() const {
    return lo.x > hi.x || lo.y > hi.y || lo.z > hi.z;
  }

  /**
   * Ray - bbox intersection.
   * Intersects ray with bounding box, does not store shading information.
   * \param r the ray to intersect with
   */
  CUDA_DEVICE [[nodiscard]] CUDA_INLINE
  bool intersect(const double3 &o, const double3 &d, double max_t) const {
    double3 invDir = 1.0 / d;
    int3 dirIsNeg = make_int3(d.x < 0.0, d.y < 0.0, d.z < 0.0);
    double3 tMin = (lo - o) * invDir;
    double3 tMax = (hi - o) * invDir;
    if (dirIsNeg.x) thrust::swap(tMin.x, tMax.x);
    if (dirIsNeg.y) thrust::swap(tMin.y, tMax.y);
    if (dirIsNeg.z) thrust::swap(tMin.z, tMax.z);
    double t_in = fmax(fmax(tMin.x, tMin.y), tMin.z);
    double t_out = fmin(fmin(tMax.x, tMax.y), tMax.z);
    if (t_in > t_out || t_out < 0.0 || t_in > max_t) return false;
    return true;
  }

  /**
   * Ray - bbox intersection.
   * Intersects ray with bounding box, does not store shading information.
   * \param r the ray to intersect with
   */
  CUDA_DEVICE [[nodiscard]] CUDA_INLINE
  bool intersect(const double3 &o, const double3 &d) const {
    double3 invDir = 1.0 / d;
    int dirIsNeg[3] = {d.x < 0.0, d.y < 0.0, d.z < 0.0};
    double3 tMin = (lo - o) * invDir;
    double3 tMax = (hi - o) * invDir;
    if (dirIsNeg[0]) thrust::swap(tMin.x, tMax.x);
    if (dirIsNeg[1]) thrust::swap(tMin.y, tMax.y);
    if (dirIsNeg[2]) thrust::swap(tMin.z, tMax.z);
    double t_in = fmax(fmax(tMin.x, tMin.y), tMin.z);
    double t_out = fmin(fmin(tMax.x, tMax.y), tMax.z);
    if (t_in > t_out || t_out < 0.0) return false;
    return true;
  }

};

inline std::ostream &operator<<(std::ostream &os, const BBox &bbox) {
  os << "BBox(" << bbox.lo.x << ", " << bbox.lo.y << ", " << bbox.lo.z << ", "
     << bbox.hi.x << ", " << bbox.hi.y << ", " << bbox.hi.z << ")";
  return os;
}

struct mergeBBox {

  CUDA_CALLABLE CUDA_INLINE
  BBox operator()(const BBox &a, const BBox &b) const {
    BBox res;
    res.lo = make_double3(fmin(a.lo.x, b.lo.x),
                          fmin(a.lo.y, b.lo.y),
                          fmin(a.lo.z, b.lo.z));
    res.hi = make_double3(fmax(a.hi.x, b.hi.x),
                          fmax(a.hi.y, b.hi.y),
                          fmax(a.hi.z, b.hi.z));
    return res;
  }
};
} // namespace CGL

#endif // CGL_BBOX_H
