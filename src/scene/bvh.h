#ifndef CGL_BVH_H
#define CGL_BVH_H

#include "scene.h"
#include "scene/bbox.h"
#include "scene/shape.h"
#include "primitive.h"
#include "../util/device-utils.h"
#include <vector>

namespace CGL {

CUDA_CALLABLE CUDA_FORCEINLINE
uint64_t expandBits21(uint32_t v) {
  auto x = static_cast<uint64_t>(v);
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

struct LBVHAccessor {
  PrimitivePoolAccessor primitives;
  ConstDeviceArrayAccessor<BBox> bbox;
  ConstDeviceArrayAccessor<int> lch;
  ConstDeviceArrayAccessor<int> rch;
  ConstDeviceArrayAccessor<int> idx;

  CUDA_DEVICE CUDA_FORCEINLINE bool isLeaf(int nodeIdx) const {
    return nodeIdx >= primitives.nPrs - 1;
  }

  CUDA_DEVICE bool intersect(const double3 &o,
                             const double3 &d,
                             double3 &isect_n,
                             double &t,
                             int &primitive_id,
                             bool debug = false) const {
    int stack[64];
    int top{};
    int nodeIdx = 0;
    if (!bbox[nodeIdx].intersect(o, d))
      return false;
    bool hit = false;
    double3 tmp_isect_n;
    t = 1e9;
    double tmp_t = 1e9;
    bool flag = false;
    int lc = -1, rc = -1;
    bool intl = false, intr = false;
    while (true) {
      // it is a leaf node
      if (isLeaf(nodeIdx)) {
        int pr_id = idx[nodeIdx - primitives.nPrs + 1];
        flag = primitives.intersect(pr_id, o, d, tmp_isect_n, tmp_t);
        if (flag && (!hit || tmp_t < t)) {
          t = tmp_t;
          isect_n = tmp_isect_n;
          primitive_id = pr_id;
          hit = true;
        }
        if (!top) return hit;
        nodeIdx = stack[--top];
        continue;
      }
      lc = lch[nodeIdx];
      rc = rch[nodeIdx];
      intl = bbox[lc].intersect(o, d, t);
      intr = bbox[rc].intersect(o, d, t);
      if (!intl && !intr) {
        if (!top) return hit;
        nodeIdx = stack[--top];
        continue;
      }
      if (intl && !intr) nodeIdx = lc;
      else if (!intl) nodeIdx = rc;
      else {
        nodeIdx = lc;
        stack[top++] = rc;
      }
    }
  }

  CUDA_DEVICE bool has_intersection(const double3 &o, const double3 &d) const {
    int stack[64];
    int top{};
    int nodeIdx = 0;
    if (!bbox[nodeIdx].intersect(o, d))
      return false;
    bool hit = false;
    while (true) {
      // it is a leaf node
      if (isLeaf(nodeIdx)) {
        int pr_id = idx[nodeIdx - primitives.nPrs + 1];
        bool flag = primitives.has_intersection(pr_id, o, d);
        if (!top && flag) return true;
        nodeIdx = stack[--top];
        continue;
      }
      int lc = lch[nodeIdx];
      int rc = rch[nodeIdx];
      bool intl = bbox[lc].intersect(o, d);
      bool intr = bbox[rc].intersect(o, d);
      if (!intl && !intr) {
        if (!top) return false;
        nodeIdx = stack[--top];
        continue;
      }
      if (intl && !intr) nodeIdx = lc;
      else if (!intl) nodeIdx = rc;
      else {
        nodeIdx = lc;
        stack[top++] = rc;
      }
    }
  }
};

struct LBVH {
  CUDA_HOST LBVH(int pr_cnt, const std::vector<Shape> &shs, const std::vector<Surface> &mats) {
    primitives = std::make_unique<PrimitivePool>(pr_cnt, shs, mats);
    bbox = std::make_unique<DeviceArray<BBox>>((pr_cnt << 1) - 1);
    mortons = std::make_unique<DeviceArray<uint64_t>>(pr_cnt);
    fa = std::make_unique<DeviceArray<int>>((pr_cnt << 1) - 1);
    lch = std::make_unique<DeviceArray<int>>(pr_cnt - 1);
    rch = std::make_unique<DeviceArray<int>>(pr_cnt - 1);
    idx = std::make_unique<DeviceArray<int>>(pr_cnt);
    build();
  }

  CUDA_HOST void build();

  CUDA_HOST ~LBVH() = default;


  CUDA_HOST LBVHAccessor accessor() const {
    return {primitives->accessor(), bbox->constAccessor(), lch->constAccessor(), rch->constAccessor(),
            idx->constAccessor()
    };
  }

  std::unique_ptr<PrimitivePool> primitives{};
  std::unique_ptr<DeviceArray<BBox>> bbox{};
  std::unique_ptr<DeviceArray<uint64_t>> mortons{};
  std::unique_ptr<DeviceArray<int>> fa{};
  std::unique_ptr<DeviceArray<int>> lch{};
  std::unique_ptr<DeviceArray<int>> rch{};
  std::unique_ptr<DeviceArray<int>> idx{};
  BBox scene_bound;
};

} // namespace CGL

#endif // CGL_BVH_H
