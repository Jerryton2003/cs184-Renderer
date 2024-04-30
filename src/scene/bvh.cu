#include "bvh.h"

#include "CGL/CGL.h"
#include "../util/device-utils.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <fstream>
#include <iosfwd>

using namespace std;

namespace CGL {

static CUDA_DEVICE CUDA_FORCEINLINE int clz(uint64_t x) {
  return __clzll(x);
}

static CUDA_DEVICE CUDA_FORCEINLINE unsigned int expandBits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

static CUDA_DEVICE CUDA_FORCEINLINE int lcp(uint64_t a, uint64_t b) {
  // fast longest common prefix
  return clz(a ^ b);
}

static CUDA_DEVICE CUDA_FORCEINLINE int delta(ConstDeviceArrayAccessor<uint64_t> mortons, int n_prs, int i, int j) {
  if (j < 0 || j > n_prs - 1) return -1;
  return lcp(mortons[i], mortons[j]);
}

CUDA_DEVICE CUDA_FORCEINLINE uint32_t mortonCode(const BBox &scene_bound, const double3 &p) {
  unsigned int x = max((p.x - scene_bound.lo.x) / (scene_bound.hi.x - scene_bound.lo.x) * 1024, 0.0);
  unsigned int y = max((p.y - scene_bound.lo.y) / (scene_bound.hi.y - scene_bound.lo.y) * 1024, 0.0);
  unsigned int z = max((p.z - scene_bound.lo.z) / (scene_bound.hi.z - scene_bound.lo.z) * 1024, 0.0);
  return (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
}

CUDA_DEVICE CUDA_FORCEINLINE uint64_t encodeAABB(const BBox &scene_bound, const BBox &aabb, int idx) {
  uint32_t morton = mortonCode(scene_bound, aabb.centroid());
  return (static_cast<uint64_t>(morton) << 32) | idx;
}

CUDA_DEVICE int findSplit(ConstDeviceArrayAccessor<uint64_t> mortons, int l, int r) {
  if (mortons[l] == mortons[r])
    return (l + r) >> 1;
  int commonPrefix = lcp(mortons[l], mortons[r]);
  int search = l;
  int step = r - l;
  do {
    step = (step + 1) >> 1;
    int newSearch = search + step;
    if (newSearch < r) {
      uint64_t splitCode = mortons[newSearch];
      if (lcp(mortons[l], splitCode) > commonPrefix)
        search = newSearch;
    }
  } while (step > 1);
  return search;
}

static CUDA_GLOBAL void kernelComputeBBox(int nPrs,
                                          ConstDeviceArrayAccessor<Shape> shapes,
                                          DeviceArrayAccessor<BBox> bbox) {
  get_and_restrict_tid(tid, nPrs);
  bbox[tid + nPrs - 1] = shapes[tid].get_bbox();
}

static CUDA_GLOBAL void
kernelComputeMortons(int nPrs,
                     BBox scene_bound,
                     ConstDeviceArrayAccessor<BBox> bbox,
                     DeviceArrayAccessor<uint64_t> mortons,
                     DeviceArrayAccessor<int> idx) {
  get_and_restrict_tid(tid, nPrs);
  mortons[tid] = encodeAABB(scene_bound, bbox[tid + nPrs - 1], tid);
  idx[tid] = tid;
}

static CUDA_GLOBAL void kernelComputeStructure(int nPrs,
                                               ConstDeviceArrayAccessor<uint64_t> mortons,
                                               DeviceArrayAccessor<int> fa,
                                               DeviceArrayAccessor<int> lch,
                                               DeviceArrayAccessor<int> rch) {
  get_and_restrict_tid(tid, nPrs - 1);
  int dir = delta(mortons, nPrs, tid, tid + 1) > delta(mortons, nPrs, tid, tid - 1) ? 1 : -1;
  int min_delta = delta(mortons, nPrs, tid, tid - dir);
  int lmax = 2;
  while (delta(mortons, nPrs, tid, tid + lmax * dir) > min_delta) lmax <<= 1;
  int len = 0;
  for (int t = lmax >> 1; t; t >>= 1) {
    if (delta(mortons, nPrs, tid, tid + (len | t) * dir) > min_delta)
      len |= t;
  }
  int l = min(tid, tid + len * dir);
  int r = max(tid, tid + len * dir);
  int split = findSplit(mortons, l, r);
  if (l == split)
    lch[tid] = nPrs - 1 + split;
  else lch[tid] = split;
  if (r == split + 1)
    rch[tid] = nPrs + split;
  else rch[tid] = split + 1;
  fa[rch[tid]] = fa[lch[tid]] = tid;
  if (tid == 0)
    fa[0] = -1;
}

static CUDA_DEVICE CUDA_FORCEINLINE BBox merge(const BBox &a, const BBox &b) {
  BBox res{};
  res.lo = make_double3(fmin(a.lo.x, b.lo.x), fmin(a.lo.y, b.lo.y), fmin(a.lo.z, b.lo.z));
  res.hi = make_double3(fmax(a.hi.x, b.hi.x), fmax(a.hi.y, b.hi.y), fmax(a.hi.z, b.hi.z));
  assert(res.lo.x <= a.lo.x && res.lo.y <= a.lo.y && res.lo.z <= a.lo.z);
  assert(res.lo.x <= b.lo.x && res.lo.y <= b.lo.y && res.lo.z <= b.lo.z);
  assert(res.hi.x >= a.hi.x && res.hi.y >= a.hi.y && res.hi.z >= a.hi.z);
  assert(res.hi.x >= b.hi.x && res.hi.y >= b.hi.y && res.hi.z >= b.hi.z);
  return res;
}

static CUDA_GLOBAL void kernelUpdateBBox(int nPrs,
                                         ConstDeviceArrayAccessor<int> fa,
                                         ConstDeviceArrayAccessor<int> lch,
                                         ConstDeviceArrayAccessor<int> rch,
                                         DeviceArrayAccessor<BBox> bbox,
                                         DeviceArrayAccessor<int> processed) {
  get_and_restrict_tid(tid, nPrs);
  int node_idx = nPrs + tid - 1;
  while (fa[node_idx] != -1) {
    __threadfence();
    int parent = fa[node_idx];
    assert(parent < nPrs - 1);
    auto old = atomicCAS(&processed[parent], 0, 1);
    if (!old) return;
    bbox[parent] = merge(bbox[lch[parent]], bbox[rch[parent]]);
    node_idx = parent;
  }
}

CUDA_HOST void LBVH::build() {
  // fill all the device arrays

  cudaSafeCheck(
      kernelComputeBBox<<<LAUNCH_THREADS(primitives->nPrs)>>>(primitives->nPrs,
                                                              primitives->shapes->constAccessor(),
                                                              bbox->accessor()));
  assert(checkDevicePtr(bbox->begin()));
  cudaSafeCheck(
      scene_bound = thrust::reduce(thrust::device,
                                   thrust::device_ptr<BBox>(bbox->begin() + primitives->nPrs - 1),
                                   thrust::device_ptr<BBox>(bbox->end()), BBox(),
                                   mergeBBox()));
  // print scene bound
  cudaSafeCheck(
      kernelComputeMortons<<<LAUNCH_THREADS(primitives->nPrs)>>>(primitives->nPrs, scene_bound, bbox->constAccessor(),
                                                                 mortons->accessor(), idx->accessor()));
  // sort the primitive pool[0, nPrs), bbox[nPrs - 1, nPrs*2 - 1) and idx by mortonCode[0, nPrs)
  // use zip iterator to sort the three arrays at the same time
  auto mortons_begin = thrust::device_ptr<uint64_t>(mortons->begin());
  auto mortons_end = thrust::device_ptr<uint64_t>(mortons->begin() + primitives->nPrs);
  cudaSafeCheck(
      thrust::sort_by_key(thrust::device, mortons_begin, mortons_end,
                          thrust::make_zip_iterator(bbox->begin() + primitives->nPrs - 1, idx->begin())));
  cudaSafeCheck(kernelComputeStructure<<<LAUNCH_THREADS(primitives->nPrs - 1)>>>(primitives->nPrs,
                                                                                 mortons->constAccessor(),
                                                                                 fa->accessor(),
                                                                                 lch->accessor(),
                                                                                 rch->accessor()));
  auto processed = make_unique<DeviceArray<int>>(primitives->nPrs - 1);
  cudaSafeCheck(thrust::fill(thrust::device, processed->begin(), processed->end(), 0));
  cudaSafeCheck(kernelUpdateBBox<<<LAUNCH_THREADS(primitives->nPrs)>>>(primitives->nPrs,
                                                                       fa->constAccessor(),
                                                                       lch->constAccessor(),
                                                                       rch->constAccessor(),
                                                                       bbox->accessor(),
                                                                       processed->accessor()));

  // print the info of the tree into a file
  std::vector<BBox> host_bbox;
  std::vector<int> host_idx;
  std::vector<int> host_fa;
  std::vector<int> host_lch;
  std::vector<int> host_rch;
  bbox->copyTo(host_bbox);
  idx->copyTo(host_idx);
  fa->copyTo(host_fa);
  lch->copyTo(host_lch);
  rch->copyTo(host_rch);
  std::ofstream fout = std::ofstream("bvh.txt");
  fout.close();
}
} // namespace CGL
