#include "pathtracer.h"

#include "scene/scene.h"
#include "scene/media.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <format>

#define approx(a, b) (fabs((a) - (b)) < EPS_F)

namespace CGL {
PathTracer::PathTracer() {
  path_pool.orig = std::make_unique<DeviceArray<double3>>(config.num_paths);
  path_pool.dir = std::make_unique<DeviceArray<double3>>(config.num_paths);
  path_pool.throughput = std::make_unique<DeviceArray<double3>>(config.num_paths);
  path_pool.indices = std::make_unique<DeviceArray<int>>(config.num_paths);
  interaction_pool.n = std::make_unique<DeviceArray<double3>>(config.num_paths);
  interaction_pool.t = std::make_unique<DeviceArray<double>>(config.num_paths);
  path_pool.scatter_type = std::make_unique<DeviceArray<int>>(config.num_paths);
  interaction_pool.primitive_id = std::make_unique<DeviceArray<int>>(config.num_paths);
  interaction_pool.mat_id = std::make_unique<DeviceArray<int>>(config.num_paths);
  interaction_pool.is_intersected = std::make_unique<DeviceArray<bool>>(config.num_paths);
  nee_path_pool.nee_pdf_light = std::make_unique<DeviceArray<double>>(config.num_paths);
  nee_path_pool.nee_pos = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_light_id = std::make_unique<DeviceArray<int>>(config.num_paths);
  nee_path_pool.nee_mat_pdf = std::make_unique<DeviceArray<double>>(config.num_paths);
  path_pool.mat_pdf = std::make_unique<DeviceArray<double>>(config.num_paths);
  path_pool.mat_eval_result = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_dir = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_t = std::make_unique<DeviceArray<double>>(config.num_paths);
  nee_path_pool.nee_light_normal = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_pos_cache = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_throughput = std::make_unique<DeviceArray<double3>>(config.num_paths);
  nee_path_pool.nee_medium_id = std::make_unique<DeviceArray<int8_t>>(config.num_paths);
  path_pool.depth = std::make_unique<DeviceArray<int>>(config.num_paths);
  path_pool.medium_id = std::make_unique<DeviceArray<int8_t>>(config.num_paths);
  path_pool.pdf_delta = std::make_unique<DeviceArray<double>>(config.num_paths);
  path_pool.pdf_ratio = std::make_unique<DeviceArray<double>>(config.num_paths);
  rngs = std::make_unique<DeviceArray<RandomGenerator>>(config.num_paths);
  work_pool = std::make_unique<MultipleWorkPool>(
      static_cast<int>(SurfaceInfo::NumSurfaceInfos) + static_cast<int>(PhaseFunctions::NumPhaseFunctions),
      config.num_paths);
  ray_keys = std::make_unique<DeviceArray<uint64_t>>(config.num_paths);
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

void PathTracer::set_frame_size(size_t width, size_t height) {
  cudaSafeCheck();
  sampleBuffer.resize(width, height);
}

void PathTracer::clear() {
  scene = nullptr;
  camera = nullptr;
  sampleBuffer.clear();
  sampleBuffer.resize(0, 0);
}

void PathTracer::write_to_framebuffer(ImageBuffer &framebuffer) {
  sampleBuffer.toColor(framebuffer);
}

static CUDA_CONSTANT PathTracer::Config kPathTracerConfig;
// put all the accessors of the device arrays of the path tracer into the constant memory
// so that we don't need to pass them as arguments to the kernels
// and we can also accelerate the access to the device arrays
static CUDA_CONSTANT DeviceArrayAccessor<double3> kOrig;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kDir;
static CUDA_CONSTANT DeviceArrayAccessor<int> kIndices;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kThroughput;
static CUDA_CONSTANT DeviceArrayAccessor<double> kMatPdf;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kMatEvalResult;
static CUDA_CONSTANT DeviceArrayAccessor<int> kWorkIndices;
static CUDA_CONSTANT DeviceArrayAccessor<RandomGenerator> kRngs;
static CUDA_CONSTANT DeviceArrayAccessor<int> kWorkSizes;
static CUDA_CONSTANT DeviceArrayAccessor<float3> kImage;
static CUDA_CONSTANT DeviceArrayAccessor<int> kPrimitiveId;
static CUDA_CONSTANT DeviceArrayAccessor<double> kT;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNormal;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNeeDir;
static CUDA_CONSTANT DeviceArrayAccessor<double> kNeeT;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNeeNormal;
static CUDA_CONSTANT LBVHAccessor kLBVH;
static CUDA_CONSTANT LightSamplerAccessor kLightSampler;
static CUDA_CONSTANT DeviceArrayAccessor<Surface> kSceneMatPool;
static CUDA_CONSTANT DeviceArrayAccessor<DiffuseData> kDiffuseData;
static CUDA_CONSTANT DeviceArrayAccessor<EmissiveData> kEmissiveData;
static CUDA_CONSTANT DeviceArrayAccessor<Shape> kShapes;
static CUDA_CONSTANT DeviceArrayAccessor<int> kScatterType;
static CUDA_CONSTANT DeviceArrayAccessor<int> kSurfId;
static CUDA_CONSTANT DeviceArrayAccessor<int> kNeeLightId;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNeeDestPos;
static CUDA_CONSTANT DeviceArrayAccessor<double> kNeeMatPdf;
static CUDA_CONSTANT DeviceArrayAccessor<bool> kIsIntersected;
static CUDA_CONSTANT DeviceArrayAccessor<uint64_t> kRayKeys;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNeePosCache;
static CUDA_CONSTANT DeviceArrayAccessor<int8_t> kMediumId;
static CUDA_CONSTANT DeviceArrayAccessor<int8_t> kNeeMediumId;
static CUDA_CONSTANT DeviceArrayAccessor<double3> kNeeThroughput;
static CUDA_CONSTANT DeviceArrayAccessor<int> kDepth;
static CUDA_CONSTANT DeviceArrayAccessor<Medium> kSceneMediumPool;
static CUDA_CONSTANT DeviceArrayAccessor<MediumInterfaceData> kMediumInterfaceData;
static CUDA_CONSTANT DeviceArrayAccessor<PhaseFunction> kPhaseFunctionPool;
static CUDA_CONSTANT DeviceArrayAccessor<double> kNeePdfDelta;
static CUDA_CONSTANT DeviceArrayAccessor<double> kNeePdfRatio;
static CUDA_CONSTANT DeviceArrayAccessor<double> kNeePdfLight;
static CUDA_CONSTANT DeviceArrayAccessor<HenyeyGreensteinData> kHenyeyGreensteinData;
static CUDA_CONSTANT DeviceArrayAccessor<IsotropicData> kIsotropicData;
static CUDA_CONSTANT DeviceArrayAccessor<double> kPdfDelta;
static CUDA_CONSTANT DeviceArrayAccessor<double> kPdfRatio;

static CUDA_CALLABLE CUDA_INLINE
void constructFrame(const double3 &normal, Mat3 &tbn) {
  double3 z = normal;
  double3 h = z;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h.x = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  normalize(z);
  double3 y = normalize(cross(h, z));
  double3 x = normalize(cross(z, y));

  tbn[0] = x;
  tbn[1] = y;
  tbn[2] = z;
}

static CUDA_DEVICE CUDA_FORCEINLINE double radians(double degrees) {
  return degrees * PI / 180.0;
}

static CUDA_GLOBAL void kernelRayGeneration(int num_paths,
                                            double hFov,
                                            double vFov,
                                            double3 cam_pos,
                                            Mat3 cam_to_world,
                                            int8_t cam_medium_id) {
  get_and_restrict_tid(tid, num_paths);
  kThroughput[tid] = make_double3(1.0, 1.0, 1.0);
  kWorkIndices[tid] = kIndices[tid] = tid;
  int pid = tid / kPathTracerConfig.spp;
  int px = pid % kPathTracerConfig.width;
  int py = pid / kPathTracerConfig.width;
  double2 uv = kRngs[tid].get_uv();
  double x = (px + uv.x) / kPathTracerConfig.width;
  double y = (py + uv.y) / kPathTracerConfig.height;
  auto sensor_x = 2 * tan(radians(hFov) / 2) * (x - 0.5);
  auto sensor_y = 2 * tan(radians(vFov) / 2) * (y - 0.5);
  double3 sensor_dir = make_double3(sensor_x, sensor_y, -1);
  double3 world_sensor_dir = normalize(cam_to_world * sensor_dir);
  kMatEvalResult[tid] = make_double3(1.0, 1.0, 1.0);
  kMatPdf[tid] = 1.0;
  kOrig[tid] = cam_pos;
  kDir[tid] = world_sensor_dir;
  kDepth[tid] = 0;
  kMediumId[tid] = cam_medium_id;
  kNeePdfDelta[tid] = 1.0;
  kNeePdfRatio[tid] = 1.0;
  kPdfDelta[tid] = 1.0;
  kPdfRatio[tid] = 1.0;
}

// cond can either be 0 or 1
static CUDA_DEVICE void warpAtomicInc(int *addr, bool cond) {
  // use ballopt to decide how many threads in this warp need to inc
  int tid = ktid(x);
  uint32_t mask = __ballot_sync(0xFFFFFFFF, cond);
  int idx_in_warp = tid & 0x1F;
  if (idx_in_warp == 0)
    atomicAdd(addr, __popc(mask));
}

static CUDA_DEVICE CUDA_FORCEINLINE void updateMedium(MediumInterfaceData interface,
                                                      const double3 &normal,
                                                      const double3 &nee_dir,
                                                      int8_t &cur_medium_id) {
  cur_medium_id = (dot(nee_dir, normal) > 0) ? interface.internal_id : interface.external_id;
}

static CUDA_DEVICE CUDA_FORCEINLINE double geometry(const double3 &pos, const double3 &normal, const double3 &orig) {
  double3 dif = orig - pos;
  double dist = length(dif);
  dif /= dist;
  return dot(dif, normal) / (dist * dist);
}

static CUDA_CALLABLE CUDA_FORCEINLINE int surfaceInfo2ScatterType(SurfaceInfo surface_info) {
  return static_cast<int>(surface_info) + static_cast<int>(PhaseFunctions::NumPhaseFunctions);
}

static CUDA_GLOBAL void kernelPathLogic(int num_paths) {
  get_and_restrict_tid(tid, num_paths);
  int orig_tid = tid;
  tid = kWorkIndices[tid];
  int pixel_id = kIndices[tid] / kPathTracerConfig.spp;
  if (kDepth[tid] > 0 && approx(kNeeT[tid], distance(kNeePosCache[tid], kNeeDestPos[tid]))) {
    double G = fabs(geometry(kNeeDestPos[tid], kNeeNormal[tid], kNeePosCache[tid]));
    double nee_scatter_pdf_in_area_measure = G * kNeeMatPdf[tid];
    double pdf_nee = kNeePdfLight[tid] * kNeePdfRatio[tid];
    double mis_weight = pdf_nee * pdf_nee / (nee_scatter_pdf_in_area_measure * nee_scatter_pdf_in_area_measure +
                                             pdf_nee * pdf_nee);
    double3 nee_throughput = G * kNeeThroughput[tid] / kNeePdfLight[tid];
    double3 contribution = nee_throughput * kEmissiveData[kNeeLightId[tid]].radiance * mis_weight;
    atomicAdd(&(kImage[pixel_id].x), static_cast<float>(contribution.x));
    atomicAdd(&(kImage[pixel_id].y), static_cast<float>(contribution.y));
    atomicAdd(&(kImage[pixel_id].z), static_cast<float>(contribution.z));
  }
  kOrig[tid] += kT[tid] * kDir[tid];
  kSurfId[tid] = kSceneMatPool[kPrimitiveId[tid]].surf_id;
  bool rr_terminate = kRngs[tid].get() < 0.05;
  if (!rr_terminate) kThroughput[tid] /= 0.95;
  kThroughput[tid] *= kMatEvalResult[tid] / kMatPdf[tid];
  bool is_terminated =
      kScatterType[orig_tid] == surfaceInfo2ScatterType(SurfaceInfo::EnvMap) ||
      kScatterType[orig_tid] == surfaceInfo2ScatterType(SurfaceInfo::Emissive) || rr_terminate;
  warpAtomicInc(&(kWorkSizes[surfaceInfo2ScatterType(SurfaceInfo::EnvMap)]), is_terminated);
  if (is_terminated) {
    if (kScatterType[orig_tid] == surfaceInfo2ScatterType(SurfaceInfo::Emissive) && !rr_terminate) {
      double3 contribution = kThroughput[tid] * kEmissiveData[kSurfId[tid]].radiance;
      if (kDepth[tid] > 0) {
        double pdf_nee =
            kLightSampler.probPrimitive(kPrimitiveId[tid]) *
            kShapes[kPrimitiveId[tid]].pdf(kOrig[tid]) * kPdfRatio[tid];
        double G = fabs(geometry(kOrig[tid], kNormal[tid], kNeePosCache[tid]));
        double pdf_dir = kPdfDelta[tid] * G * kMatPdf[tid];
        double mis_weight = pdf_dir * pdf_dir /
                            (pdf_nee * pdf_nee + pdf_dir * pdf_dir);
        contribution *= mis_weight;
      }
      atomicAdd(&(kImage[pixel_id].x), static_cast<float>(contribution.x));
      atomicAdd(&(kImage[pixel_id].y), static_cast<float>(contribution.y));
      atomicAdd(&(kImage[pixel_id].z), static_cast<float>(contribution.z));
    }
    kScatterType[orig_tid] = surfaceInfo2ScatterType(SurfaceInfo::EnvMap);
    return;
  }
  auto light_sample_record = kLightSampler.sample(kRngs[tid].get_uv());
  kNeeLightId[tid] = kSceneMatPool[light_sample_record.pr_idx].surf_id;
  double nee_shape_pdf;
  double3 light_normal;
  double3 light_pos = kShapes[light_sample_record.pr_idx].sample(kRngs[tid].get_uv(),
                                                                 &nee_shape_pdf,
                                                                 &light_normal);
  // the shape_pdf should be converted to the pdf with respect to the solid angle
  kNeePdfLight[tid] = light_sample_record.pdf * nee_shape_pdf;
  kNeeDir[tid] = normalize(light_pos - kOrig[tid]);
  kNeeDestPos[tid] = light_pos;
  kNeePosCache[tid] = kOrig[tid] + EPS_F * kNeeDir[tid];
  kNeeNormal[tid] = light_normal;
  kNeePdfRatio[tid] = kNeePdfDelta[tid] = kPdfDelta[tid] = kPdfRatio[tid] = 1.0;
  kDepth[tid]++;
  kNeeThroughput[tid] = kThroughput[tid];
  atomicAdd(&(kWorkSizes[kScatterType[orig_tid]]), 1);
}

static CUDA_GLOBAL void kernelExtendPath(int num_paths) {
  get_and_restrict_tid(tid, num_paths);
  int orig_tid = tid;
  tid = kWorkIndices[tid];
  CUDA_SHARED double3 shared_orig[kThreadBlockSize];
  CUDA_SHARED double3 shared_dir[kThreadBlockSize];
  CUDA_SHARED double shared_t[kThreadBlockSize];
  CUDA_SHARED double3 shared_normal[kThreadBlockSize];
  CUDA_SHARED int shared_primitive_id[kThreadBlockSize];
  CUDA_SHARED int8_t shared_medium_id[kThreadBlockSize];
  double3 &orig = shared_orig[threadIdx.x];
  double3 &dir = shared_dir[threadIdx.x];
  double &t_hit = shared_t[threadIdx.x];
  double3 &normal = shared_normal[threadIdx.x];
  int &primitive_id = shared_primitive_id[threadIdx.x];
  int8_t &medium_id = shared_medium_id[threadIdx.x];
  orig = kOrig[tid];
  dir = kDir[tid];
  medium_id = kMediumId[tid];
  t_hit = 1e9;
  normal = make_double3(0.0, 0.0, 0.0);
  primitive_id = -1;
  double accumulated_t = 0.0;
  while (true) {
    bool scattered = false;
    kIsIntersected[tid] = kLBVH.intersect(orig, dir, normal, t_hit, primitive_id);
    if (!kIsIntersected[tid]) {
      // reserved for env map
      kScatterType[orig_tid] = surfaceInfo2ScatterType(SurfaceInfo::EnvMap);
      break;
    }
    auto surface = kSceneMatPool[primitive_id];
    if (medium_id >= 0) {
      int channel = floor(kRngs[tid].get() * 3);
      double3 sigma_m = kSceneMediumPool[medium_id].getMajorant();
//      int null_collision_cnt = 0;
      double3 T = make_constant(1.0);
      double3 trans_pdf_delta = make_constant(1.0);
      double3 trans_pdf_ratio = make_constant(1.0);
      double sigma_m_channel = channel == 0 ? sigma_m.x : (channel == 1 ? sigma_m.y : sigma_m.z);
      while (true) {
        double u = kRngs[tid].get();
        if (sigma_m_channel <= 0.0) break;
        double t = -log(1.0 - u) / sigma_m_channel;
        if (t >= t_hit) {
          trans_pdf_delta *= exp(-t_hit * sigma_m);
          trans_pdf_ratio *= exp(-t_hit * sigma_m);
          T *= exp(-t_hit * sigma_m);
          orig += (t_hit + EPS_F) * dir;
          accumulated_t += t_hit + EPS_F;
          if (surface.surface_info == SurfaceInfo::MediumInterface)
            updateMedium(kMediumInterfaceData[surface.surf_id], normal, dir, medium_id);
          else {
            kScatterType[orig_tid] = surfaceInfo2ScatterType(surface.surface_info);
            scattered = true;
            break;
          }
        } else {
          orig += t * dir;
          accumulated_t += t;
          double3 sigma_s = kSceneMediumPool[medium_id].getScattering(orig);
          double3 sigma_t = kSceneMediumPool[medium_id].getAbsorption(orig) + sigma_s;
          double sigma_t_channel = channel == 0 ? sigma_t.x : (channel == 1 ? sigma_t.y : sigma_t.z);
          if (kRngs[tid].get() < sigma_t_channel / sigma_m_channel) {
            T *= exp(-sigma_m * t) / maxComponent(sigma_m);
            trans_pdf_delta *= exp(-sigma_m * t) * sigma_t / maxComponent(sigma_m);
            kScatterType[orig_tid] = kPhaseFunctionPool[medium_id].pf;
            scattered = true;
            break;
          } else { // null collision, go ahead
            double3 sigma_n = sigma_m - sigma_t;
            T *= exp(-sigma_m * t) * sigma_n / maxComponent(sigma_m);
            trans_pdf_delta *= exp(-sigma_m * t) * sigma_n / maxComponent(sigma_m);
            trans_pdf_ratio *= exp(-sigma_m * t) * sigma_m / maxComponent(sigma_m);
            if (maxComponent(T) <= 0.0) {
              scattered = true;
              kScatterType[orig_tid] = surfaceInfo2ScatterType(EnvMap);
              break;
            }
//            null_collision_cnt++;
//            if (null_collision_cnt >= kPathTracerConfig.maxNullCollisions) {
//              kScatterType[orig_tid] = surfaceInfo2ScatterType(kPhaseFunctionPool[medium_id].pf);
//              break;
//            }
          }
        }
      }
      kThroughput[tid] *= T / avg(trans_pdf_delta);
      kPdfDelta[tid] *= avg(trans_pdf_delta);
      kPdfRatio[tid] *= avg(trans_pdf_ratio);
      if (scattered) break;
    } else {
      if (surface.surface_info == SurfaceInfo::MediumInterface) {
        accumulated_t += t_hit + EPS_F;
        orig += (t_hit + EPS_F) * dir;
        updateMedium(kMediumInterfaceData[surface.surf_id], normal, dir, medium_id);
      } else {
        accumulated_t += t_hit;
        kScatterType[orig_tid] = surfaceInfo2ScatterType(surface.surface_info);
        break;
      }
    }
  }
  kT[tid] = accumulated_t;
  kNormal[tid] = normal;
  kPrimitiveId[tid] = primitive_id;
}

static CUDA_DEVICE double3 deltaTracking(double t_hit,
                                         const Surface &surface,
                                         double3 &orig,
                                         const double3 &dir,
                                         int cur_medium_id,
                                         RandomGenerator &rng,
                                         double &pdf_delta,
                                         double &pdf_ratio) {
  int channel = floor(rng.get() * 3);
  double3 sigma_m = kSceneMediumPool[cur_medium_id].getMajorant();
  int null_collision_cnt = 0;
  double3 T = make_double3(1.0, 1.0, 1.0);
  double3 trans_pdf_delta = make_double3(1.0, 1.0, 1.0),
      trans_pdf_ratio = make_double3(1.0, 1.0, 1.0);
  while (true) {
    double sigma_m_channel = channel == 0 ? sigma_m.x : (channel == 1 ? sigma_m.y : sigma_m.z);
    if (sigma_m_channel <= 0.0) break;
    double u = rng.get();
    double t = -log(1.0 - u) / sigma_m_channel;
    if (t >= t_hit) {
      if (surface.surface_info ==
          SurfaceInfo::MediumInterface) {
        T *= exp(-sigma_m * t_hit);
        trans_pdf_delta *= exp(-sigma_m * t_hit);
        trans_pdf_ratio *= exp(-sigma_m * t_hit);
      }
      break;
    } else {
      double3 sigma_s = kSceneMediumPool[cur_medium_id].getScattering(orig);
      double3 sigma_t = kSceneMediumPool[cur_medium_id].getAbsorption(orig) + sigma_s;
      orig += t * dir;
      double3 sigma_n = sigma_m - sigma_t;
      T *= exp(-sigma_m * t) * sigma_n / maxComponent(sigma_m);
      trans_pdf_delta *= exp(-sigma_m * t) * sigma_n / maxComponent(sigma_m);
      trans_pdf_ratio *= exp(-sigma_m * t) * sigma_m / maxComponent(sigma_m);
      null_collision_cnt++;
      if (null_collision_cnt >= kPathTracerConfig.maxNullCollisions) break;
      if (maxComponent(T) <= 0.0) return {};
    }
    t_hit -= t;
  }
  pdf_delta *= avg(trans_pdf_delta);
  pdf_ratio *= avg(trans_pdf_ratio);
  return T / avg(trans_pdf_ratio);
}

static CUDA_GLOBAL void kernelShadowExtend(int num_paths) {
  get_and_restrict_tid(tid, num_paths);
  tid = kWorkIndices[tid];
  CUDA_SHARED double3 shared_orig[kThreadBlockSize];
  CUDA_SHARED double3 shared_nee_dir[kThreadBlockSize];
  CUDA_SHARED double shared_nee_t[kThreadBlockSize];
  CUDA_SHARED double3 shared_normal[kThreadBlockSize];
  CUDA_SHARED int shared_primitive_id[kThreadBlockSize];
  CUDA_SHARED int8_t shared_medium_id[kThreadBlockSize];
  double3 &orig = shared_orig[threadIdx.x];
  double3 &nee_dir = shared_nee_dir[threadIdx.x];
  double &nee_t_hit = shared_nee_t[threadIdx.x];
  double3 &normal = shared_normal[threadIdx.x];
  int &primitive_id = shared_primitive_id[threadIdx.x];
  int8_t &medium_id = shared_medium_id[threadIdx.x];
  medium_id = kMediumId[tid];
  orig = kNeePosCache[tid];
  nee_dir = kNeeDir[tid];
  nee_t_hit = 1e9;
  double accumulated_t = 0.0;
  while (true) {
    bool is_intersected = kLBVH.intersect(orig, nee_dir, normal, nee_t_hit, primitive_id);
    auto surface = kSceneMatPool[primitive_id];
    if (is_intersected) {
      if (medium_id >= 0)
        kNeeThroughput[tid] *= deltaTracking(nee_t_hit, surface, orig, nee_dir, medium_id, kRngs[tid],
                                             kNeePdfDelta[tid], kNeePdfRatio[tid]);
      orig += nee_t_hit * nee_dir;
      accumulated_t += nee_t_hit;
      if (surface.surface_info != SurfaceInfo::MediumInterface) break;
      updateMedium(kMediumInterfaceData[surface.surf_id], normal, nee_dir, medium_id);
    } else {
      // reserved for env map
    }
  }
  kNeeT[tid] = accumulated_t;
}

static CUDA_GLOBAL void kernelNewPath(int new_path_num,
                                      int total_samples,
                                      int new_path_start,
                                      Mat3 c2w,
                                      double3 camera_pos,
                                      double hFov,
                                      double vFov,
                                      int8_t cam_medium_id
) {
  get_and_restrict_tid(tid, new_path_num);
  int orig_tid = tid;
  tid = kWorkIndices[new_path_start + tid];
  kIndices[tid] = total_samples + orig_tid;
  kThroughput[tid] = make_double3(1.0, 1.0, 1.0);
  int pid = kIndices[tid] / kPathTracerConfig.spp;
  int px = pid % kPathTracerConfig.width;
  int py = pid / kPathTracerConfig.width;
  double2 uv = kRngs[tid].get_uv();
  double x = (px + uv.x) / kPathTracerConfig.width;
  double y = (py + uv.y) / kPathTracerConfig.height;
  auto sensor_x = 2 * tan(radians(hFov) / 2) * (x - 0.5);
  auto sensor_y = 2 * tan(radians(vFov) / 2) * (y - 0.5);
  double3 sensor_dir = make_double3(sensor_x, sensor_y, -1);
  double3 world_sensor_dir = normalize(c2w * sensor_dir);
  kMatEvalResult[tid] = make_double3(1.0, 1.0, 1.0);
  kMatPdf[tid] = 1.0;
  kOrig[tid] = camera_pos;
  kDir[tid] = world_sensor_dir;
  kDepth[tid] = 0;
  kMediumId[tid] = cam_medium_id;
  kNeePdfDelta[tid] = 1.0;
  kNeePdfRatio[tid] = 1.0;
  kPdfDelta[tid] = 1.0;
  kPdfRatio[tid] = 1.0;
}

static CUDA_GLOBAL void kernelShadeDiffuse(int num_paths, int diffuse_start) {
  get_and_restrict_tid(tid, num_paths);
  int idx = diffuse_start + tid;
  idx = kWorkIndices[idx];
  const auto &albedo = kDiffuseData[kSurfId[idx]].albedo;
  const auto &normal = kNormal[idx];
  const auto &local_wi = cosHemisphereSample(kRngs[tid].get_uv());
  Mat3 tbn;
  constructFrame(normal, tbn);
  auto world_to_local = inv(tbn);
  auto world_wi = tbn * local_wi;
  auto local_nee_dir = world_to_local * kNeeDir[idx];
  // we must split pdf and other parts of the estimator1
  // because we will need the pdf for mis weight calculation
  kMatPdf[idx] = cosHemispherePdf(local_wi);
  kNeeMatPdf[idx] = cosHemispherePdf(local_nee_dir);
  // note: albedo / PI is the BRDF, local_wi.z is the cosine term
  kMatEvalResult[idx] = albedo / PI * local_wi.z;
  kNeeThroughput[idx] *= albedo / PI * local_nee_dir.z;
  kDir[idx] = world_wi;
  kOrig[idx] += EPS_F * world_wi;
}

static CUDA_GLOBAL void kernelShadeHenyeyGreenstein(int num_paths, int hg_start) {
  get_and_restrict_tid(tid, num_paths);
  int idx = hg_start + tid;
  idx = kWorkIndices[idx];
  const auto &dir = kDir[idx];
  Mat3 tbn;
  constructFrame(dir, tbn);
  auto world_to_local = inv(tbn);
  auto local_nee_dir = world_to_local * kNeeDir[idx];
  double u = kRngs[tid].get();
  double g = kHenyeyGreensteinData[kPhaseFunctionPool[kMediumId[idx]].pf_id].g;
  double term = (1 - g * g) / (1 + g - 2 * g * u);
  double cos_theta = (term * term - 1 - g) / (2 * g);
  double sin_theta = sqrt(fmax(1 - cos_theta * cos_theta, 0.0));
  double phi = 2 * PI * kRngs[tid].get();
  double3 local_wi = make_double3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
  auto world_wi = tbn * local_wi;
  kMatPdf[idx] = (1.0 - g * g) / (u * sqrt(u)) * M_PI_4;
  kNeeMatPdf[idx] = M_PI_4 * (1.0 - g * g) / ((1 + g * g + 2 * g * cos_theta) * sqrtf(1 + g * g + 2 * g * cos_theta));
  double3 sigma_s = kSceneMediumPool[kMediumId[idx]].getScattering(kOrig[idx]);
  kMatEvalResult[idx] = sigma_s * make_constant(M_PI_4 * (1.0 - g * g) / ((1 + g * g + 2 * g * dot(world_wi, dir)) *
                                                                          sqrtf(1.0 + g * g +
                                                                                2 * g * dot(world_wi, dir))));
  kNeeThroughput[idx] *= sigma_s * M_PI_4 * (1.0 - g * g) /
                         ((1.0 + g * g + 2 * g * dot(kNeeDir[tid], dir)) *
                          sqrtf(1.0 + g * g + 2 * g * dot(kNeeDir[tid], dir)));
  kDir[idx] = world_wi;
}

static CUDA_GLOBAL void kernelShadeIsotropic(int num_paths, int iso_start) {
  get_and_restrict_tid(tid, num_paths);
  int idx = iso_start + tid;
  idx = kWorkIndices[idx];
  const auto &dir = kDir[idx];
  Mat3 tbn;
  constructFrame(dir, tbn);
  double3 local_wi = uniformHemisphereSample(kRngs[tid].get_uv());
  auto world_wi = tbn * local_wi;
  kMatPdf[idx] = uniformHemispherePdf();
  kNeeMatPdf[idx] = uniformHemispherePdf();
  double3 sigma_s = kSceneMediumPool[kMediumId[idx]].getScattering(kOrig[idx]);
  double3 albedo = kIsotropicData[kPhaseFunctionPool[kMediumId[idx]].pf_id].albedo;
  kMatEvalResult[idx] = sigma_s * albedo * M_1_4PI;
  kNeeThroughput[idx] *= sigma_s * albedo * M_1_4PI;
  kDir[idx] = world_wi;
}

void PathTracer::shadeScatter(const std::vector<int> &work_sizes,
                              int total_samples,
                              int num_new_paths) const {
  constexpr int num_work_types =
      static_cast<int>(SurfaceInfo::NumSurfaceInfos) + static_cast<int>(PhaseFunctions::NumPhaseFunctions);
  int start[num_work_types];
  start[0] = 0;
  for (int i = 1; i < num_work_types; i++)
    start[i] = start[i - 1] + work_sizes[i - 1];
  if (num_new_paths > 0) {
    std::cout << std::format("Generate {} new paths\n", num_new_paths);
    cudaSafeCheck(kernelNewPath<<<LAUNCH_THREADS(num_new_paths)>>>(
        num_new_paths,
        total_samples,
        start[surfaceInfo2ScatterType(SurfaceInfo::EnvMap)],
        to_Mat3(camera->c2w),
        camera->pos,
        camera->hFov,
        camera->vFov,
        camera->medium_id
    ));
  }
  if (work_sizes[surfaceInfo2ScatterType(SurfaceInfo::Diffuse)] > 0) {
    std::cout
        << std::format("{} paths intersected with diffuse material\n",
                       work_sizes[surfaceInfo2ScatterType(SurfaceInfo::Diffuse)]);
    cudaSafeCheck(kernelShadeDiffuse<<<LAUNCH_THREADS(work_sizes[surfaceInfo2ScatterType(SurfaceInfo::Diffuse)])>>>(
        work_sizes[surfaceInfo2ScatterType(SurfaceInfo::Diffuse)],
        start[surfaceInfo2ScatterType(SurfaceInfo::Diffuse)]
    ));
  }
  if (work_sizes[PhaseFunctions::Isotropic] > 0) {
    std::cout << std::format("{} paths scatter with isotropic phase function\n",
                             work_sizes[PhaseFunctions::Isotropic]);
    cudaSafeCheck(kernelShadeIsotropic<<<LAUNCH_THREADS(work_sizes[PhaseFunctions::Isotropic])>>>(
        work_sizes[PhaseFunctions::Isotropic],
        start[PhaseFunctions::Isotropic]
    ));
  }
}

static CUDA_GLOBAL void kernelComputeRayKeys(int num_paths, BBox scene_bound) {
  get_and_restrict_tid(tid, num_paths);
  int orig_tid = tid;
  tid = kWorkIndices[tid];
  double transformed_dir_x = 0.5 * kDir[tid].x + 0.5;
  double transformed_dir_y = 0.5 * kDir[tid].y + 0.5;
  double transformed_dir_z = 0.5 * kDir[tid].z + 0.5;
  double transformed_orig_x = (kOrig[tid].x - scene_bound.lo.x) / (scene_bound.hi.x - scene_bound.lo.x);
  double transformed_orig_y = (kOrig[tid].y - scene_bound.lo.y) / (scene_bound.hi.y - scene_bound.lo.y);
  double transformed_orig_z = (kOrig[tid].z - scene_bound.lo.z) / (scene_bound.hi.z - scene_bound.lo.z);
  auto dir_key_x = static_cast<uint32_t>(transformed_dir_x * 1024.0);
  auto dir_key_y = static_cast<uint32_t>(transformed_dir_y * 1024.0);
  auto dir_key_z = static_cast<uint32_t>(transformed_dir_z * 1024.0);
  auto orig_key_x = static_cast<uint32_t>(transformed_orig_x * 1024.0);
  auto orig_key_y = static_cast<uint32_t>(transformed_orig_y * 1024.0);
  auto orig_key_z = static_cast<uint32_t>(transformed_orig_z * 1024.0);
  uint32_t key_x = (orig_key_x << 10) | dir_key_x;
  uint32_t key_y = (orig_key_y << 10) | dir_key_y;
  uint32_t key_z = (orig_key_z << 10) | dir_key_z;
  kRayKeys[orig_tid] = expandBits21(key_x) | (expandBits21(key_y) << 1) | (expandBits21(key_z) << 2);
}

// by default, at this stage, the material_pool is sorted by material type
// and new paths are all at the beginning of the pool
// so we can directly reuse this property to distinguish new paths and old paths
// for all paths we extend, for old paths we cast shadow rays
void PathTracer::pathExtend(int new_paths, int current_live_paths, const BBox &scene_bound) const {
  int num_old_paths = current_live_paths - new_paths;
  assert(current_live_paths > 0);
  thrust::device_ptr<uint64_t> ray_keys_ptr(ray_keys->begin());
  thrust::device_ptr<int> work_indices_ptr(work_pool->indices->begin());
  if (num_old_paths > 0) {
    kernelComputeRayKeys<<<LAUNCH_THREADS(num_old_paths)>>>(num_old_paths, scene_bound);
    thrust::sort_by_key(ray_keys_ptr, ray_keys_ptr + num_old_paths, work_indices_ptr);
    kernelShadowExtend<<<LAUNCH_THREADS(num_old_paths)>>>(num_old_paths);
  }
  kernelComputeRayKeys<<<LAUNCH_THREADS(current_live_paths)>>>(current_live_paths, scene_bound);
  thrust::sort_by_key(ray_keys_ptr, ray_keys_ptr + current_live_paths, work_indices_ptr);
  kernelExtendPath<<<LAUNCH_THREADS(current_live_paths)>>>(current_live_paths);
}

static CUDA_GLOBAL void kernelAverageImage(int num_pixels, DeviceArrayAccessor<float3> image) {
  get_and_restrict_tid(tid, num_pixels);
  image[tid] /= static_cast<float>(kPathTracerConfig.spp);
}

static CUDA_GLOBAL void kernelSetRng(int num_paths,
                                     DeviceArrayAccessor<RandomGenerator> rngs,
                                     DeviceArrayAccessor<uint64_t> seeds) {
  get_and_restrict_tid(tid, num_paths);
  rngs[tid] = RandomGenerator(seeds[tid]);
}

void PathTracer::initRng() const {
  std::vector<uint64_t> host_seeds(config.num_paths);
  for (int i = 0; i < config.num_paths; i++) {
    chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
    chrono::nanoseconds ns = chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch());
    host_seeds[i] = ns.count();
  }
  auto device_seeds = std::make_unique<DeviceArray<uint64_t>>
      (host_seeds);
  cudaSafeCheck(
      kernelSetRng<<<LAUNCH_THREADS(config.num_paths)>>>(
          config.num_paths,
          rngs->accessor(),
          device_seeds->accessor()
      ));
}

#define COPY_TO_CONSTANT(device_array_name, constant_name) \
do {                                                       \
     auto device_array_accessor = (device_array_name)->accessor(); \
     cudaMemcpyToSymbol(constant_name, &device_array_accessor, sizeof(device_array_accessor)); \
} while (0)

void PathTracer::copyToConstantMemory() const {
  COPY_TO_CONSTANT(path_pool.orig, kOrig);
  COPY_TO_CONSTANT(path_pool.dir, kDir);
  COPY_TO_CONSTANT(path_pool.indices, kIndices);
  COPY_TO_CONSTANT(path_pool.throughput, kThroughput);
  COPY_TO_CONSTANT(path_pool.mat_pdf, kMatPdf);
  COPY_TO_CONSTANT(path_pool.mat_eval_result, kMatEvalResult);
  COPY_TO_CONSTANT(work_pool->indices, kWorkIndices);
  COPY_TO_CONSTANT(rngs, kRngs);
  COPY_TO_CONSTANT(work_pool->work_sizes, kWorkSizes);
  COPY_TO_CONSTANT(sampleBuffer.data, kImage);
  COPY_TO_CONSTANT(interaction_pool.primitive_id, kPrimitiveId);
  COPY_TO_CONSTANT(interaction_pool.t, kT);
  COPY_TO_CONSTANT(interaction_pool.n, kNormal);
  COPY_TO_CONSTANT(nee_path_pool.nee_dir, kNeeDir);
  COPY_TO_CONSTANT(nee_path_pool.nee_t, kNeeT);
  COPY_TO_CONSTANT(nee_path_pool.nee_light_normal, kNeeNormal);
  COPY_TO_CONSTANT(nee_path_pool.nee_light_id, kNeeLightId);
  COPY_TO_CONSTANT(nee_path_pool.nee_pos, kNeeDestPos);
  COPY_TO_CONSTANT(nee_path_pool.nee_pdf_light, kNeePdfLight);
  COPY_TO_CONSTANT(nee_path_pool.nee_mat_pdf, kNeeMatPdf);
  COPY_TO_CONSTANT(path_pool.scatter_type, kScatterType);
  COPY_TO_CONSTANT(interaction_pool.mat_id, kSurfId);
  COPY_TO_CONSTANT(interaction_pool.is_intersected, kIsIntersected);
  COPY_TO_CONSTANT(ray_keys, kRayKeys);
  COPY_TO_CONSTANT(lbvh->primitives->materials, kSceneMatPool);
  COPY_TO_CONSTANT(scene->diffuse_data, kDiffuseData);
  COPY_TO_CONSTANT(scene->emissive_data, kEmissiveData);
  COPY_TO_CONSTANT(lbvh->primitives->shapes, kShapes);
  COPY_TO_CONSTANT(interaction_pool.is_intersected, kIsIntersected);
  COPY_TO_CONSTANT(nee_path_pool.nee_pos_cache, kNeePosCache);
  COPY_TO_CONSTANT(path_pool.depth, kDepth);
  COPY_TO_CONSTANT(nee_path_pool.nee_throughput, kNeeThroughput);
  COPY_TO_CONSTANT(nee_path_pool.nee_medium_id, kNeeMediumId);
  COPY_TO_CONSTANT(path_pool.medium_id, kMediumId);
  COPY_TO_CONSTANT(path_pool.pdf_delta, kNeePdfDelta);
  COPY_TO_CONSTANT(path_pool.pdf_ratio, kNeePdfRatio);
  COPY_TO_CONSTANT(path_pool.pdf_ratio, kPdfRatio);
  COPY_TO_CONSTANT(path_pool.pdf_delta, kPdfDelta);
  COPY_TO_CONSTANT(scene->media, kSceneMediumPool);
  COPY_TO_CONSTANT(scene->phase_functions, kPhaseFunctionPool);
  COPY_TO_CONSTANT(scene->isotropic_data, kIsotropicData);
//  COPY_TO_CONSTANT(scene->medium_interface_data, kMediumInterfaceData);
  auto lbvh_accessor = lbvh->accessor();
  cudaMemcpyToSymbol(kLBVH, &lbvh_accessor, sizeof(lbvh_accessor));
  auto light_sampler_accessor = scene->light_sampler->accessor();
  cudaMemcpyToSymbol(kLightSampler, &light_sampler_accessor, sizeof(light_sampler_accessor));
}

void PathTracer::raytrace() {
  std::vector<Medium> host_media(1);
  host_media[0].getHomogeneousMedium() = HomogeneousMedium(make_double3(0.1, 0.1, 0.1), make_double3(0.1, 0.7, 0.1));
  scene->media = std::make_unique<DeviceArray<Medium>>(host_media);
  std::vector<PhaseFunction> host_phase_functions(1);
  host_phase_functions[0].pf = PhaseFunctions::Isotropic;
  host_phase_functions[0].pf_id = 0;
  scene->phase_functions = std::make_unique<DeviceArray<PhaseFunction>>(host_phase_functions);
  scene->phase_functions->copyFrom(host_phase_functions);
  std::vector<IsotropicData> host_isotropic_data(1);
  host_isotropic_data[0].albedo = make_double3(1.0, 1.0, 1.0);
  scene->isotropic_data = std::make_unique<DeviceArray<IsotropicData>>(host_isotropic_data);
  camera->medium_id = 0;
  stream = std::make_unique<CudaStream>();
  int rest_samples = static_cast<int>(targetSamplePaths());
  int current_live_paths = min(rest_samples, config.num_paths);
  int total_samples = 0;
  config.width = static_cast<int>(sampleBuffer.w);
  config.height = static_cast<int>(sampleBuffer.h);
  config.num_paths = std::min(config.num_paths, rest_samples);
  cudaMemcpyToSymbol(kPathTracerConfig, &config, sizeof(Config));
  std::vector<int> work_sizes(work_pool->nWorks);
  std::cout << std::format("Config: num_paths: {}, spp: {}, width: {}, height: {}\n",
                           config.num_paths, config.spp, config.width, config.height);
  initRng();
  // print camera->c2w
  copyToConstantMemory();
  cudaSafeCheck(kernelRayGeneration<<<LAUNCH_THREADS(current_live_paths)>>>(
      current_live_paths,
      camera->hFov,
      camera->vFov,
      camera->pos,
      to_Mat3(camera->c2w),
      camera->medium_id
  ));
  std::cout << std::format("Start with {} paths\n", current_live_paths);
  cudaSafeCheck(kernelExtendPath<<<LAUNCH_THREADS(current_live_paths)>>>(
      current_live_paths));
  total_samples += current_live_paths;
  int num_new_paths{}, old_live_paths{};
  while (true) {
    printf("current live paths: %d\n", current_live_paths);
    cudaMemset(work_pool->work_sizes->data(), 0, sizeof(int) * work_pool->nWorks);
    cudaSafeCheck(kernelPathLogic<<<LAUNCH_THREADS(current_live_paths)>>>(current_live_paths));
    cudaWait();
    work_pool->getWorkSize(work_sizes);
    int num_terminated_paths = work_sizes[surfaceInfo2ScatterType(SurfaceInfo::EnvMap)];
    rest_samples -= num_terminated_paths;
    old_live_paths = current_live_paths - num_terminated_paths;
    std::cout << std::format("{} paths terminated, {} samples left\n", num_terminated_paths, rest_samples);
    if (rest_samples == 0) break;
    auto mat_begin = thrust::device_ptr<int>(path_pool.scatter_type->begin());
    auto mat_end = thrust::device_ptr<int>(path_pool.scatter_type->begin() + current_live_paths);
    auto idx_begin = thrust::device_ptr<int>(work_pool->indices->begin());
    cudaSafeCheck(thrust::sort_by_key(mat_begin, mat_end, idx_begin));
    current_live_paths = std::min(rest_samples, config.num_paths);
    num_new_paths = current_live_paths - old_live_paths;
    // put the terminated paths at the end of the pool, and we only care about the first current_live_paths paths
    shadeScatter(work_sizes, total_samples, num_new_paths);
    total_samples += num_new_paths;
    pathExtend(num_new_paths, current_live_paths, lbvh->scene_bound);
  }
  cudaSafeCheck(kernelAverageImage<<<LAUNCH_THREADS(sampleBuffer.w * sampleBuffer.h)>>>(
      static_cast<int>(sampleBuffer.w * sampleBuffer.h),
      sampleBuffer.data->accessor()));
}

void PathTracer::autofocus(Vector2D loc) {
  // Ray r = camera->generate_ray(loc.x / sampleBuffer.w, loc.y / sampleBuffer.h);
  // SurfaceInteraction isect;

  // bvh->intersect(r, &isect);

  // camera->focalDistance = isect.t;
}

}
