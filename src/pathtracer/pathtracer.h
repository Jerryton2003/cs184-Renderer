#ifndef CGL_PATHTRACER_H
#define CGL_PATHTRACER_H

#include "CGL/timer.h"

#include "scene/bvh.h"
#include "pathtracer/sampler.h"

#include "application/renderer.h"

#include "scene/scene.h"

using CGL::Scene;

namespace CGL {

// this will devide the threads into multiple work pools
// nWorks is the number of works, nThreads is the number of threads
struct MultipleWorkPool {
  int nWorks, nThreads;
  std::unique_ptr<DeviceArray < int>> indices;
  std::unique_ptr<DeviceArray < int>> work_sizes;

  MultipleWorkPool(int nWorks_, int nThreads_)
      : nWorks(nWorks_), nThreads(nThreads_) {
    indices = std::make_unique<DeviceArray < int>>
    (nThreads);
    work_sizes = std::make_unique<DeviceArray < int>>
    (nWorks);
  }

  void getWorkSize(std::vector<int> &host_work_sizes) const {
    work_sizes->copyTo(host_work_sizes);
  }
};

class PathTracer {
 public:
  PathTracer();

  ~PathTracer() = default;

  /**
   * Sets the pathtracer's frame size. If in a running state (VISUALIZE,
   * RENDERING, or DONE), transitions to READY b/c a changing window size
   * would invalidate the output. If in INIT and configuration is done,
   * transitions to READY.
   * \param width width of the frame
   * \param height height of the frame
   */
  void set_frame_size(size_t width, size_t height);

  void write_to_framebuffer(ImageBuffer &framebuffer);

  /**
   * If the pathtracer is in READY, delete all internal data, transition to INIT.
   */
  void clear();

  void autofocus(Vector2D loc);

  void shadeScatter(const std::vector<int> &host_work_sizes, int total_samples, int num_new_paths) const;

  void pathExtend(int new_paths, int current_live_paths, const BBox &scene_bound) const;

  /**
   * Trace a camera ray given by the pixel coordinate.
   */
  void raytrace();

  // Integrator sampling settings //
  struct Config {
    int num_paths = 1 << 20; // size of path pool
    int spp = 256;
    int maxNullCollisions = 1024;
    int width{};
    int height{};
    double tm_gamma = 2.2; ///< gamma
  };
//  CUDA_CONSTANT Config config;
  size_t max_ray_depth{}; ///< maximum allowed ray depth (applies to all rays)
  size_t ns_aa{}; ///< number of camera rays in one pixel (along one axis)
  size_t ns_area_light{}; ///< number samples per area light source
  size_t ns_diff{}; ///< number of samples - diffuse surfaces
  size_t ns_glsy{}; ///< number of samples - glossy surfaces
  size_t ns_refr{}; ///< number of samples - refractive surfaces

  size_t samplesPerBatch{};
  double maxTolerance{};
  bool direct_hemisphere_sample{};

  void initRng() const;

//  [[nodiscard]] size_t targetSamplePaths() const {
//    return config.spp * sampleBuffer.w * sampleBuffer.h;
//  }

  ///< true if sampling uniformly from hemisphere for direct lighting. Otherwise, light sample
  // SOA
  struct PathPool {
    std::unique_ptr<DeviceArray<double3>> orig{};
    std::unique_ptr<DeviceArray<double3>> dir{};
    std::unique_ptr<DeviceArray<double3>> throughput{};
    std::unique_ptr<DeviceArray<int>> indices{};
    std::unique_ptr<DeviceArray<int>> depth{};
    std::unique_ptr<DeviceArray<double>> mat_pdf{};
    std::unique_ptr<DeviceArray<double3>> mat_eval_result{};
    std::unique_ptr<DeviceArray<int8_t>> medium_id{};
    std::unique_ptr<DeviceArray<int>> scatter_type{};
    std::unique_ptr<DeviceArray<double>> pdf_delta{};
    std::unique_ptr<DeviceArray<double>> pdf_ratio{};
  } path_pool;
  struct NeePathPool {
    std::unique_ptr<DeviceArray<double3>> nee_light_normal{};
    std::unique_ptr<DeviceArray<double3>> nee_dir{};
    std::unique_ptr<DeviceArray<double>> nee_t{};
    std::unique_ptr<DeviceArray<double>> nee_pdf_light{};
    std::unique_ptr<DeviceArray<double>> nee_pdf_delta{};
    std::unique_ptr<DeviceArray<double>> nee_pdf_ratio{};
    std::unique_ptr<DeviceArray<double>> nee_mat_pdf{};
    std::unique_ptr<DeviceArray<double3>> nee_pos{};
    std::unique_ptr<DeviceArray<int>> nee_light_id{};
    std::unique_ptr<DeviceArray<double3>> nee_pos_cache{};
    std::unique_ptr<DeviceArray<int8_t>> nee_medium_id{};
    std::unique_ptr<DeviceArray<double3>> nee_throughput{};
    std::unique_ptr<DeviceArray<SurfaceInfo>> nee_scatter_type{};
  } nee_path_pool;
  struct InteractionPool {
    std::unique_ptr<DeviceArray<double3>> n{};
    std::unique_ptr<DeviceArray<double>> t{};
    std::unique_ptr<DeviceArray<int>> primitive_id{};

    std::unique_ptr<DeviceArray<int>> mat_id{};
    std::unique_ptr<DeviceArray<bool>> is_intersected{};
  } interaction_pool;

  void copyToConstantMemory() const;

  std::unique_ptr<LBVH> lbvh{}; ///< BVH accelerator aggregate
  std::unique_ptr<CudaStream> stream{}; ///< CUDA stream
  std::unique_ptr<MultipleWorkPool> work_pool;
  std::unique_ptr<DeviceArray<RandomGenerator>> rngs{};
  std::unique_ptr<DeviceArray<uint64_t>> ray_keys{};
  HDRImageBuffer sampleBuffer; ///< sample buffer
  Timer timer; ///< performance test timer
  std::unique_ptr<Scene> scene{}; ///< current scene
  std::unique_ptr<Camera> camera{}; ///< current camera

  // Tonemapping Controls //

  double tm_level; ///< exposure level
  double tm_key; ///< key value
  double tm_wht; ///< white point
};

} // namespace CGL

#endif  // CGL_PATHTRACER_H
