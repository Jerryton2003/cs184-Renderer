#ifndef CGL_RAYTRACER_H
#define CGL_RAYTRACER_H

#include <stack>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <algorithm>

#include "CGL/timer.h"

#include "scene/bvh.h"
#include "pathtracer/camera.h"
#include "pathtracer/sampler.h"
#include "util/image.h"
#include "util/work_queue.h"

#include "scene/scene.h"


#include "pathtracer.h"

namespace CGL {

/**
 * A pathtracer with BVH accelerator and BVH visualization capabilities.
 * It is always in exactly one of the following states:
 * -> INIT: is missing some data needed to be usable, like a camera or scene.
 * -> READY: fully configured, but not rendering.
 * -> VISUALIZE: visualizatiNG BVH aggregate.
 * -> RENDERING: rendering a scene.
 * -> DONE: completed rendering a scene.
 */
class RaytracedRenderer {
 public:

  /**
   * Default constructor.
   * Creates a new pathtracer instance.
   */
  explicit RaytracedRenderer(size_t ns_aa = 1,
                             size_t max_ray_depth = 4, bool is_accumulate_bounces = false, size_t ns_area_light = 1,
                             size_t ns_diff = 1, size_t ns_glsy = 1, size_t ns_refr = 1,
                             size_t num_threads = 1,
                             size_t samples_per_batch = 32,
                             float max_tolerance = 0.05f,
                             HDRImageBuffer *envmap = NULL,
                             bool direct_hemisphere_sample = false,
                             string filename = "",
                             double lensRadius = 0.25,
                             double focalDistance = 4.7);

  /**
   * Destructor.
   * Frees all the internal resources used by the pathtracer.
   */
  ~RaytracedRenderer() = default;

  void clear();

  /**
   * If the pathtracer is in READY, transition to RENDERING.
   */
  void start_raytracing();

  void render_to_file(const std::string &filename);

  /**
   * Save rendered result to png file.
   */
  void save_image(std::string filename = "", ImageBuffer *buffer = NULL);

  void set_camera(std::unique_ptr<Camera> &camera_);
  void set_scene(std::unique_ptr<Scene> &scene_);
  void build_accel(int pr_cnt,
                   const std::vector<Shape> &shapes,
                   const std::vector<Surface> &materials);
 private:
  std::unique_ptr<PathTracer> pt{};

  // Configurables //

  std::unique_ptr<LBVH> lbvh{};           ///< LBVH accelerator
  std::unique_ptr<Scene> scene{};         ///< current scene
  std::unique_ptr<Camera> camera{};       ///< current camera

  // Integration state //

  size_t frame_w, frame_h;

  double lensRadius;
  double focalDistance;

  // Components //

  ImageBuffer frameBuffer;       ///< frame buffer
  Timer timer;                   ///< performance test timer

  std::string filename;
};

}  // namespace CGL

#endif  // CGL_RAYTRACER_H
