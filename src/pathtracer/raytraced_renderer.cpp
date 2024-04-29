#include "raytraced_renderer.h"
#include "bsdf.h"

#include <stack>
#include <algorithm>
#include <sstream>

#include "CGL/vector3D.h"
#include "CGL/lodepng.h"

#include "GL/glew.h"

#include "scene/light.h"
#include "scene/scene.h"
#include "sampler.h"

using std::min;
using std::max;

namespace CGL {
/**
 * Raytraced Renderer is a render controller that in this case.
 * It controls a path tracer to produce an rendered image from the input parameters.
 *
 * A pathtracer with BVH accelerator and BVH visualization capabilities.
 * It is always in exactly one of the following states:
 * -> INIT: is missing some data needed to be usable, like a camera or scene.
 * -> READY: fully configured, but not rendering.
 * -> VISUALIZE: visualizatiNG BVH aggregate.
 * -> RENDERING: rendering a scene.
 * -> DONE: completed rendering a scene.
 */
RaytracedRenderer::RaytracedRenderer(size_t ns_aa,
                                     size_t max_ray_depth,
                                     bool isAccumBounces,
                                     size_t ns_area_light,
                                     size_t ns_diff,
                                     size_t ns_glsy,
                                     size_t ns_refr,
                                     size_t num_threads,
                                     size_t samples_per_batch,
                                     float max_tolerance,
                                     HDRImageBuffer *envmap,
                                     bool direct_hemisphere_sample,
                                     string filename,
                                     double lensRadius,
                                     double focalDistance) {
  state = INIT;

  pt = new PathTracer();

  pt->ns_aa = ns_aa; // Number of samples per pixel
  pt->max_ray_depth = max_ray_depth; // Maximum recursion ray depth
  pt->ns_area_light = ns_area_light; // Number of samples for area light
  pt->ns_diff = ns_diff; // Number of samples for diffuse surface
  pt->ns_glsy = ns_diff; // Number of samples for glossy surface
  pt->ns_refr = ns_refr; // Number of samples for refraction
  pt->samplesPerBatch = samples_per_batch; // Number of samples per batch
  pt->maxTolerance = max_tolerance; // Maximum tolerance for early termination
  pt->direct_hemisphere_sample = direct_hemisphere_sample;
  // Whether to use direct hemisphere sampling vs. Importance Sampling

  this->lensRadius = lensRadius;
  this->focalDistance = focalDistance;

  this->filename = filename;

  if (envmap) {
    // pt->envLight = new EnvironmentLight(envmap);
  } else {
    // pt->envLight = NULL;
  }

  scene = NULL;
  camera = NULL;
}

/**
 * Destructor.
 * Frees all the internal resources used by the pathtracer.
 */
RaytracedRenderer::~RaytracedRenderer() {
  delete pt;
}

/**
 * If in the INIT state, configures the pathtracer to use the given scene. If
 * configuration is done, transitions to the READY state.
 * This DOES take ownership of the scene, and therefore deletes it if a new
 * scene is later passed in.
 * \param scene pointer to the new scene to be rendered
 */
void RaytracedRenderer::set_scene(SceneObjects::Scene *scene) {
  if (state != INIT) {
    return;
  }

  if (this->scene != nullptr) {
    // delete scene;
    // delete bvh;
  }

  // if (pt->envLight != nullptr) {
  // scene->lights.push_back(pt->envLight);
  // }
  std::cout << "Set scene" << std::endl;
  this->scene = std::make_unique<Scene>();
  build_accel(scene);
  if (has_valid_configuration()) {
    state = READY;
  }
}

/**
 * If in the INIT state, configures the pathtracer to use the given camera. If
 * configuration is done, transitions to the READY state.
 * This DOES NOT take ownership of the camera, and doesn't delete it ever.
 * \param camera the camera to use in rendering
 */
void RaytracedRenderer::set_camera(Camera *camera) {
  if (state != INIT) {
    return;
  }

  camera->focalDistance = focalDistance;
  camera->lensRadius = lensRadius;
  this->camera = std::make_unique<Camera>(*camera);
  if (has_valid_configuration()) {
    state = READY;
  }
}

/**
 * Sets the pathtracer's frame size. If in a running state (VISUALIZE,
 * RENDERING, or DONE), transitions to READY b/c a changing window size
 * would invalidate the output. If in INIT and configuration is done,
 * transitions to READY.
 * \param width width of the frame
 * \param height height of the frame
 */
void RaytracedRenderer::set_frame_size(size_t width, size_t height) {
  if (state != INIT && state != READY) {
    stop();
  }
  cudaSafeCheck();
  frame_w = width;
  frame_h = height;

  frameBuffer.resize(width, height);
  cell_tl = Vector2D(0, 0);
  cell_br = Vector2D(width, height);
  render_cell = false;

  pt->set_frame_size(width, height);

  if (has_valid_configuration()) {
    state = READY;
  }
}

bool RaytracedRenderer::has_valid_configuration() {
  return scene && camera;
}

/**
 * Update result on screen.
 * If the pathtracer is in RENDERING or DONE, it will display the result in
 * its frame buffer. If the pathtracer is in VISUALIZE mode, it will draw
 * the BVH visualization with OpenGL.
 */
void RaytracedRenderer::update_screen() {
  switch (state) {
    case INIT:
    case READY:break;
    case VISUALIZE:visualize_accel();
      break;
    case RENDERING:
      glDrawPixels(frameBuffer.w,
                   frameBuffer.h,
                   GL_RGBA,
                   GL_UNSIGNED_BYTE,
                   &frameBuffer.data[0]);
      if (render_cell)
        visualize_cell();
      break;
    case DONE:
      glDrawPixels(frameBuffer.w,
                   frameBuffer.h,
                   GL_RGBA,
                   GL_UNSIGNED_BYTE,
                   &frameBuffer.data[0]);
      if (render_cell)
        visualize_cell();
      break;
  }
}

/**
 * Transitions from any running state to READY.
 */
void RaytracedRenderer::stop() {
  switch (state) {
    case INIT:
    case READY:break;
    case VISUALIZE:state = READY;
      break;
    case RENDERING:
    case DONE:state = READY;
      break;
  }
}

/**
 * If the pathtracer is in READY, delete all internal data, transition to INIT.
 */
void RaytracedRenderer::clear() {
  if (state != READY) return;
  scene = NULL;
  camera = NULL;
  frameBuffer.resize(0, 0);
  state = INIT;
  render_cell = false;

  pt->clear();
}

/**
 * If the pathtracer is in READY, transition to VISUALIZE.
 */
void RaytracedRenderer::start_visualizing() {
  if (state != READY) {
    return;
  }
  state = VISUALIZE;
}

/**
 * If the pathtracer is in READY, transition to RENDERING.
 */
void RaytracedRenderer::start_raytracing() {
  if (state != READY) return;

  state = RENDERING;

  size_t width = frameBuffer.w;
  size_t height = frameBuffer.h;

  pt->clear();
  pt->set_frame_size(width, height);
  pt->lbvh = std::move(lbvh);
  pt->camera = std::move(camera);
  pt->scene = std::move(scene);

  // launch threads
  fprintf(stdout, "[PathTracer] Rendering... \n");
  fflush(stdout);
  timer.start();
  pt->raytrace();
  timer.stop();
  fprintf(stdout, "[PathTracer] Rendering complete: %.4f sec\n", timer.duration());
  state = DONE;
}

void RaytracedRenderer::render_to_file(string filename, size_t x, size_t y, size_t dx, size_t dy) {
  if (x == -1) {
    printf("Rendering to file %s...\n", filename.c_str());
    start_raytracing();
    save_image(filename);
    fprintf(stdout, "[PathTracer] Job completed.\n");
  } else {
    render_cell = true;
    cell_tl = Vector2D(x, y);
    cell_br = Vector2D(x + dx, y + dy);
    ImageBuffer buffer;
    raytrace_cell(buffer);
    save_image(filename, &buffer);
    fprintf(stdout, "[PathTracer] Cell job completed.\n");
  }
}

static Surface decideMaterial(BSDF *bsdf,
                              const std::map<DiffuseBSDF *, int> &diffuse_map,
                              const std::map<MirrorBSDF *, int> &mirror_map,
                              const std::map<GlassBSDF *, int> &glass_map,
                              const std::map<EmissionBSDF *, int> &emission_map) {
  if (auto diffuse = dynamic_cast<DiffuseBSDF *>(bsdf); diffuse != nullptr) {
    return Surface(SurfaceInfo::Diffuse, diffuse_map.at(diffuse));
  }
  if (auto mirror = dynamic_cast<MirrorBSDF *>(bsdf); mirror != nullptr) {
    return Surface(SurfaceInfo::Mirror, mirror_map.at(mirror));
  }
  if (auto glass = dynamic_cast<GlassBSDF *>(bsdf); glass != nullptr) {
    return Surface(SurfaceInfo::Dielectric, glass_map.at(glass));
  }
  if (auto emissive = dynamic_cast<EmissionBSDF *>(bsdf); emissive != nullptr) {
    return Surface(SurfaceInfo::Emissive, emission_map.at(emissive));
  }
  std::cerr << "Unsupported BSDF type" << std::endl;
  exit(-1);
}

void RaytracedRenderer::build_accel(SceneObjects::Scene *cpu_scene) {
  cudaSafeCheck();
  // collect primitives //
  fprintf(stdout, "[PathTracer] Collecting primitives... ");
  fflush(stdout);
  timer.start();
//  buildLightAsObjects(*cpu_scene);
  vector<SceneObjects::Primitive *> primitives;
  std::map<DiffuseBSDF *, int> diffuse_map;
  std::map<MirrorBSDF *, int> mirror_map;
  std::map<GlassBSDF *, int> glass_map;
  std::map<EmissionBSDF *, int> emission_map;
  int mesh_count = 0;
  int triangle_count = 0;
  int sphere_count = 0;
  int diffuse_count = 0;
  int metal_count = 0;
  int glass_count = 0;
  int emission_count = 0;
  std::vector<DiffuseData> diffuse_data;
  std::vector<MirrorData> mirror_data;
  std::vector<DielectricData> glass_data;
  std::vector<EmissiveData> emissive_data;
  int light_num = 0;
  for (SceneObjects::SceneObject *obj : cpu_scene->objects) {
    if (auto mesh = dynamic_cast<SceneObjects::Mesh *>(obj); mesh != nullptr) {
      mesh_count++;
      assert(mesh->indices.size() % 3 == 0);
      triangle_count += mesh->indices.size() / 3;
      if (auto emissive = dynamic_cast<EmissionBSDF *>(obj->get_bsdf()); emissive != nullptr) {
        light_num += mesh->indices.size() / 3;
      }
    } else if (dynamic_cast<SceneObjects::SphereObject *>(obj)) {
      sphere_count++;
      if (auto emissive = dynamic_cast<EmissionBSDF *>(obj->get_bsdf()); emissive != nullptr) {
        light_num++;
      }
    } else {
      std::cerr << "Unsupported object type" << std::endl;
      exit(-1);
    }
    BSDF *bsdf = obj->get_bsdf();
    if (auto diffuse = dynamic_cast<DiffuseBSDF *>(bsdf); diffuse != nullptr) {
      if (diffuse_map.contains(diffuse)) continue;
      diffuse_map[diffuse] = diffuse_count++;
      diffuse_data.emplace_back(diffuse->reflectance);
    } else if (auto mirror = dynamic_cast<MirrorBSDF *>(bsdf); mirror != nullptr) {
      if (mirror_map.contains(mirror)) continue;
      mirror_map[mirror] = metal_count++;
//      mirror_data.emplace_back(mirror->reflectance);
    }
//    else if (auto glass = dynamic_cast<GlassBSDF *>(bsdf); glass != nullptr) {
//      if (glass_map.contains(glass)) continue;
//      glass_map[glass] = glass_count++;
//      dielectric_data.emplace_back(glass->reflectance, glass->transmittance);
//    }
    else if (auto emissive = dynamic_cast<EmissionBSDF *>(bsdf); emissive != nullptr) {
      if (emission_map.contains(emissive)) continue;
      emission_map[emissive] = emission_count++;
      emissive_data.emplace_back(emissive->radiance);
    } else {
      std::cerr << "Unsupported BSDF type" << std::endl;
      exit(-1);
    }
  }
  // PrimitivePool
  std::vector<Shape> cpu_shapes(triangle_count + sphere_count);
  std::vector<Surface> cpu_materials(triangle_count + sphere_count);
  // Meshes
  std::vector<DeviceArray<uint32_t>> device_indices(mesh_count);
  std::vector<DeviceArray<double3>> device_positions(mesh_count);
  std::vector<DeviceArray<double3>> device_normals(mesh_count);
  int pr_id = 0, mesh_id = 0;
  scene->meshes = std::make_unique<DeviceArray<Mesh>>(mesh_count);
  std::vector<int> host_light_indices;
  std::vector<int> host_light_idx_map(triangle_count + sphere_count, 0);
  std::vector<Mesh> cpu_meshes(mesh_count);
  for (int obj_id = 0; obj_id < cpu_scene->objects.size(); obj_id++) {
    auto obj = cpu_scene->objects[obj_id];
    Surface mat = decideMaterial(obj->get_bsdf(), diffuse_map, mirror_map, glass_map, emission_map);
    if (auto mesh = dynamic_cast<SceneObjects::Mesh *>(obj); mesh != nullptr) {
      std::vector<double3> vertices(mesh->num_vertices);
      std::vector<double3> normals(mesh->num_vertices);
      std::vector<uint32_t> indices(mesh->indices.size());
      for (int i = 0; i < mesh->indices.size(); i++)
        indices[i] = static_cast<uint32_t>(mesh->indices[i]);
      for (int i = 0; i < mesh->num_vertices; i++) {
        vertices[i] = to_double3(mesh->positions[i]);
        normals[i] = to_double3(mesh->normals[i]);
      }
      device_indices[mesh_id] = DeviceArray<uint32_t>(indices);
      device_positions[mesh_id] = DeviceArray<double3>(vertices);
      device_normals[mesh_id] = DeviceArray<double3>(normals);
      auto *gpu_mesh_ptr = scene->meshes->data() + mesh_id;
      for (int i = 0; i < mesh->indices.size(); i += 3) {
        cpu_shapes[pr_id].getTriangle() = Triangle(gpu_mesh_ptr, i);
        cpu_materials[pr_id] = mat;
        if (mat.surface_info == SurfaceInfo::Emissive) {
          int light_id = host_light_indices.size();
          assert(host_light_indices.size() == light_id);
          host_light_indices.push_back(pr_id);
          host_light_idx_map[pr_id] = light_id;
        }
        pr_id++;
      }
      mesh_id++;
    } else if (auto sphere = dynamic_cast<SceneObjects::SphereObject *>(obj); sphere) {
      double3 center = to_double3(sphere->o);
      double radius = sphere->r;
      cpu_shapes[pr_id].getSphere() = Sphere(center, radius);
      if (mat.surface_info == SurfaceInfo::Emissive) {
        int light_id = host_light_indices.size();
        assert(host_light_indices.size() == light_id);
        host_light_indices.push_back(pr_id);
        host_light_idx_map[pr_id] = light_id;
      }
      cpu_materials[pr_id] = mat;
      pr_id++;
    }
  }
  std::vector<double> weights(host_light_indices.size());
  for (int i = 0; i < host_light_indices.size(); i++)
    weights[i] = illum(emissive_data[cpu_materials[host_light_indices[i]].surf_id].radiance);
  scene->light_sampler = std::make_unique<LightSampler>();
  scene->light_sampler->light_dist = std::make_unique<DiscreteDistribution>(weights.size());
  scene->light_sampler->light_dist->buildFromWeights(weights);
  scene->light_sampler->light_indices = std::make_unique<DeviceArray<int>>(host_light_indices);
  scene->light_sampler->map_light_to_primitive = std::make_unique<DeviceArray<int>>(host_light_idx_map);
  scene->mesh_pool.indices = std::move(device_indices);
  scene->mesh_pool.vertices = std::move(device_positions);
  scene->mesh_pool.normals = std::move(device_normals);
  for (int i = 0; i < mesh_count; i++)
    cpu_meshes[i] = Mesh(scene->mesh_pool.vertices[i].constAccessor(),
                         scene->mesh_pool.normals[i].constAccessor(),
                         scene->mesh_pool.indices[i].constAccessor());
  scene->meshes->copyFrom(cpu_meshes);
  scene->diffuse_data = std::make_unique<DeviceArray<DiffuseData>>(diffuse_data);
  scene->mirror_data = std::make_unique<DeviceArray<MirrorData>>(mirror_data);
  scene->dielectric_data = std::make_unique<DeviceArray<DielectricData>>(glass_data);
  scene->emissive_data = std::make_unique<DeviceArray<EmissiveData>>(emissive_data);
  timer.stop();
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());

  // build BVH //
  fprintf(stdout, "[PathTracer] Building BVH from %lu primitives... ", pr_id);
  fflush(stdout);
  timer.start();
  lbvh = std::make_unique<LBVH>(pr_id, cpu_shapes, cpu_materials);
  timer.stop();
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());
}

void RaytracedRenderer::visualize_accel() const {
}

void RaytracedRenderer::visualize_cell() const {
}

/**
 * If the pathtracer is in VISUALIZE, handle key presses to traverse the bvh.
 */
void RaytracedRenderer::key_press(int key) {
  switch (key) {
    case ']':pt->ns_aa *= 2;
      fprintf(stdout, "[PathTracer] Samples per pixel changed to %lu\n", pt->ns_aa);
      //tm_key = clamp(tm_key + 0.02f, 0.0f, 1.0f);
      break;
    case '[':
      //tm_key = clamp(tm_key - 0.02f, 0.0f, 1.0f);
      pt->ns_aa /= 2;
      if (pt->ns_aa < 1) pt->ns_aa = 1;
      fprintf(stdout, "[PathTracer] Samples per pixel changed to %lu\n", pt->ns_aa);
      break;
    case '=':
    case '+':pt->ns_area_light *= 2;
      fprintf(stdout, "[PathTracer] Area light sample count increased to %zu.\n", pt->ns_area_light);
      break;
    case '-':
    case '_':if (pt->ns_area_light > 1) pt->ns_area_light /= 2;
      fprintf(stdout, "[PathTracer] Area light sample count decreased to %zu.\n", pt->ns_area_light);
      break;
    case '.':
    case '>':pt->max_ray_depth++;
      fprintf(stdout, "[PathTracer] Max ray depth increased to %zu.\n", pt->max_ray_depth);
      break;
    case ',':
    case '<':if (pt->max_ray_depth) pt->max_ray_depth--;
      fprintf(stdout, "[PathTracer] Max ray depth decreased to %zu.\n", pt->max_ray_depth);
      break;
    case 'h':
    case 'H':pt->direct_hemisphere_sample = !pt->direct_hemisphere_sample;
      fprintf(stdout,
              "[PathTracer] Toggled direct lighting to %s\n",
              (pt->direct_hemisphere_sample ? "uniform hemisphere sampling" : "importance light sampling"));
      break;
    case 'k':
    case 'K':pt->camera->lensRadius = std::max(pt->camera->lensRadius - 0.05, 0.0);
      fprintf(stdout, "[PathTracer] Camera lens radius reduced to %f.\n", pt->camera->lensRadius);
      break;
    case 'l':
    case 'L':pt->camera->lensRadius = pt->camera->lensRadius + 0.05;
      fprintf(stdout, "[PathTracer] Camera lens radius increased to %f.\n", pt->camera->lensRadius);
      break;
    case ';':pt->camera->focalDistance = std::max(pt->camera->focalDistance - 0.1, 0.0);
      fprintf(stdout, "[PathTracer] Camera focal distance reduced to %f.\n", pt->camera->focalDistance);
      break;
    case '\'':pt->camera->focalDistance = pt->camera->focalDistance + 0.1;
      fprintf(stdout, "[PathTracer] Camera focal distance increased to %f.\n", pt->camera->focalDistance);
      break;
    case 'C':render_cell = !render_cell;
      if (render_cell)
        fprintf(stdout, "[PathTracer] Now in cell render mode.\n");
      else
        fprintf(stdout, "[PathTracer] No longer in cell render mode.\n");
      break;

    default:return;
  }
}

/**
 * Raytrace a tile of the scene and update the frame buffer. Is run
 * in a worker thread.
 */
void RaytracedRenderer::raytrace_tile(int tile_x,
                                      int tile_y,
                                      int tile_w,
                                      int tile_h) {
}

void RaytracedRenderer::raytrace_cell(ImageBuffer &buffer) {
  size_t tile_start_x = cell_tl.x;
  size_t tile_start_y = cell_tl.y;

  size_t tile_end_x = cell_br.x;
  size_t tile_end_y = cell_br.y;

  size_t w = tile_end_x - tile_start_x;
  size_t h = tile_end_y - tile_start_y;
  HDRImageBuffer sb(w, h);
  buffer.resize(w, h);

  stop();
  render_cell = true;
  start_raytracing();
  for (size_t y = tile_start_y; y < tile_end_y; y++) {
    for (size_t x = tile_start_x; x < tile_end_x; x++) {
      buffer.data[w * (y - tile_start_y) + (x - tile_start_x)] = frameBuffer.data[x + y * frame_w];
    }
  }
}

void RaytracedRenderer::autofocus(Vector2D loc) {
  pt->autofocus(loc);
}

void RaytracedRenderer::worker_thread() {
}

void RaytracedRenderer::save_image(string filename, ImageBuffer *buffer) {
  if (state != DONE) return;

  if (!buffer)
    buffer = &frameBuffer;

  if (filename == "") {
    time_t rawtime;
    time(&rawtime);

    time_t t = time(nullptr);
    tm *lt = localtime(&t);
    stringstream ss;
    ss << this->filename << "_screenshot_" << lt->tm_mon + 1 << "-" << lt->tm_mday << "_"
       << lt->tm_hour << "-" << lt->tm_min << "-" << lt->tm_sec << ".png";
    filename = ss.str();
  }

  pt->sampleBuffer.toColor(*buffer);
  uint32_t *frame = &buffer->data[0];
  size_t w = buffer->w;
  size_t h = buffer->h;
  uint32_t *frame_out = new uint32_t[w * h];
  for (size_t i = 0; i < h; ++i) {
    memcpy(frame_out + i * w, frame + (h - i - 1) * w, 4 * w);
  }

  for (size_t i = 0; i < w * h; ++i) {
    frame_out[i] |= 0xFF000000;
  }
  std::cout << "[PathTracer] Saving to file: " << filename << std::endl;
  lodepng::encode(filename, (unsigned char *) frame_out, w, h);
  std::cout << "[PathTracer] Saved!" << std::endl;
  delete[] frame_out;

}

} // namespace CGL