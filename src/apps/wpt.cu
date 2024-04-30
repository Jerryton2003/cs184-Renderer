//
// Created by creeper on 4/30/24.
//
#include <iostream>
#include <scene/scene.h>
#include "pathtracer/camera.h"
#include "scene/shape.h"
#include "scene/primitive.h"
#include "pathtracer/raytraced_renderer.h"
#include "util/obj-io.h"
#include "util/medium-io.h"
#include "scene/media.cuh"
#include "util/device-vec-ops.h"
#include <tuple>

std::string outputDir = "/home/creeper/cs184-Renderer/output/";
std::string lightningRedDir = "/home/creeper/cs184-Renderer/assets/lightning_red/";
std::string lightningBlueDir = "/home/creeper/cs184-Renderer/assets/lightning_blue/";
std::string smokeADir = "/home/creeper/cs184-Renderer/assets/smoke_a/";
std::string smokeBDir = "/home/creeper/cs184-Renderer/assets/smoke_b/";
std::string bunnyDir = "/home/creeper/cs184-Renderer/assets/";
std::string bunnyFilename = "complex_bunny.obj";
std::string fluidObjDir = "/home/creeper/cs184-Renderer/assets/obj_file/";

using namespace CGL;

void relocateMesh(CGL::ObjMesh &mesh, const double3 &size) {
  double max_coord = -1e9, min_coord = 1e9;
  for (const auto &v : mesh.vertices) {
    max_coord = std::max(max_coord, std::max(std::max(v.x, v.y), v.z));
    min_coord = std::min(min_coord, std::min(std::min(v.x, v.y), v.z));
  }
  double scene_scale = std::min(std::min(size.x, size.y), size.z);
  double scale = scene_scale / (max_coord - min_coord);
  for (auto &v : mesh.vertices) {
    v = (v - make_constant(min_coord)) * scale;
    v *= 0.5;
    assert(
        v.x >= 0.0 && v.x <= scene_scale && v.y >= 0.0 && v.y <= scene_scale &&
        v.z >= 0.0 && v.z <= scene_scale);
  }
}

void loadBunny(ObjMesh &mesh) {
  std::string filename = bunnyDir + bunnyFilename;
  std::cout << "loading " << filename << std::endl;
  if (!myLoadObj(filename, &mesh)) {
    std::cerr << "failed to load " << filename << std::endl;
    exit(1);
  }
  relocateMesh(mesh, make_constant(1.0));
}

void loadFluid(ObjMesh &mesh, int idx) {
  std::string filename = fluidObjDir + "fluid_" + std::to_string(idx) + ".obj";
  std::cout << "loading " << filename << std::endl;
  if (!myLoadObj(filename, &mesh)) {
    std::cerr << "failed to load " << filename << std::endl;
    exit(1);
  }
}

double rad(double deg) {
  return deg / 180.0 * M_PI;
}

double deg(double rad) {
  return rad / M_PI * 180.0;
}

std::unique_ptr<CGL::Camera> hardCodedCamera(int frame_idx) {
  auto camera = std::make_unique<CGL::Camera>();
  camera->nClip = 0.1;
  camera->fClip = 100;
  camera->hFov = 43.85528;
  double aspect = 1.0;
  camera->vFov = deg(2 * atan(tan(rad(camera->hFov / 2)) / aspect));
  printf("vFov: %f\n", camera->vFov);
  camera->screenW = 480;
  camera->screenH = 360;
  camera->pos = make_double3(0.2984593, 0.4037158, -0.8174311);
  camera->c2w(0, 0) = -0.9995065;
  camera->c2w(0, 1) = -2.26441e-4;
  camera->c2w(0, 2) = -0.03141583;
  camera->c2w(1, 0) = -7.76905e-6;
  camera->c2w(1, 1) = 0.9999758;
  camera->c2w(1, 2) = -0.006960499;
  camera->c2w(2, 0) = 0.03141665;
  camera->c2w(2, 1) = -0.006956819;
  camera->c2w(2, 2) = -0.9994822;
  return std::move(camera);
}

void addMesh(int idx,
             std::vector<CGL::Shape> &shapes,
             std::vector<CGL::Surface> &materials,
             std::unique_ptr<CGL::Scene> &scene,
             std::vector<CGL::Mesh> &host_meshes,
             const CGL::ObjMesh &host_mesh,
             const CGL::Surface &material) {
  scene->mesh_pool.vertices[idx].copyFrom(host_mesh.vertices);
  scene->mesh_pool.normals[idx].copyFrom(host_mesh.normals);
  scene->mesh_pool.indices[idx].copyFrom(host_mesh.indices);
  for (int i = 0; i < host_mesh.triangleCount; i++) {
    Shape shape;
    shape.getTriangle() = Triangle(scene->meshes->data() + idx, i * 3);
    shapes.emplace_back(shape);
    materials.emplace_back(material);
  }
  host_meshes[idx] = CGL::Mesh(scene->mesh_pool.vertices[idx].constAccessor(),
                               scene->mesh_pool.normals[idx].constAccessor(),
                               scene->mesh_pool.indices[idx].constAccessor());
}

void addSphere(const double3 &centre,
               double radius,
               std::vector<CGL::Shape> &shapes,
               std::vector<CGL::Surface> &materials,
               const CGL::Surface &material) {
  Shape shape;
  shape.getSphere() = Sphere(centre, radius);
  shapes.emplace_back(shape);
  materials.emplace_back(material);
  std::cout << "add sphere at " << centre << " with radius " << radius << std::endl;
}

int addIsotropicPhaseFunction(const double3 &albedo,
                              std::vector<IsotropicData> &isotropic_data,
                              std::vector<PhaseFunction> &phase_functions) {
  isotropic_data.emplace_back(albedo);
  phase_functions.emplace_back(PhaseFunction{Isotropic, static_cast<int>(isotropic_data.size() - 1)});
  return phase_functions.size() - 1;
}

int addDiffuse(const double3 &albedo,
               std::vector<DiffuseData> &diffuse_data) {
  diffuse_data.emplace_back(albedo);
  return diffuse_data.size() - 1;
}

int addEmissive(const double3 &radiance,
                std::vector<EmissiveData> &emissive_data) {
  emissive_data.emplace_back(radiance);
  return emissive_data.size() - 1;
}

void loadLightning(int frame_idx,
                   int8_t medium_idx,
                   const double3 &centre,
                   double radius,
                   const std::unique_ptr<CGL::Scene> &scene,
                   std::vector<CGL::Shape> &shapes,
                   std::vector<Medium> &media,
                   std::vector<MediumInterfaceData> &medium_interface_data,
                   std::vector<CGL::Surface> &materials,
                   const CGL::Surface &material) {
  Volume volume_red;
  Volume volume_blue;
  std::string filepath_red = lightningRedDir + "frame_" + std::to_string(frame_idx) + ".vol";
  std::string filepath_blue = lightningBlueDir + "frame_" + std::to_string(frame_idx) + ".vol";
  loadVolume(filepath_red, &volume_red);
  loadVolume(filepath_blue, &volume_blue);
  auto lightning = mixVolumes(volume_red, volume_blue);
  lightning.orig = centre - make_double3(radius, radius, radius);
  lightning.spacing = make_double3(2 * radius / lightning.resolution.x,
                                   2 * radius / lightning.resolution.y,
                                   2 * radius / lightning.resolution.z);
  addSphere(centre, radius, shapes, materials, material);
  medium_interface_data.push_back(MediumInterfaceData{medium_idx, -1});
  materials.emplace_back(CGL::Surface{MediumInterface, static_cast<int>(medium_interface_data.size() - 1)});
  Medium medium;
  medium.getHeterogeneousMedium().orig = lightning.orig;
  medium.getHeterogeneousMedium().spacing = lightning.spacing;
  medium.getHeterogeneousMedium().resolution = lightning.resolution;
  scene->vol_textures.emplace_back(std::make_unique<CudaTexture<float4>>(make_uint3(lightning.resolution.x,
                                                                                    lightning.resolution.y,
                                                                                    lightning.resolution.z)));
  scene->vol_textures.back()->copyFrom(lightning.density);
  int density_tex_idx = scene->vol_textures.size() - 1;
  scene->vol_textures.emplace_back(std::make_unique<CudaTexture<float4>>(make_uint3(lightning.resolution.x,
                                                                                    lightning.resolution.y,
                                                                                    lightning.resolution.z)));
  scene->vol_textures.back()->copyFrom(lightning.albedo);
  int albedo_tex_idx = scene->vol_textures.size() - 1;
  medium.getHeterogeneousMedium().density = scene->vol_textures[density_tex_idx]->texAccessor();
  medium.getHeterogeneousMedium().albedo = scene->vol_textures[albedo_tex_idx]->texAccessor();
  medium.getHeterogeneousMedium().majorant = lightning.majorant;
  media.emplace_back(medium);
  std::cout << "load lightning " << frame_idx << " at " << centre << " with radius " << radius << std::endl;
}

void addSphereLight(const double3 &centre,
                    double radius,
                    std::vector<CGL::Shape> &shapes,
                    std::vector<CGL::Surface> &materials,
                    std::vector<int> &host_light_indices,
                    std::vector<int> &host_light_idx_map,
                    std::vector<EmissiveData> &emissive_data,
                    std::vector<double> &weights,
                    int emissive_idx) {
  int pr_id = shapes.size();
  int light_id = host_light_indices.size();
  addSphere(centre, radius, shapes, materials, CGL::Surface(SurfaceInfo::Emissive, emissive_idx));
  host_light_indices.push_back(pr_id);
  if (host_light_idx_map.size() <= pr_id)
    host_light_idx_map.resize(pr_id + 1);
  host_light_idx_map[pr_id] = light_id;
  weights.push_back(illum(emissive_data[emissive_idx].radiance));
  std::cout << "add light " << light_id << " at primitive " << pr_id << std::endl;
}

std::tuple<std::unique_ptr<Scene>,
           std::vector<Shape>,
           std::vector<Surface>> hardCodedScene(int frame_idx) {
  auto scene = std::make_unique<CGL::Scene>();
  std::vector<Shape> shapes;
  std::vector<Surface> materials;
  std::vector<Medium> host_media;
  std::vector<PhaseFunction> host_phase_functions;
  std::vector<IsotropicData> host_isotropic_data;
  std::vector<DiffuseData> host_diffuse_data;
  std::vector<EmissiveData> host_emissive_data;
  std::vector<MediumInterfaceData> host_medium_interface_data;
  std::vector<int> host_light_indices;
  std::vector<int> host_light_idx_map;
  std::vector<double> weights;
  auto isotropic = addIsotropicPhaseFunction(make_constant(1.0), host_isotropic_data, host_phase_functions);
  auto diffuse_bunny = addDiffuse(make_constant(0.5), host_diffuse_data);
  auto diffuse_fluid = addDiffuse(make_double3(0.0, 0.3, 0.7), host_diffuse_data);
  auto emissive = addEmissive(make_constant(5.0), host_emissive_data);
  int mesh_cnt = 2;
  std::vector<CGL::Mesh> host_meshes(mesh_cnt);
  scene->meshes = std::make_unique<CGL::DeviceArray<CGL::Mesh>>(mesh_cnt);
  scene->mesh_pool.vertices = std::vector<CGL::DeviceArray<double3>>(mesh_cnt);
  scene->mesh_pool.normals = std::vector<CGL::DeviceArray<double3>>(mesh_cnt);
  scene->mesh_pool.indices = std::vector<CGL::DeviceArray<uint32_t>>(mesh_cnt);
//  ObjMesh bunny;
//  loadBunny(bunny);
//  addMesh(0, shapes, materials, scene, host_meshes, bunny, {SurfaceInfo::Diffuse, diffuse_bunny});
  ObjMesh fluid;
  loadFluid(fluid, frame_idx);
  addMesh(1, shapes, materials, scene, host_meshes, fluid, CGL::Surface(SurfaceInfo::Diffuse, diffuse_fluid));
  addSphereLight(make_double3(2.0, 2.0, 2.0),
                 0.3,
                 shapes,
                 materials,
                 host_light_indices,
                 host_light_idx_map,
                 host_emissive_data,
                 weights,
                 emissive);
  addSphereLight(make_double3(2.0, 2.0, -1.0),
                 0.3,
                 shapes,
                 materials,
                 host_light_indices,
                 host_light_idx_map,
                 host_emissive_data,
                 weights,
                 emissive);
  scene->light_sampler = std::make_unique<CGL::LightSampler>();
  scene->light_sampler->light_dist = std::make_unique<DiscreteDistribution>(weights.size());
  scene->light_sampler->light_dist->buildFromWeights(weights);
  scene->light_sampler->light_indices = std::make_unique<DeviceArray<int>>(host_light_indices);
  scene->light_sampler->map_light_to_primitive = std::make_unique<DeviceArray<int>>(host_light_idx_map);
  scene->meshes->copyFrom(host_meshes);
  scene->phase_functions = std::make_unique<CGL::DeviceArray<PhaseFunction>>(host_phase_functions);
  scene->isotropic_data = std::make_unique<CGL::DeviceArray<IsotropicData>>(host_isotropic_data);
  scene->diffuse_data = std::make_unique<CGL::DeviceArray<DiffuseData>>(host_diffuse_data);
  scene->emissive_data = std::make_unique<CGL::DeviceArray<EmissiveData>>(host_emissive_data);
  scene->media = std::make_unique<CGL::DeviceArray<Medium>>(host_media);
  scene->medium_interface_data = std::make_unique<CGL::DeviceArray<MediumInterfaceData>>(host_medium_interface_data);
  return std::make_tuple(std::move(scene), std::move(shapes), std::move(materials));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: wpt <frame_idx>" << std::endl;
    return 1;
  }
  int frame_idx = std::stoi(argv[1]);
  auto camera = hardCodedCamera(frame_idx);
  auto [scene, shapes, materials] = hardCodedScene(frame_idx);
  auto ray_tracer = std::make_unique<CGL::RaytracedRenderer>();
  ray_tracer->set_camera(camera);
  ray_tracer->set_scene(scene);
  ray_tracer->build_accel(shapes.size(), shapes, materials);
  std::string filepath = outputDir + "frame_" + std::to_string(frame_idx) + ".png";
  std::cout << "render to frame_" << frame_idx << ".png" << std::endl;
  ray_tracer->render_to_file(filepath);
  return 0;
}