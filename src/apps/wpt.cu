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
#include <tuple>

std::string outputDir = "./output/";
std::string lightningRedDir = "./assets/lightning_red/";
std::string lightningBlueDir = "./assets/lightning_blue/";
std::string smokeADir = "./assets/smoke_a/";
std::string smokeBDir = "./assets/smoke_b/";
std::string bunnyDir = "./assets/";
std::string bunnyFilename = "complex_bunny.obj";
std::string fluidObjDir = "./assets/obj_file/";

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

std::unique_ptr<CGL::Camera> hardCodedCamera(int frame_idx) {
  auto camera = std::make_unique<CGL::Camera>();
  camera->nClip = 0.1;
  camera->fClip = 100;

}

std::tuple<std::unique_ptr<CGL::Scene>,
           std::vector<CGL::Shape>,
           std::vector<CGL::Surface>> hardCodedScene(int frame_idx) {
  auto scene = std::make_unique<CGL::Scene>();
  std::vector<CGL::Shape> shapes;
  std::vector<CGL::Surface> materials;
  ObjMesh bunny;
  loadBunny(bunny);

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