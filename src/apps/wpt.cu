//
// Created by creeper on 4/30/24.
//
#include <iostream>
#include <scene/scene.h>
#include "pathtracer/camera.h"
#include "scene/shape.h"
#include "scene/primitive.h"
#include <tuple>

std::unique_ptr<CGL::Camera> hardCodedCamera(int frame_idx) {
  auto camera = std::make_unique<CGL::Camera>();
  camera->nClip = 0.1;
    camera->fClip = 100;

}

std::tuple<std::unique_ptr<CGL::Scene>, std::vector<CGL::Shape>, std::vector<CGL::Surface>> hardCodedScene(int frame_idx) {
  auto scene = std::make_unique<CGL::Scene>();
  std::vector<CGL::Shape> shapes;
  std::vector<CGL::Surface> materials;

  return std::make_tuple(std::move(scene), std::move(shapes), std::move(materials));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: wpt <frame_idx>" << std::endl;
    return 1;
  }
  int frame_idx = std::stoi(argv[1]);
  auto camera = hardCodedCamera(frame_idx);
  auto scene = hardCodedScene(frame_idx);
  return 0;
}