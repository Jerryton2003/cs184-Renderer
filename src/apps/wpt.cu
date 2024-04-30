//
// Created by creeper on 4/30/24.
//
#include <iostream>
#include <scene/scene.h>
#include "pathtracer/camera.h"


std::unique_ptr<CGL::Camera> hardCodedCamera(int frame_idx) {
  auto camera = std::make_unique<CGL::Camera>();

}

std::unique_ptr<CGL::Scene> hardCodedScene(int frame_idx) {
  auto scene = std::make_unique<CGL::Scene>();

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