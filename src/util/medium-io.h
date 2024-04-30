//
// Created by creeper on 4/30/24.
//

#ifndef SOSONOWPT_SRC_UTIL_MEDIUM_IO_H_
#define SOSONOWPT_SRC_UTIL_MEDIUM_IO_H_

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "device-vec-ops.h"

namespace CGL {

struct Volume {
  int3 resolution;
  std::vector<float4> density;
  std::vector<float4> albedo;
  double3 orig, spacing, majorant;
};

inline Volume mixVolumes(const Volume &a, const Volume &b) {
  Volume volume;
  assert(a.resolution.x == b.resolution.x && a.resolution.y == b.resolution.y && a.resolution.z == b.resolution.z);
  assert(a.orig.x == b.orig.x && a.orig.y == b.orig.y && a.orig.z == b.orig.z);
  assert(a.spacing.x == b.spacing.x && a.spacing.y == b.spacing.y && a.spacing.z == b.spacing.z);
  volume.resolution = a.resolution;
  volume.orig = a.orig;
  volume.spacing = a.spacing;
  volume.density.resize(a.density.size());
  volume.albedo.resize(a.albedo.size());
  volume.majorant = make_constant(0.0);
  for (int i = 0; i < a.density.size(); i++) {
    volume.density[i] = a.density[i] + b.density[i];
    volume.albedo[i] = a.albedo[i] + b.albedo[i];
    volume.majorant.x = std::max(volume.majorant.x, static_cast<double>(volume.density[i].x));
    volume.majorant.y = std::max(volume.majorant.y, static_cast<double>(volume.density[i].y));
    volume.majorant.z = std::max(volume.majorant.z, static_cast<double>(volume.density[i].z));
  }
  return volume;
}

inline void loadVolume(const std::string &path, Volume *volume) {
  // read the first 3 integerws as resolution
  // use fstream
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return;
  }
  file >> volume->resolution.x >> volume->resolution.y >> volume->resolution.z;
  volume->density.resize(volume->resolution.x * volume->resolution.y * volume->resolution.z);
  volume->albedo.resize(volume->resolution.x * volume->resolution.y * volume->resolution.z);
  for (int i = 0; i < volume->resolution.x * volume->resolution.y * volume->resolution.z; i++) {
    file >> volume->density[i].x >> volume->density[i].y >> volume->density[i].z;
    volume->density[i].w = 0;
  }
  for (int i = 0; i < volume->resolution.x * volume->resolution.y * volume->resolution.z; i++) {
    file >> volume->albedo[i].x >> volume->albedo[i].y >> volume->albedo[i].z;
    volume->albedo[i].w = 0;
  }
}
}
#endif //SOSONOWPT_SRC_UTIL_MEDIUM_IO_H_
