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

namespace CGL {

struct Volume {
  int3 resolution;
  std::vector<float3> density;
  std::vector<float3> albedo;
  double3 orig, spacing, majorant;
};

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
  for (int i = 0; i < volume->resolution.x * volume->resolution.y * volume->resolution.z; i++)
    file >> volume->density[i].x >> volume->density[i].y >> volume->density[i].z;
  for (int i = 0; i < volume->resolution.x * volume->resolution.y * volume->resolution.z; i++)
    file >> volume->albedo[i].x >> volume->albedo[i].y >> volume->albedo[i].z;
}
}
#endif //SOSONOWPT_SRC_UTIL_MEDIUM_IO_H_
