//
// Created by creeper on 4/30/24.
//
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <format>
#include "obj-io.h"
#include "device-utils.h"
#include "device-vec-ops.h"
namespace CGL {
bool myLoadObj(const std::string& path, ObjMesh* mesh) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return false;
  }
  std::string line;
  std::vector<double3> normals;
  std::vector<int> normal_cnt;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string prefix;
    stream >> prefix;
    if (prefix == "v") {
      double x, y, z;
      stream >> x >> y >> z;
      mesh->vertices.emplace_back(x, y, z);
      mesh->normals.emplace_back(0.0, 0.0, 0.0);
      normal_cnt.emplace_back(0);
    } else if (prefix == "f") {
      for (int i = 0; i < 3; ++i) {
        std::string data;
        stream >> data;
        size_t pos = data.find("//");
        if (pos == std::string::npos) {
          int idx = std::stoi(data) - 1;
          mesh->indices.emplace_back(idx);
          continue;
        }
        int idx = std::stoi(data.substr(0, pos)) - 1;
        int normal_idx = std::stoi(data.substr(pos + 2, data.length())) - 1;
        mesh->indices.emplace_back(idx);
        mesh->normals[idx] += normals[normal_idx];
        normal_cnt[idx]++;
      }
    } else if (prefix == "vn") {
      double x, y, z;
      stream >> x >> y >> z;
      normals.emplace_back(make_double3(x, y, z));
    }
  }
  assert(mesh->indices.size() % 3 == 0);
  mesh->triangleCount = mesh->indices.size() / 3;
  for (int i = 0; i < mesh->normals.size(); i++) {
    if (normal_cnt[i] == 0)
      continue;
    mesh->normals[i] /= normal_cnt[i];
    mesh->normals[i] = normalize(mesh->normals[i]);
  }
  file.close();
  return true;
}

}