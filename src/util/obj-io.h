//
// Created by creeper on 4/30/24.
//

#ifndef SOSONOWPT_SRC_UTIL_OBJ_IO_H_
#define SOSONOWPT_SRC_UTIL_OBJ_IO_H_
#include <vector>
#include <cuda_runtime.h>
namespace CGL {
struct ObjMesh {
  std::vector<double3> vertices;
  std::vector<double3> normals;
  std::vector<int> indices;
  int triangleCount;
};
bool myLoadObj(const std::string& path, ObjMesh* mesh);
}
#endif //SOSONOWPT_SRC_UTIL_OBJ_IO_H_
