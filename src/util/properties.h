//
// Created by creeper on 3/3/24.
//

#ifndef PROPERTIES_H
#define PROPERTIES_H

namespace CGL {
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

struct DeviceOnly {
  __device__ DeviceOnly() = default;
};

struct HostOnly {
  __host__ HostOnly() = default;
};

}
#endif //PROPERTIES_H
