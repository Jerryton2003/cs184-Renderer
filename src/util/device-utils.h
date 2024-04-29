//
// Created by creeper on 3/3/24.
//

#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "properties.h"

#define CUDA_CALLABLE __host__ __device__
#define CUDA_FORCEINLINE __forceinline__
#define CUDA_INLINE __inline__
#define CUDA_SHARED __shared__

#ifndef NDEBUG
#define cudaSafeCheck(kernel) do { \
  kernel;                          \
  cudaDeviceSynchronize();         \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    assert(false); \
  } \
} while (0)
#else
#define cudaSafeCheck(kernel) kernel
#endif
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#define CUDA_CONSTANT __constant__

#define ktid(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)
#define get_tid(tid) int tid = ktid(x)
#define get_and_restrict_tid(tid, max) int tid = ktid(x); do { if (tid >= max) return; } while (0)
#define cuExit() asm("exit;")

template<typename T>
struct best_return_type_for_const {
  using const_ref = std::add_lvalue_reference_t<std::add_const_t<T>>;
  using type = std::conditional_t<std::is_trivially_copyable_v<T>, T, const_ref>;
};

template<typename T>
using best_return_type_for_const_t = typename best_return_type_for_const<T>::type;

template<typename T>
requires std::is_pointer_v<T>
bool checkDevicePtr(T arg) {
  cudaPointerAttributes attr{};
  cudaSafeCheck(cudaPointerGetAttributes(&attr, arg));
  return attr.devicePointer != nullptr;
}

static constexpr int kThreadBlockSize = 128;
static constexpr int kWarpSize = 32;
#define LAUNCH_THREADS(x) ((x) + kThreadBlockSize - 1) / kThreadBlockSize, kThreadBlockSize

#define CUDA_LAMBDA(...) [=] CUDA_DEVICE (__VA_ARGS__)

template<typename T>
struct DeviceAutoPtr {
  T *ptr{};

  DeviceAutoPtr() = default;

  explicit DeviceAutoPtr(T *ptr_) {
    assert(checkDevicePtr(ptr_));
    ptr = ptr_;
  }

  DeviceAutoPtr(const DeviceAutoPtr &) = delete;

  DeviceAutoPtr &operator=(const DeviceAutoPtr &) = delete;

  DeviceAutoPtr(DeviceAutoPtr &&other) noexcept {
    ptr = other.ptr;
    other.ptr = nullptr;
  }

  DeviceAutoPtr &operator=(DeviceAutoPtr &&other) noexcept {
    if (this != &other) {
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  T *get() const {
    return ptr;
  }

  ~DeviceAutoPtr() {
    if (ptr) {
      assert(checkDevicePtr(ptr));
      cudaFree(ptr);
    }
  }
};

template<typename T>
DeviceAutoPtr<T> makeDeviceAutoPtr() {
  T *ptr{};
  T object{};
  cudaMalloc(&ptr, sizeof(T));
  cudaMemcpy(ptr, &object, sizeof(T), cudaMemcpyHostToDevice);
  return DeviceAutoPtr<T>(ptr);
}

template<typename T, typename... Args>
DeviceAutoPtr<T> makeDeviceAutoPtr(Args &&... args) {
  T *ptr{};
  T object(std::forward<Args>(args)...);
  cudaMalloc(&ptr, sizeof(T));
  cudaMemcpy(ptr, &object, sizeof(T), cudaMemcpyHostToDevice);
  return DeviceAutoPtr<T>(ptr);
}

inline void cudaWait() {
  cudaDeviceSynchronize();
}

struct CudaStream : CGL::NonCopyable {
  cudaStream_t stream{};

  CudaStream() {
    cudaStreamCreate(&stream);
  }

  CudaStream(CudaStream &&other) noexcept {
    stream = other.stream;
    other.stream = nullptr;
  }

  CudaStream &operator=(CudaStream &&other) noexcept {
    if (this != &other) {
      stream = other.stream;
      other.stream = nullptr;
    }
    return *this;
  }

  [[nodiscard]] cudaStream_t get() const {
    return stream;
  }

  void wait() const {
    cudaStreamSynchronize(stream);
  }

  ~CudaStream() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};

#endif //DEVICE_UTILS_H
