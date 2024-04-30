//
// Created by creeper on 23-9-1.
//

#ifndef UTILS_GPU_ARRAYS_H_
#define UTILS_GPU_ARRAYS_H_

#include "properties.h"
#include "device-utils.h"
#include <cuda_runtime.h>

namespace CGL {
template<typename T>
struct DeviceArrayAccessor;

template<typename T>
struct ConstDeviceArrayAccessor;

template<typename T>
class DeviceArray : NonCopyable {
 public:
  DeviceArray() = default;

  CUDA_CALLABLE DeviceArray &operator=(DeviceArray &&other) noexcept {
    if (this != &other) {
      m_size = other.m_size;
      ptr = other.ptr;
      other.ptr = nullptr;
      other.m_size = 0;
    }
    return *this;
  }

  CUDA_CALLABLE DeviceArray(DeviceArray &&other) noexcept
      : m_size(other.m_size), ptr(other.ptr) {
    other.ptr = nullptr;
    other.m_size = 0;
  }
  // constructor
  CUDA_CALLABLE explicit DeviceArray(size_t size_) : m_size(size_) {
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
  }

  // construct from vector
  CUDA_HOST
  explicit DeviceArray(const std::vector<T> &vec)
      : m_size(vec.size()) {
    // cudaMalloc takes 2 arguments
    if (vec.empty())
      return ;
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
    cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  CUDA_HOST explicit DeviceArray(std::vector<T> &&vec)
      : m_size(vec.size()) {
    if (vec.empty())
        return ;
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
    cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
    vec.clear();
  }

  // construct from std::array
  template<size_t N>
  explicit DeviceArray(const std::array<T, N> &arr)
      : m_size(N) {
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
    cudaSafeCheck(cudaMemcpy(ptr, arr.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  CUDA_CALLABLE ~DeviceArray() {
    cudaFree(ptr);
  }

  [[nodiscard]] T *data() const { return ptr; }

  [[nodiscard]] size_t size() const { return m_size; }

  void copyFrom(const T *data) {
    cudaSafeCheck(cudaMemcpy(ptr, data, m_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyFrom(const std::vector<T> &vec) {
    if (vec.empty()) {
      if (ptr)
        cudaSafeCheck(cudaFree(ptr));
      ptr = nullptr;
      m_size = 0;
      return ;
    }
    if (m_size == vec.size()) {
      cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
      return ;
    }
    m_size = vec.size();
    if (ptr)
      cudaSafeCheck(cudaFree(ptr));
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
    cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void moveFrom(std::vector<T>&& vec) {
    if (vec.empty()) {
      if (ptr)
        cudaSafeCheck(cudaFree(ptr));
      ptr = nullptr;
      m_size = 0;
      return ;
    }
    if (m_size == vec.size()) {
      cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
      return ;
    }
    m_size = vec.size();
    if (ptr)
      cudaSafeCheck(cudaFree(ptr));
    cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
    cudaSafeCheck(cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
    vec.clear();
  }

  void copyTo(T *data) const {
    cudaSafeCheck(cudaMemcpy(data, ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copyTo(std::vector<T> &vec) const {
    vec.resize(m_size);
    cudaSafeCheck(cudaMemcpy(vec.data(), ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void resize(size_t size_) {
    if (m_size == size_)
      return;
    if (ptr)
      cudaSafeCheck(cudaFree(ptr));
    m_size = size_;
    if (!size_)
      ptr = nullptr;
    else
      cudaSafeCheck(cudaMalloc(&ptr, m_size * sizeof(T)));
  }

  [[nodiscard]] DeviceArrayAccessor<T> accessor() const {
    return {ptr};
  }

  [[nodiscard]] ConstDeviceArrayAccessor<T> constAccessor() const {
    return {ptr};
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  const T &operator[](size_t idx) const { return ptr[idx]; }

  CUDA_CALLABLE CUDA_FORCEINLINE
  T &operator[](size_t idx) { return ptr[idx]; }

  T *begin() { return ptr; }

  T *end() { return ptr + m_size; }
 private:
  uint32_t m_size{};
  T *ptr{};
};

template <typename T>
void printDeviceArray(const DeviceArray<int> &arr) {
    std::vector<T> vec;
    arr.copyTo(vec);
    for (const auto &v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

template<typename T>
struct DeviceArrayAccessor {
  T *ptr;
  CUDA_DEVICE CUDA_FORCEINLINE T &operator[](size_t idx) { return ptr[idx]; }

  CUDA_DEVICE CUDA_FORCEINLINE best_return_type_for_const_t<T> operator[](size_t idx) const { return ptr[idx]; }
};

template<typename T>
struct ConstDeviceArrayAccessor {
  const T *ptr;
  CUDA_DEVICE CUDA_FORCEINLINE best_return_type_for_const_t<T> operator[](size_t idx) const { return ptr[idx]; }
};


} // namespace CGL
#endif // UTILS_GPU_ARRAYS_H_
