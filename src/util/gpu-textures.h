//
// Created by creeper on 4/20/24.
//

#ifndef ASSIGNMENT3_SRC_UTIL_GPU_TEXTURES_H_
#define ASSIGNMENT3_SRC_UTIL_GPU_TEXTURES_H_
#include "device-utils.h"
#include "properties.h"
namespace CGL {
template<typename T>
struct CudaArray3D : NonCopyable {
  cudaArray *cuda_array{};
  uint3 dim;

  explicit CudaArray3D(const uint3 &dim_)
      : dim(dim_) {
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(&cuda_array,
                      &channel_desc,
                      extent,
                      cudaArraySurfaceLoadStore);
  }

  void copyFrom(const T *data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcPtr =
        make_cudaPitchedPtr(static_cast<void *>(data),
                            dim.x * sizeof(T),
                            dim.x,
                            dim.y);
    copy3DParams.dstArray = cuda_array;
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy3DParams);
  }
  void copyFrom(std::vector<T> &data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcPtr =
        make_cudaPitchedPtr(static_cast<void *>(data.data()),
                            dim.x * sizeof(T),
                            dim.x,
                            dim.y);
    copy3DParams.dstArray = cuda_array;
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy3DParams);
  }

  void copyTo(T *data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcArray = cuda_array;
    copy3DParams.dstPtr =
        make_cudaPitchedPtr(static_cast<void *>(data),
                            dim.x * sizeof(T),
                            dim.x,
                            dim.y);
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copy3DParams);
  }

  [[nodiscard]] cudaArray *Array() const { return cuda_array; }

  cudaArray_t *ArrayPtr() { return &cuda_array; }

  ~CudaArray3D() { cudaFreeArray(cuda_array); }
};
template<typename T>
struct CudaSurfaceAccessor {
  cudaSurfaceObject_t cuda_surf;

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE T read(int x, int y, int z) {
    return surf3Dread<T>(cuda_surf, x * sizeof(T), y, z, mode);
  }

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE T read(const int3 &idx) {
    return surf3Dread<T>(cuda_surf, idx.x * sizeof(T), idx.y, idx.z, mode);
  }

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE void write(T val, int x, int y, int z) {
    surf3Dwrite<T>(val, cuda_surf, x * sizeof(T), y, z, mode);
  }
};

template<typename T>
struct CudaSurface : CudaArray3D<T> {
  cudaSurfaceObject_t cuda_surf{};

  explicit CudaSurface(const uint3 &dim_)
      : CudaArray3D<T>(dim_) {
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = CudaArray3D<T>::Array();
    cudaCreateSurfaceObject(&cuda_surf, &res_desc);
  }

  [[nodiscard]] cudaSurfaceObject_t surface() const { return cuda_surf; }

  CudaSurfaceAccessor<T> surfaceAccessor() const { return {cuda_surf}; }
  ~CudaSurface() { cudaDestroySurfaceObject(cuda_surf); }
};

template<class T>
struct CudaTextureAccessor {
  cudaTextureObject_t m_cuTex;
  __device__ __forceinline__ T sample(double x, double y, double z) const {
    return tex3D<T>(m_cuTex, static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(z));
  }
  __device__ __forceinline__ T sample(const double3 &pos) const {
    return tex3D<T>(m_cuTex, static_cast<float>(pos.x), static_cast<float>(pos.y),
                    static_cast<float>(pos.z));
  }
  __device__ __forceinline__ T sample(float x, float y, float z) const {
    return tex3D<T>(m_cuTex, x, y, z);
  }
  __device__ __forceinline__ T sample(const float3 &pos) const {
    return tex3D<T>(m_cuTex, pos.x, pos.y, pos.z);
  }
};

template<class T>
struct CudaTexture : CudaSurface<T> {
  struct Parameters {
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
    cudaTextureFilterMode filterMode{cudaFilterModeLinear};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    bool normalizedCoords{false};
  };

  cudaTextureObject_t cuda_tex{};

  explicit CudaTexture(uint3 const &_dim, Parameters const &_args = {})
      : CudaSurface<T>(_dim) {
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = CudaSurface<T>::Array();

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = _args.addressMode;
    texDesc.addressMode[1] = _args.addressMode;
    texDesc.addressMode[2] = _args.addressMode;
    texDesc.filterMode = _args.filterMode;
    texDesc.readMode = _args.readMode;
    texDesc.normalizedCoords = _args.normalizedCoords;

    cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr);
  }

  [[nodiscard]] cudaTextureObject_t texture() const { return cuda_tex; }

  CudaTextureAccessor<T> texAccessor() const { return {cuda_tex}; }

  ~CudaTexture() { cudaDestroyTextureObject(cuda_tex); }
};
}
#endif //ASSIGNMENT3_SRC_UTIL_GPU_TEXTURES_H_
