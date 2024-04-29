//
// Created by creeper on 3/31/24.
//
#include "image.h"
#include "device-vec-ops.h"
#include "device-utils.h"

namespace CGL {

void HDRImageBuffer::toColor(CGL::ImageBuffer &target) {
  float gamma = 2.2f;
  float level = 1.0f;
  float key = 0.18;
  float wht = 5.0;
  std::vector<float3> tmp(data->size());
  data->copyTo(tmp);
  float average = 0;
   for (size_t i = 0; i < w * h; ++i) {
     // the small delta value below is used to avoids singularity
     average += log(0.0000001f + illum(tmp[i]));
   }
   average = std::exp(average / (w * h));
   // apply on pixels
   float one_over_gamma = 1.0f / gamma;
   float exposure = sqrt(pow(2,level));
   for (size_t y = 0; y < h; ++y) {
     for (size_t x = 0; x < w; ++x) {
       float3 s = tmp[x + y * w];
//       float l = illum(s);
//       s *= key / average;
//       s *= ((l + 1) / (wht * wht)) / (l + 1);
       float r = pow(s.x * exposure, one_over_gamma);
       float g = pow(s.y * exposure, one_over_gamma);
       float b = pow(s.z * exposure, one_over_gamma);
       target.data[x + y * w] = to_rgba(make_double3(r, g, b));
     }
   }
}

static CUDA_GLOBAL void kernelClear(DeviceArrayAccessor<float3> data, int size) {
  get_and_restrict_tid(tid, size);
  data[tid] = make_float3(0, 0, 0);
}

void HDRImageBuffer::clear() const {
  if (data->data() == nullptr) return;
  cudaSafeCheck(kernelClear<<<LAUNCH_THREADS(w * h)>>>(data->accessor(), w * h));
}

}