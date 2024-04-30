//
// Created by creeper on 3/4/24.
//

#ifndef DEVICE_VEC_OPS_H
#define DEVICE_VEC_OPS_H

#include <cuda_runtime.h>
#include <CGL/matrix3x3.h>
#include <CGL/matrix4x4.h>
#include "device-utils.h"

namespace CGL {
class Mat3 {
 public:
  // The default constructor. Returns identity.
  CUDA_CALLABLE CUDA_FORCEINLINE
  Mat3() {
    entries[0] = make_double3(1, 0, 0);
    entries[1] = make_double3(0, 1, 0);
    entries[2] = make_double3(0, 0, 1);
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  Mat3(double m00,
       double m01,
       double m02,
       double m10,
       double m11,
       double m12,
       double m20,
       double m21,
       double m22) {
    entries[0] = make_double3(m00, m01, m02);
    entries[1] = make_double3(m10, m11, m12);
    entries[2] = make_double3(m20, m21, m22);
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  const double3 &column(int i) const {
    return entries[i];
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double3 &column(int i) {
    return entries[i];
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  const double3 &operator[](int i) const {
    return entries[i];
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double3 &operator[](int i) {
    return entries[i];
  }

  CUDA_CALLABLE CUDA_FORCEINLINE
  double operator()(int i, int j) const {
    if (i == 0) return entries[j].x;
    if (i == 1) return entries[j].y;
    return entries[j].z;
  }
  CUDA_CALLABLE CUDA_FORCEINLINE
  double& operator()(int i, int j) {
    if (i == 0) return entries[j].x;
    if (i == 1) return entries[j].y;
    return entries[j].z;
  }
  double3 entries[3];
}; // class Matrix3x3

CUDA_HOST CUDA_FORCEINLINE
double3 to_double3(const Vector3D &v) {
  return make_double3(v.x, v.y, v.z);
}

CUDA_HOST CUDA_FORCEINLINE
double4 to_double4(const Vector4D &v) {
  return make_double4(v.x, v.y, v.z, v.w);
}

CUDA_HOST CUDA_FORCEINLINE
Vector3D to_Vector3D(const double3 &v) {
  return {v.x, v.y, v.z};
}

CUDA_HOST CUDA_FORCEINLINE
Vector4D to_Vector4D(const double4 &v) {
  return {v.x, v.y, v.z, v.w};
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator+(const double4 &a, const double4 &b) {
  return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator-(const double4 &a, const double4 &b) {
  return make_double4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator*(const double4 &a, const double4 &b) {
  return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator/(const double4 &a, const double4 &b) {
  return make_double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator*(const double4 &a, double b) {
  return make_double4(a.x * b, a.y * b, a.z * b, a.w * b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator/(const double4 &a, double b) {
  return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double4 operator/(double a, const double4 &b) {
  return make_double4(a / b.x, a / b.y, a / b.z, a / b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator+(const double3 &a, const double3 &b) {
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator-(const double3 &a, const double3 &b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator*(const double3 &a, const double3 &b) {
  return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator/(const double3 &a, const double3 &b) {
  return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator*(const double3 &a, double b) {
  return make_double3(a.x * b, a.y * b, a.z * b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator/(const double3 &a, double b) {
  return make_double3(a.x / b, a.y / b, a.z / b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator/(double a, const double3 &b) {
  return make_double3(a / b.x, a / b.y, a / b.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator*(double b, const double3 &a) {
  return make_double3(a.x * b, a.y * b, a.z * b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double2 operator-(const double2 &a, const double2 &b) {
  return make_double2(a.x - b.x, a.y - b.y);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator-(const double3 &a) {
  return make_double3(-a.x, -a.y, -a.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 cross(const double3 &a, const double3 &b) {
  return make_double3(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double dot(const double4 &a, const double4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double length(const double3 &a) {
  return sqrt(dot(a, a));
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 normalize(const double3 &a) {
  if (a.x == 0 && a.y == 0 && a.z == 0) return a;
  return a / length(a);
}

CUDA_HOST inline Mat3 to_Mat3(const Matrix3x3 &mat) {
  return {mat(0, 0), mat(0, 1), mat(0, 2),
          mat(1, 0), mat(1, 1), mat(1, 2),
          mat(2, 0), mat(2, 1), mat(2, 2)};
}

// matrix multiplication
CUDA_CALLABLE CUDA_FORCEINLINE
double3 operator*(const Mat3 &m, const double3 &v) {
  return make_double3(m.column(0).x * v.x + m.column(1).x * v.y + m.column(2).x * v.z,
                      m.column(0).y * v.x + m.column(1).y * v.y + m.column(2).y * v.z,
                      m.column(0).z * v.x + m.column(1).z * v.y + m.column(2).z * v.z);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator/=(double3 &a, double b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator*=(double3 &a, double b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator+=(double3 &a, const double3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator-=(double3 &a, const double3 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator*=(double3 &a, const double3 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 &operator/=(double3 &a, const double3 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE
float3 &operator/=(float3 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  return a;
}

CUDA_CALLABLE CUDA_FORCEINLINE double illum(float3 rgb) {
  return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

CUDA_CALLABLE CUDA_FORCEINLINE double illum(const double3& rgb) {
  return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

CUDA_CALLABLE CUDA_FORCEINLINE uint32_t to_rgba(const double3 &rgb) {
  double r = fmin(255.0, fmax(0.0, 255.0 * rgb.x));
  double g = fmin(255.0, fmax(0.0, 255.0 * rgb.y));
  double b = fmin(255.0, fmax(0.0, 255.0 * rgb.z));
  return (static_cast<uint32_t>(r)) | (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b) << 16) | 0xff000000;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double distance(const double3 &a, const double3 &b) {
  return length(a - b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double distSqr(const double3 &a, const double3 &b) {
  return dot(a - b, a - b);
}

CUDA_CALLABLE CUDA_FORCEINLINE
double3 reflect(const double3 &d, const double3 &n) {
  return d - 2 * dot(d, n) * n;
}

CUDA_CALLABLE CUDA_FORCEINLINE
double det(const Mat3 &A) {
  const auto &A00 = A[0].x;
  const auto &A01 = A[1].x;
  const auto &A02 = A[2].x;
  const auto &A10 = A[0].y;
  const auto &A11 = A[1].y;
  const auto &A12 = A[2].y;
  const auto &A20 = A[0].z;
  const auto &A21 = A[1].z;
  const auto &A22 = A[2].z;
  return A00 * (A11 * A22 - A12 * A21) - A01 * (A10 * A22 - A12 * A20) + A02 * (A10 * A21 - A11 * A20);
}

CUDA_CALLABLE CUDA_FORCEINLINE
Mat3 inv(const Mat3 &A) {
  Mat3 B;
  const auto &A00 = A[0].x;
  const auto &A01 = A[1].x;
  const auto &A02 = A[2].x;
  const auto &A10 = A[0].y;
  const auto &A11 = A[1].y;
  const auto &A12 = A[2].y;
  const auto &A20 = A[0].z;
  const auto &A21 = A[1].z;
  const auto &A22 = A[2].z;
  // column major
  double B00 = -A12 * A21 + A11 * A22;
  double B01 = A02 * A21 - A01 * A22;
  double B02 = -A02 * A11 + A01 * A12;
  double B10 = A12 * A20 - A10 * A22;
  double B11 = -A02 * A20 + A00 * A22;
  double B12 = A02 * A10 - A00 * A12;
  double B20 = -A11 * A20 + A10 * A21;
  double B21 = A01 * A20 - A00 * A21;
  double B22 = -A01 * A10 + A00 * A11;
  double d = det(A);
  double rd = 1.0 / d;
  B[0] = make_double3(B00, B10, B20) * rd;
  B[1] = make_double3(B01, B11, B21) * rd;
  B[2] = make_double3(B02, B12, B22) * rd;
  return B;
}
// operators for float4
CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator+(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator-(const float4 &a, const float4 &b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator*(const float4 &a, const float4 &b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator/(const float4 &a, const float4 &b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator-(const float4 &a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

CUDA_CALLABLE CUDA_FORCEINLINE
float4 operator-(float a, const float4 &b) {
  return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}
CUDA_CALLABLE CUDA_FORCEINLINE
double maxComponent(const double3 &v) {
  return fmax(v.x, fmax(v.y, v.z));
}
CUDA_CALLABLE CUDA_FORCEINLINE
double3 exp(const double3 &v) {
  return make_double3(expf(v.x), expf(v.y), expf(v.z));
}
CUDA_CALLABLE CUDA_FORCEINLINE
double avg(const double3 &v) {
  return (v.x + v.y + v.z) / 3;
}
CUDA_CALLABLE CUDA_FORCEINLINE
double3 make_constant(double v) {
  return make_double3(v, v, v);
}
CUDA_CALLABLE CUDA_FORCEINLINE
float3& operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
CUDA_CALLABLE CUDA_FORCEINLINE
float3& operator-=(float3 &a, const float3 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}
CUDA_CALLABLE CUDA_FORCEINLINE
float3& operator*=(float3 &a, const float3 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  return a;
}
CUDA_CALLABLE CUDA_FORCEINLINE
float3& operator/=(float3 &a, const float3 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  return a;
}
CUDA_CALLABLE CUDA_FORCEINLINE
float3& operator*=(float3 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}
}
#endif //DEVICE_VEC_OPS_H
