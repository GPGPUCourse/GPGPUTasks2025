#ifndef common_cu // pragma once
#define common_cu

#include <libgpu/cuda/utils.h>
#include <libgpu/work_size.h>
#include <math.h>

using gpu::WorkSize;

//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define printf_assert(condition, message) \
        if (!(condition)) printf("%s Line %d\n", message, __LINE__);
#else
#define printf_assert(condition, message)
#endif
#define assert_isfinite(value) \
        printf_assert(isfinite(value), "Value should be finite");

#define WARP_SIZE 32 // TODO: WHY WARP_SZ IS UNDEFINED?

#ifndef M_PI_F
#define M_PI_F 3.141592654f
#endif

namespace cuda {

	template <typename T>
	__device__ T max(T a, T b) {
		return a > b ? a : b;
	}

	template <typename T>
	__device__ T max(T a, T b, T c) {
		return max(a, max(b, c));
	}

	template <typename T>
	__device__ T min(T a, T b) {
		return a < b ? a : b;
	}

	template <typename T>
	__device__ T min(T a, T b, T c) {
		return min(a, min(b, c));
	}

	inline __device__ uint3 fetch_uint3(const unsigned int* ptr, size_t index)
	{
		return make_uint3(ptr[3 * index + 0], ptr[3 * index + 1], ptr[3 * index + 2]);
	}

	inline __device__ uint4 fetch_uint4(const unsigned int* ptr, size_t index)
	{
		return make_uint4(ptr[4 * index + 0], ptr[4 * index + 1], ptr[4 * index + 2], ptr[4 * index + 3]);
	}

	inline __device__ float2 fetch_float2(const float* ptr, size_t index)
	{
		return make_float2(ptr[2 * index + 0], ptr[2 * index + 1]);
	}

	inline __device__ float3 fetch_float3(const float* ptr, size_t index)
	{
		return make_float3(ptr[3 * index + 0], ptr[3 * index + 1], ptr[3 * index + 2]);
	}

	inline __device__ float4 fetch_float4(const float* ptr, size_t index)
	{
		return make_float4(ptr[4 * index + 0], ptr[4 * index + 1], ptr[4 * index + 2], ptr[4 * index + 3]);
	}

	inline __device__ void set_uint3(unsigned int* ptr, size_t index, uint3 value)
	{
		ptr[3 * index + 0] = value.x;
		ptr[3 * index + 1] = value.y;
		ptr[3 * index + 2] = value.z;
	}

	inline __device__ void set_float2(float* ptr, size_t index, float2 value)
	{
		ptr[2 * index + 0] = value.x;
		ptr[2 * index + 1] = value.y;
	}

	inline __device__ void set_float3(float* ptr, size_t index, float3 value)
	{
		ptr[3 * index + 0] = value.x;
		ptr[3 * index + 1] = value.y;
		ptr[3 * index + 2] = value.z;
	}

	inline __device__ __host__ float clamp(float f, float a, float b)
	{
		return fmaxf(a, fminf(f, b));
	}

	inline __device__ float atomicCAS_f32(float *p, float cmp, float val) {
		return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
	}

	inline __device__ float atomicCAS_32(float* p, float cmp, float val) {
		return atomicCAS_f32(p, cmp, val);
	}

	inline __device__ unsigned int atomicCAS_32(unsigned int* p, unsigned int cmp, unsigned int val) {
		return atomicCAS(p, cmp, val);
	}

	inline __device__ float2 operator-(const float2 &a)
	{
		return make_float2(-a.x, -a.y);
	}

	inline __device__ float3 operator-(const float3 &a)
	{
		return make_float3(-a.x, -a.y, -a.z);
	}

	inline __device__ float2 operator-(float2 a, float2 b)
	{
		return make_float2(a.x - b.x, a.y - b.y);
	}

	inline __device__ void operator-=(float2 &a, float2 b)
	{
		a.x -= b.x;
		a.y -= b.y;
	}

	inline __device__ float2 operator+(float2 a, float2 b)
	{
		return make_float2(a.x + b.x, a.y + b.y);
	}

	inline __device__ void operator+=(float2 &a, float2 b)
	{
		a.x += b.x;
		a.y += b.y;
	}

	inline __device__ float3 operator-(float3 a, float3 b)
	{
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	inline __device__ float3 operator+(float3 a, float3 b)
	{
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	inline __device__ void operator+=(float3 &a, float3 b)
	{
		a.x += b.x;
		a.y += b.y;
		a.z += b.z;
	}

	inline __device__ float4 operator+(float4 a, float4 b)
	{
		return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}

	inline __device__ float2 operator*(float2 a, float b)
	{
		return make_float2(a.x * b, a.y * b);
	}

	inline __device__ float3 operator*(float3 a, float b)
	{
		return make_float3(a.x * b, a.y * b, a.z * b);
	}

	inline __device__ float3 operator*(float a, float3 b)
	{
		return make_float3(b.x * a, b.y * a, b.z * a);
	}

	inline __device__ float4 operator*(float4 a, float b)
	{
		return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
	}

	inline __device__ float2 operator/(float2 a, float b)
	{
		return make_float2(a.x / b, a.y / b);
	}

	inline __device__ float3 operator/(float3 a, float b)
	{
		return make_float3(a.x / b, a.y / b, a.z / b);
	}

	inline __device__ void operator/=(float3& a, float b)
	{
		a.x /= b;
		a.y /= b;
		a.z /= b;
	}

	inline __device__ float3 operator*(float3 a, float3 b)
	{
		return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
	}

	inline __device__ void operator*=(float3 &a, float b)
	{
		a.x *= b;
		a.y *= b;
		a.z *= b;
	}

	inline __device__ bool operator==(float3 a, float3 b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}

	inline __device__ bool operator==(float4 a, float4 b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}

	inline __device__ bool operator!=(float3 a, float3 b)
	{
		return !(a == b);
	}

	inline __device__ bool operator!=(float4 a, float4 b)
	{
		return !(a == b);
	}

	inline __device__ bool is_zero(float3 v)
	{
		return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
	}

	inline __device__ float dot(float2 a, float2 b)
	{
		return a.x * b.x + a.y * b.y;
	}

	inline __device__ float dot(float3 a, float3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	inline __device__ float dot(float4 a, float4 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	inline __device__ float3 cross(float3 a, float3 b)
	{
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}

	inline __device__ float norm2(float2 v)
	{
		return dot(v, v);
	}

	inline __device__ float norm2(float3 v)
	{
		return dot(v, v);
	}

	inline __device__ float norm2(float4 v)
	{
		return dot(v, v);
	}

	inline __device__ float norm(float2 v)
	{
		return sqrtf(norm2(v));
	}

	inline __device__ float norm(float3 v)
	{
		return sqrtf(norm2(v));
	}

	inline __device__ float norm(float4 v)
	{
		return sqrtf(norm2(v));
	}

	inline __device__ float3 normalize(float3 v)
	{
		return v / sqrtf(dot(v, v));
	}

//______SHARED_STRUCTS__________________________________________________________________________________________________

	// https://devtalk.nvidia.com/default/topic/673965/are-there-any-cuda-libararies-for-3x3-matrix-amp-vector3-amp-quaternion-operations-/
	typedef struct {
		float4 m_row[3];
	} Matrix3x3f;

	typedef struct {
		float4 m_row[4];
	} Matrix4x4f;

//______HOST_CODE_______________________________________________________________________________________________________

//______DEVICE_CODE_____________________________________________________________________________________________________

#ifdef DEBUG
	inline __device__ void print_matrix_f3x3(const Matrix3x3f &m)
	{
		printf("[\n");
		printf("  [%f, %f, %f],\n", m.m_row[0].x, m.m_row[0].y, m.m_row[0].z);
		printf("  [%f, %f, %f],\n", m.m_row[1].x, m.m_row[1].y, m.m_row[1].z);
		printf("  [%f, %f, %f],\n", m.m_row[2].x, m.m_row[2].y, m.m_row[2].z);
		printf("]\n");
	}
#endif

	inline __device__ Matrix3x3f make_matrix_f3x3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(a00, a01, a02, 0.0f);
		m.m_row[1] = make_float4(a10, a11, a12, 0.0f);
		m.m_row[2] = make_float4(a20, a21, a22, 0.0f);
		return m;
	}

	inline __device__ Matrix3x3f make_zero_matrix_f3x3()
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		return m;
	}

	inline __device__ Matrix3x3f make_eye_matrix_f3x3()
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
		m.m_row[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
		return m;
	}

	inline __device__ Matrix3x3f transpose_f3x3(const Matrix3x3f &m)
	{
		Matrix3x3f t;
		t.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.0f);
		t.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.0f);
		t.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.0f);
		return t;
	}

	inline __device__ Matrix3x3f add_f3x3(const Matrix3x3f& a, const Matrix3x3f& b)
	{
		Matrix3x3f m;
		m.m_row[0] = a.m_row[0] + b.m_row[0];
		m.m_row[1] = a.m_row[1] + b.m_row[1];
		m.m_row[2] = a.m_row[2] + b.m_row[2];
		return m;
	}

	inline __device__ Matrix3x3f mul_f3x3(const Matrix3x3f &a, const Matrix3x3f &b)
	{
		Matrix3x3f bt = transpose_f3x3(b);
		Matrix3x3f res;
		res.m_row[0] = make_float4(dot(a.m_row[0], bt.m_row[0]), dot(a.m_row[0], bt.m_row[1]), dot(a.m_row[0], bt.m_row[2]), 0.0f);
		res.m_row[1] = make_float4(dot(a.m_row[1], bt.m_row[0]), dot(a.m_row[1], bt.m_row[1]), dot(a.m_row[1], bt.m_row[2]), 0.0f);
		res.m_row[2] = make_float4(dot(a.m_row[2], bt.m_row[0]), dot(a.m_row[2], bt.m_row[1]), dot(a.m_row[2], bt.m_row[2]), 0.0f);
		return res;
	}

	inline __device__ Matrix3x3f mul_f_f3x3(float k, const Matrix3x3f& a)
	{
		Matrix3x3f m;
		m.m_row[0] = a.m_row[0] * k;
		m.m_row[1] = a.m_row[1] * k;
		m.m_row[2] = a.m_row[2] * k;
		return m;
	}

	inline __device__ float3 mul_f3x3_f3(const Matrix3x3f& a, const float3 &b)
	{
		return make_float3(a.m_row[0].x * b.x + a.m_row[0].y * b.y + a.m_row[0].z * b.z,
						   a.m_row[1].x * b.x + a.m_row[1].y * b.y + a.m_row[1].z * b.z,
						   a.m_row[2].x * b.x + a.m_row[2].y * b.y + a.m_row[2].z * b.z);
	}

	inline __device__ float2 transformPoint_f3x3(const Matrix3x3f &m, const float2 &p)
	{
		float3 temp = mul_f3x3_f3(m, make_float3(p.x, p.y, 1.0f));
		return make_float2(temp.x, temp.y) / temp.z;
	}

#ifdef DEBUG
	inline __device__ void print_matrix_f4x4(const Matrix4x4f &m)
	{
		printf("[\n");
		printf("  [%f, %f, %f, %f],\n", m.m_row[0].x, m.m_row[0].y, m.m_row[0].z, m.m_row[0].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[1].x, m.m_row[1].y, m.m_row[1].z, m.m_row[1].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[2].x, m.m_row[2].y, m.m_row[2].z, m.m_row[2].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[3].x, m.m_row[3].y, m.m_row[3].z, m.m_row[3].w);
		printf("]\n");
	}
#endif

	inline __device__ Matrix4x4f make_matrix_f4x4(float a00, float a01, float a02, float a03,
												  float a10, float a11, float a12, float a13,
												  float a20, float a21, float a22, float a23,
												  float a30, float a31, float a32, float a33)
	{
		Matrix4x4f m;
		m.m_row[0] = make_float4(a00, a01, a02, a03);
		m.m_row[1] = make_float4(a10, a11, a12, a13);
		m.m_row[2] = make_float4(a20, a21, a22, a23);
		m.m_row[3] = make_float4(a30, a31, a32, a33);
		return m;
	}

	inline __device__ Matrix4x4f make_translation_f4x4(const float3 &t)
	{
		return make_matrix_f4x4(1.0f, 0.0f, 0.0f, t.x,
								0.0f, 1.0f, 0.0f, t.y,
								0.0f, 0.0f, 1.0f, t.z,
								0.0f, 0.0f, 0.0f, 1.0f);
	}

	inline __device__ Matrix4x4f make_rotation_f4x4(const Matrix3x3f &r)
	{
		Matrix4x4f m;
		m.m_row[0] = r.m_row[0];
		m.m_row[1] = r.m_row[1];
		m.m_row[2] = r.m_row[2];

		m.m_row[0].w = 0.0f;
		m.m_row[1].w = 0.0f;
		m.m_row[2].w = 0.0f;
		m.m_row[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		return m;
	}

	inline __device__ float3 extract_translation_f4x4(const Matrix4x4f &m)
	{
		float norm = 1.0f / m.m_row[3].w;
		return make_float3(m.m_row[0].w, m.m_row[1].w, m.m_row[2].w) * norm;
	}

	inline __device__ Matrix3x3f extract_rotation_f4x4(const Matrix4x4f &m)
	{
		Matrix3x3f R = make_matrix_f3x3(
				m.m_row[0].x, m.m_row[0].y, m.m_row[0].z,
				m.m_row[1].x, m.m_row[1].y, m.m_row[1].z,
				m.m_row[2].x, m.m_row[2].y, m.m_row[2].z);

		// matrix4x4f.scale3()
		Matrix3x3f MtM = mul_f3x3(transpose_f3x3(R), R);

		float3 d = make_float3(MtM.m_row[0].x, MtM.m_row[1].y, MtM.m_row[2].z);

		if (d.x > 0) d.x = sqrtf(d.x);
		if (d.y > 0) d.y = sqrtf(d.y);
		if (d.z > 0) d.z = sqrtf(d.z);

		float3 s = d;

		if (s.x) s.x = 1.0f / s.x;
		if (s.y) s.y = 1.0f / s.y;
		if (s.z) s.z = 1.0f / s.z;

		return mul_f3x3(R, make_matrix_f3x3(s.x, 0.0f, 0.0f,
											0.0f, s.y, 0.0f,
											0.0f, 0.0f, s.z));
	}

	inline __device__ Matrix4x4f transpose_f4x4(const Matrix4x4f &m)
	{
		Matrix4x4f t;
		t.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, m.m_row[3].x);
		t.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, m.m_row[3].y);
		t.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, m.m_row[3].z);
		t.m_row[3] = make_float4(m.m_row[0].w, m.m_row[1].w, m.m_row[2].w, m.m_row[3].w);
		return t;
	}

	inline __device__ Matrix4x4f mul_f4x4(const Matrix4x4f &a, const Matrix4x4f &b)
	{
		Matrix4x4f bt = transpose_f4x4(b);
		Matrix4x4f res;
		res.m_row[0] = make_float4(dot(a.m_row[0], bt.m_row[0]), dot(a.m_row[0], bt.m_row[1]), dot(a.m_row[0], bt.m_row[2]), dot(a.m_row[0], bt.m_row[3]));
		res.m_row[1] = make_float4(dot(a.m_row[1], bt.m_row[0]), dot(a.m_row[1], bt.m_row[1]), dot(a.m_row[1], bt.m_row[2]), dot(a.m_row[1], bt.m_row[3]));
		res.m_row[2] = make_float4(dot(a.m_row[2], bt.m_row[0]), dot(a.m_row[2], bt.m_row[1]), dot(a.m_row[2], bt.m_row[2]), dot(a.m_row[2], bt.m_row[3]));
		res.m_row[3] = make_float4(dot(a.m_row[3], bt.m_row[0]), dot(a.m_row[3], bt.m_row[1]), dot(a.m_row[3], bt.m_row[2]), dot(a.m_row[3], bt.m_row[3]));
		return res;
	}

	inline __device__ float4 mul_f4x4_f4(const Matrix4x4f& a, const float4 &b)
	{
		return make_float4(a.m_row[0].x * b.x + a.m_row[0].y * b.y + a.m_row[0].z * b.z + a.m_row[0].w * b.w,
						   a.m_row[1].x * b.x + a.m_row[1].y * b.y + a.m_row[1].z * b.z + a.m_row[1].w * b.w,
						   a.m_row[2].x * b.x + a.m_row[2].y * b.y + a.m_row[2].z * b.z + a.m_row[2].w * b.w,
						   a.m_row[3].x * b.x + a.m_row[3].y * b.y + a.m_row[3].z * b.z + a.m_row[3].w * b.w);
	}

	inline __device__ float3 transformPoint(const Matrix4x4f &m, const float3 &p)
	{
		float4 temp = mul_f4x4_f4(m, make_float4(p.x, p.y, p.z, 1.0f));
		return make_float3(temp.x, temp.y, temp.z) / temp.w;
	}

	inline __device__ float3 transformVector(const Matrix4x4f &m, const float3 &v)
	{
		float4 temp = mul_f4x4_f4(m, make_float4(v.x, v.y, v.z, 0.0f));
		return make_float3(temp.x, temp.y, temp.z);
	}

	inline __device__ float smootherstep(float edge0, float edge1, float x)
	{
		if (x < edge0) {
			return 0.0f;
		} else if (x >= edge1) {
			return 1.0f;
		}

		// Scale, and clamp x to 0..1 range
		x = (x - edge0) / (edge1 - edge0);
		// Evaluate polynomial
		return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
	}

}

#endif // pragma once