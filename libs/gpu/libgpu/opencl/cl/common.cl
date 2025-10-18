#ifndef common_cl // pragma once
#define common_cl

#line 5

#ifdef HOST_CODE
#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl_platform.h>
#else
#include "clion_defines.cl"
#endif

//#define DEBUG

#ifndef HOST_CODE
#ifdef DEBUG
#define printf_assert(condition, message) \
        if (!(condition)) printf("%s Line %d\n", message, __LINE__);
#else
#define printf_assert(condition, message)
#endif

#define assert_isfinite(value) \
        printf_assert(isfinite(value), "Value should be finite");
#endif

#define BOOL_TYPE int
#define BOOL_TRUE 1
#define BOOL_FALSE 0

//______DEVICE_CODE_____________________________________________________________________________________________________

#ifndef HOST_CODE
	#define make_uint3  (uint3)
	#define make_uint4  (uint4)
	#define make_float2 (float2)
	#define make_float3 (float3)
	#define make_float4 (float4)

	#define tanf tan
	#define cosf cos
	#define sinf sin
	#define sincosf sincos
	#define atan2f atan2
	#define atanf atan
	#define asinf asin

	STATIC_KEYWORD float sqrtf(float x)
	{
		return sqrt(x);
	}

	STATIC_KEYWORD float expf(float x)
	{
		return exp(x);
	}

	#define norm(v) length(v)
	#define norm2(v) dot(v, v)

	#define min3(x, y, z) (min(min(x, y), z))
	#define max3(x, y, z) (max(max(x, y), z))

	#define cl_float4 float4
	#define cl_float2 float2

	STATIC_KEYWORD uint3 fetch_uint3(__global const unsigned int* ptr, size_t index)
	{
		return make_uint3(ptr[3 * index + 0], ptr[3 * index + 1], ptr[3 * index + 2]);
	}

	STATIC_KEYWORD uint4 fetch_uint4(__global const unsigned int* ptr, size_t index)
	{
		return make_uint4(ptr[4 * index + 0], ptr[4 * index + 1], ptr[4 * index + 2], ptr[4 * index + 3]);
	}

	STATIC_KEYWORD float2 fetch_float2(__global const float* ptr, size_t index)
	{
		return make_float2(ptr[2 * index + 0], ptr[2 * index + 1]);
	}

	STATIC_KEYWORD float3 fetch_float3(__global const float* ptr, size_t index)
	{
		return make_float3(ptr[3 * index + 0], ptr[3 * index + 1], ptr[3 * index + 2]);
	}

	STATIC_KEYWORD float4 fetch_float4(__global const float* ptr, size_t index)
	{
		return make_float4(ptr[4 * index + 0], ptr[4 * index + 1], ptr[4 * index + 2], ptr[4 * index + 3]);
	}

	STATIC_KEYWORD void set_uint3(__global unsigned int* ptr, size_t index, uint3 value)
	{
		ptr[3 * index + 0] = value.x;
		ptr[3 * index + 1] = value.y;
		ptr[3 * index + 2] = value.z;
	}

	STATIC_KEYWORD void set_float3(__global float* ptr, size_t index, float3 value)
	{
		ptr[3 * index + 0] = value.x;
		ptr[3 * index + 1] = value.y;
		ptr[3 * index + 2] = value.z;
	}

	STATIC_KEYWORD unsigned char rounded_cast_uchar(float value) {
		return (unsigned char) (value + 0.5f);
	}

	STATIC_KEYWORD unsigned short rounded_cast_ushort(float value) {
		return (unsigned short) (value + 0.5f);
	}

	STATIC_KEYWORD unsigned int rounded_cast_uint(float value) {
		return (unsigned int) (value + 0.5f);
	}

	STATIC_KEYWORD float rounded_cast_float(float value) {
		return value;
	}

	#define ALIGNED_(x)
#endif

//______SHARED_STRUCTS__________________________________________________________________________________________________

// https://devtalk.nvidia.com/default/topic/673965/are-there-any-cuda-libararies-for-3x3-matrix-amp-vector3-amp-quaternion-operations-/
typedef struct {
	cl_float4 m_row[3];
} Matrix3x3f;

typedef struct {
	cl_float2 m_row[2];
} Matrix2x2f;

typedef struct {
	cl_float4 m_row[4];
} Matrix4x4f;

//______HOST_CODE_______________________________________________________________________________________________________

#ifdef HOST_CODE
	inline cl_float2 make_float2(const vector2d &a)
	{
		cl_float2 v = {{(float) a.x(), (float) a.y()}};
		return v;
	}

	inline cl_float3 make_float3(const vector3d &a)
	{
		cl_float3 v = {{(float) a.x(), (float) a.y(), (float) a.z()}};
		return v;
	}

	inline cl_float4 make_float4(const vector4d &a)
	{
		cl_float4 v = {{(float) a.x(), (float) a.y(), (float) a.z(), (float) a.w()}};
		return v;
	}

	inline cl_float4 make_float4(float x, float y, float z, float w)
	{
		cl_float4 v = {{x, y, z, w}};
		return v;
	}

	inline Matrix3x3f make_matrix_f3x3(const matrix3x3d &a)
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4((float) a(0, 0), (float) a(0, 1), (float) a(0, 2), 0.0f);
		m.m_row[1] = make_float4((float) a(1, 0), (float) a(1, 1), (float) a(1, 2), 0.0f);
		m.m_row[2] = make_float4((float) a(2, 0), (float) a(2, 1), (float) a(2, 2), 0.0f);
		return m;
	}

	inline Matrix4x4f make_matrix_f4x4(const matrix4x4d &a)
	{
		Matrix4x4f m;
		m.m_row[0] = make_float4((float) a(0, 0), (float) a(0, 1), (float) a(0, 2), (float) a(0, 3));
		m.m_row[1] = make_float4((float) a(1, 0), (float) a(1, 1), (float) a(1, 2), (float) a(1, 3));
		m.m_row[2] = make_float4((float) a(2, 0), (float) a(2, 1), (float) a(2, 2), (float) a(2, 3));
		m.m_row[3] = make_float4((float) a(3, 0), (float) a(3, 1), (float) a(3, 2), (float) a(3, 3));
		return m;
	}

	#if defined(_MSC_VER)
		#define ALIGNED_(x) __declspec(align(x))
	#else
		#if defined(__GNUC__)
			#define ALIGNED_(x) __attribute__ ((aligned(x)))
		#else
			#error cannot specify alignment for platform
		#endif
	#endif
#endif

//______DEVICE_CODE_____________________________________________________________________________________________________

#ifndef HOST_CODE

	STATIC_KEYWORD size_t get_group_id_offsetted(uint dimindx)
	{
		return get_global_id(dimindx) / get_local_size(dimindx);
	}

	STATIC_KEYWORD BOOL_TYPE is_zero(float3 v)
	{
		return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
	}

#ifdef DEBUG
	STATIC_KEYWORD void print_matrix_f3x3(const Matrix3x3f m)
	{
		printf("[\n");
		printf("  [%f, %f, %f],\n", m.m_row[0].x, m.m_row[0].y, m.m_row[0].z);
		printf("  [%f, %f, %f],\n", m.m_row[1].x, m.m_row[1].y, m.m_row[1].z);
		printf("  [%f, %f, %f],\n", m.m_row[2].x, m.m_row[2].y, m.m_row[2].z);
		printf("]\n");
	}
#endif

	STATIC_KEYWORD Matrix3x3f make_matrix_f3x3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(a00, a01, a02, 0.0f);
		m.m_row[1] = make_float4(a10, a11, a12, 0.0f);
		m.m_row[2] = make_float4(a20, a21, a22, 0.0f);
		return m;
	}

	STATIC_KEYWORD Matrix3x3f make_zero_matrix_f3x3()
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		return m;
	}

	STATIC_KEYWORD Matrix3x3f make_eye_matrix_f3x3()
	{
		Matrix3x3f m;
		m.m_row[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		m.m_row[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
		m.m_row[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
		return m;
	}

	STATIC_KEYWORD Matrix3x3f transpose_f3x3(const Matrix3x3f m)
	{
		Matrix3x3f t;
		t.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.0f);
		t.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.0f);
		t.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.0f);
		return t;
	}

	STATIC_KEYWORD Matrix3x3f add_f3x3(const Matrix3x3f a, const Matrix3x3f b)
	{
		Matrix3x3f m;
		m.m_row[0] = a.m_row[0] + b.m_row[0];
		m.m_row[1] = a.m_row[1] + b.m_row[1];
		m.m_row[2] = a.m_row[2] + b.m_row[2];
		return m;
	}

	STATIC_KEYWORD Matrix3x3f mul_f3x3(const Matrix3x3f a, const Matrix3x3f b)
	{
		Matrix3x3f bt = transpose_f3x3(b);
		Matrix3x3f res;
		res.m_row[0] = make_float4(dot(a.m_row[0], bt.m_row[0]), dot(a.m_row[0], bt.m_row[1]), dot(a.m_row[0], bt.m_row[2]), 0.0f);
		res.m_row[1] = make_float4(dot(a.m_row[1], bt.m_row[0]), dot(a.m_row[1], bt.m_row[1]), dot(a.m_row[1], bt.m_row[2]), 0.0f);
		res.m_row[2] = make_float4(dot(a.m_row[2], bt.m_row[0]), dot(a.m_row[2], bt.m_row[1]), dot(a.m_row[2], bt.m_row[2]), 0.0f);
		return res;
	}

	STATIC_KEYWORD Matrix3x3f mul_f_f3x3(float k, const Matrix3x3f a)
	{
		Matrix3x3f m;
		m.m_row[0] = a.m_row[0] * k;
		m.m_row[1] = a.m_row[1] * k;
		m.m_row[2] = a.m_row[2] * k;
		return m;
	}

	STATIC_KEYWORD float3 mul_f3x3_f3(const Matrix3x3f a, const float3 b)
	{
		return make_float3(a.m_row[0].x * b.x + a.m_row[0].y * b.y + a.m_row[0].z * b.z,
						   a.m_row[1].x * b.x + a.m_row[1].y * b.y + a.m_row[1].z * b.z,
						   a.m_row[2].x * b.x + a.m_row[2].y * b.y + a.m_row[2].z * b.z);
	}

	STATIC_KEYWORD float2 transformPoint_f3x3(const Matrix3x3f m, const float2 p)
	{
		float3 temp = mul_f3x3_f3(m, make_float3(p.x, p.y, 1.0f));
		return make_float2(temp.x, temp.y) / temp.z;
	}

#ifdef DEBUG
	STATIC_KEYWORD void print_matrix_f4x4(const Matrix4x4f m)
	{
		printf("[\n");
		printf("  [%f, %f, %f, %f],\n", m.m_row[0].x, m.m_row[0].y, m.m_row[0].z, m.m_row[0].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[1].x, m.m_row[1].y, m.m_row[1].z, m.m_row[1].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[2].x, m.m_row[2].y, m.m_row[2].z, m.m_row[2].w);
		printf("  [%f, %f, %f, %f],\n", m.m_row[3].x, m.m_row[3].y, m.m_row[3].z, m.m_row[3].w);
		printf("]\n");
	}
#endif

	STATIC_KEYWORD Matrix4x4f make_matrix_f4x4(float a00, float a01, float a02, float a03,
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

	STATIC_KEYWORD Matrix4x4f make_translation_f4x4(const float3 t)
	{
		return make_matrix_f4x4(1.0f, 0.0f, 0.0f, t.x,
								0.0f, 1.0f, 0.0f, t.y,
								0.0f, 0.0f, 1.0f, t.z,
								0.0f, 0.0f, 0.0f, 1.0f);
	}

	STATIC_KEYWORD Matrix4x4f make_rotation_f4x4(const Matrix3x3f r)
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

	STATIC_KEYWORD float3 extract_translation_f4x4(const Matrix4x4f m)
	{
		float norm = 1.0f / m.m_row[3].w;
		return make_float3(m.m_row[0].w, m.m_row[1].w, m.m_row[2].w) * norm;
	}

	STATIC_KEYWORD Matrix3x3f extract_rotation_f4x4(const Matrix4x4f m)
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

	STATIC_KEYWORD Matrix4x4f transpose_f4x4(const Matrix4x4f m)
	{
		Matrix4x4f t;
		t.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, m.m_row[3].x);
		t.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, m.m_row[3].y);
		t.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, m.m_row[3].z);
		t.m_row[3] = make_float4(m.m_row[0].w, m.m_row[1].w, m.m_row[2].w, m.m_row[3].w);
		return t;
	}

	STATIC_KEYWORD Matrix4x4f mul_f4x4(const Matrix4x4f a, const Matrix4x4f b)
	{
		Matrix4x4f bt = transpose_f4x4(b);
		Matrix4x4f res;
		res.m_row[0] = make_float4(dot(a.m_row[0], bt.m_row[0]), dot(a.m_row[0], bt.m_row[1]), dot(a.m_row[0], bt.m_row[2]), dot(a.m_row[0], bt.m_row[3]));
		res.m_row[1] = make_float4(dot(a.m_row[1], bt.m_row[0]), dot(a.m_row[1], bt.m_row[1]), dot(a.m_row[1], bt.m_row[2]), dot(a.m_row[1], bt.m_row[3]));
		res.m_row[2] = make_float4(dot(a.m_row[2], bt.m_row[0]), dot(a.m_row[2], bt.m_row[1]), dot(a.m_row[2], bt.m_row[2]), dot(a.m_row[2], bt.m_row[3]));
		res.m_row[3] = make_float4(dot(a.m_row[3], bt.m_row[0]), dot(a.m_row[3], bt.m_row[1]), dot(a.m_row[3], bt.m_row[2]), dot(a.m_row[3], bt.m_row[3]));
		return res;
	}

	STATIC_KEYWORD float4 mul_f4x4_f4(const Matrix4x4f a, const float4 b)
	{
		return make_float4(a.m_row[0].x * b.x + a.m_row[0].y * b.y + a.m_row[0].z * b.z + a.m_row[0].w * b.w,
						   a.m_row[1].x * b.x + a.m_row[1].y * b.y + a.m_row[1].z * b.z + a.m_row[1].w * b.w,
						   a.m_row[2].x * b.x + a.m_row[2].y * b.y + a.m_row[2].z * b.z + a.m_row[2].w * b.w,
						   a.m_row[3].x * b.x + a.m_row[3].y * b.y + a.m_row[3].z * b.z + a.m_row[3].w * b.w);
	}

	STATIC_KEYWORD float3 transformPoint(const Matrix4x4f m, const float3 p)
	{
		float4 temp = mul_f4x4_f4(m, make_float4(p.x, p.y, p.z, 1.0f));
		return make_float3(temp.x, temp.y, temp.z) / temp.w;
	}

	STATIC_KEYWORD float3 transformVector(const Matrix4x4f m, const float3 v)
	{
		float4 temp = mul_f4x4_f4(m, make_float4(v.x, v.y, v.z, 0.0f));
		return make_float3(temp.x, temp.y, temp.z);
	}

	STATIC_KEYWORD float smootherstep(float edge0, float edge1, float x)
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

#endif

#endif // pragma once
