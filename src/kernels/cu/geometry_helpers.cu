// cuda_ray_triangle.h
#pragma once
#include <cuda_runtime.h>
#include <float.h>

#include "../shared_structs/aabb_gpu_shared.h"

namespace {
__device__ float3 loadVertex(const float* vertices, unsigned int vi)
{
    return {vertices[3 * vi + 0], vertices[3 * vi + 1], vertices[3 * vi + 2]};
}

__device__ uint3 loadFace(const unsigned int* faces, unsigned int fi)
{
    return {faces[3 * fi + 0], faces[3 * fi + 1], faces[3 * fi + 2]};
}

__device__ inline float3 cross_f3(const float3& a, const float3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ inline float3 normalize_f3(const float3& v) {
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z + 1e-20f);
    return {v.x*inv, v.y*inv, v.z*inv};
}
__device__ inline float  length_f3(const float3& v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// dot/cross for float3
static __device__ __forceinline__ float dot3(const float3 a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static __device__ __forceinline__ float3 cross3(const float3 a, const float3 b) {
    return make_float3(a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
}

// Möller–Trumbore: ray_o + t*ray_d vs triangle (v0,v1,v2)
// Returns true on hit; outputs t (distance), u,v (barycentrics). w = 1-u-v.
// tMin/tMax control valid interval; set tMin=0.0f, tMax=FLT_MAX for "any".
static __device__ __forceinline__
    bool intersect_ray_triangle(const float3 ray_o, const float3 ray_d,
        const float3 v0,    const float3 v1,    const float3 v2,
        const float  tMin,  const float  tMax,
        const bool   backface_cull,
        float &t, float &u, float &v)
{
    // Edges
    const float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    // pvec = d x e2
    const float3 pvec = cross3(ray_d, e2);
    const float det = dot3(e1, pvec);

    const float eps = 1e-8f; // geometric epsilon

    if (backface_cull) {
        if (det <= eps) return false;
    } else {
        if (fabsf(det) <= eps) return false;
    }

    const float invDet = 1.0f / det;

    // s = o - v0; u = (s·pvec)/det
    const float3 s = make_float3(ray_o.x - v0.x, ray_o.y - v0.y, ray_o.z - v0.z);
    u = dot3(s, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    // qvec = s x e1; v = (d·qvec)/det
    const float3 qvec = cross3(s, e1);
    v = dot3(ray_d, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    // t = (e2·qvec)/det
    t = dot3(e2, qvec) * invDet;
    if (t < tMin || t > tMax) return false;

    return true;
}

// Convenience wrapper: any-hit with default interval [0, +inf)
static __device__ __forceinline__
    bool intersect_ray_triangle_any(const float3 ray_o, const float3 ray_d,
        const float3 v0, const float3 v1, const float3 v2,
        bool backface_cull,
        float &t, float &u, float &v)
{
    return intersect_ray_triangle(ray_o, ray_d, v0, v1, v2, 0.0f, FLT_MAX, backface_cull, t, u, v);
}

// Ray vs AABB slab test
// ray_o + t * ray_d, t in [tMin, tMax]
// Returns true on hit; outputs tHitNear (entry) and tHitFar (exit)
static __device__ __forceinline__
bool intersect_ray_aabb(const float3 ray_o, const float3 ray_d,
                        const AABBGPU &box,
                        float tMin, float tMax,
                        float &tHitNear, float &tHitFar)
{
    float t0 = tMin;
    float t1 = tMax;

    const float eps = 1e-8f; // geometric epsilon for parallel check

    // X slab
    if (fabsf(ray_d.x) < eps) {
        // Ray is parallel to X planes; must be inside slab
        if (ray_o.x < box.min_x || ray_o.x > box.max_x)
            return false;
    } else {
        float invDx = 1.0f / ray_d.x;
        float tx0 = (box.min_x - ray_o.x) * invDx;
        float tx1 = (box.max_x - ray_o.x) * invDx;
        if (invDx < 0.0f) {
            float tmp = tx0; tx0 = tx1; tx1 = tmp;
        }
        if (tx0 > t0) t0 = tx0;
        if (tx1 < t1) t1 = tx1;
        if (t1 < t0) return false;
    }

    // Y slab
    if (fabsf(ray_d.y) < eps) {
        if (ray_o.y < box.min_y || ray_o.y > box.max_y)
            return false;
    } else {
        float invDy = 1.0f / ray_d.y;
        float ty0 = (box.min_y - ray_o.y) * invDy;
        float ty1 = (box.max_y - ray_o.y) * invDy;
        if (invDy < 0.0f) {
            float tmp = ty0; ty0 = ty1; ty1 = tmp;
        }
        if (ty0 > t0) t0 = ty0;
        if (ty1 < t1) t1 = ty1;
        if (t1 < t0) return false;
    }

    // Z slab
    if (fabsf(ray_d.z) < eps) {
        if (ray_o.z < box.min_z || ray_o.z > box.max_z)
            return false;
    } else {
        float invDz = 1.0f / ray_d.z;
        float tz0 = (box.min_z - ray_o.z) * invDz;
        float tz1 = (box.max_z - ray_o.z) * invDz;
        if (invDz < 0.0f) {
            float tmp = tz0; tz0 = tz1; tz1 = tmp;
        }
        if (tz0 > t0) t0 = tz0;
        if (tz1 < t1) t1 = tz1;
        if (t1 < t0) return false;
    }

    tHitNear = t0;
    tHitFar  = t1;
    return true;
}

static __device__ __forceinline__
bool intersect_ray_aabb_fast(const float3 ray_o, const float3 invDir,
                             const AABBGPU &box,
                             float tMin, float tMax,
                             float &tHitNear, float &tHitFar)
{
    float t0 = tMin;
    float t1 = tMax;

    {
        float tx0 = (box.min_x - ray_o.x) * invDir.x;
        float tx1 = (box.max_x - ray_o.x) * invDir.x;
        float txMin = fminf(tx0, tx1);
        float txMax = fmaxf(tx0, tx1);
        t0 = fmaxf(t0, txMin);
        t1 = fminf(t1, txMax);
    }

    {
        float ty0 = (box.min_y - ray_o.y) * invDir.y;
        float ty1 = (box.max_y - ray_o.y) * invDir.y;
        float tyMin = fminf(ty0, ty1);
        float tyMax = fmaxf(ty0, ty1);
        t0 = fmaxf(t0, tyMin);
        t1 = fminf(t1, tyMax);
    }

    {
        float tz0 = (box.min_z - ray_o.z) * invDir.z;
        float tz1 = (box.max_z - ray_o.z) * invDir.z;
        float tzMin = fminf(tz0, tz1);
        float tzMax = fmaxf(tz0, tz1);
        t0 = fmaxf(t0, tzMin);
        t1 = fminf(t1, tzMax);
    }

    tHitNear = t0;
    tHitFar  = t1;
    return t1 >= t0;
}

static __device__ __forceinline__
bool intersect_ray_aabb_any(const float3 ray_o, const float3 ray_d,
                            const AABBGPU &box,
                            float &tHitNear, float &tHitFar)
{
    return intersect_ray_aabb(ray_o, ray_d, box, 0.0f, FLT_MAX, tHitNear, tHitFar);
}

__device__ __forceinline__ float3 loadVertexLdg(const float* vertices, unsigned int vi)
{
    return make_float3(__ldg(&vertices[3 * vi + 0]), 
                       __ldg(&vertices[3 * vi + 1]), 
                       __ldg(&vertices[3 * vi + 2]));
}

__device__ __forceinline__ uint3 loadFaceLdg(const unsigned int* faces, unsigned int fi)
{
    return make_uint3(__ldg(&faces[3 * fi + 0]), 
                      __ldg(&faces[3 * fi + 1]), 
                      __ldg(&faces[3 * fi + 2]));
}
}
