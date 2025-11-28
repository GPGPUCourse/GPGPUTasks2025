#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../shared_structs/aabb_gpu_shared.h"

static inline int common_bits_from(__global const uint *arr, uint n, int i, int j)
{
    if (i < 0 || j < 0 || i >= n || j >= n)
    {
        return -1;
    }
    return clz(arr[i] ^ arr[j]);
}

// From CPU implementation
static inline uint expandBits(uint v)
{

    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint getMortonCode(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

// Load vertex/face from compact arrays
static inline float3 loadVertex(__global const float *vertices,
                                uint vi)
{
    return (float3)(vertices[3 * vi + 0],
                    vertices[3 * vi + 1],
                    vertices[3 * vi + 2]);
}

static inline uint3 loadFace(__global const uint *faces,
                             uint fi)
{
    return (uint3)(faces[3 * fi + 0],
                   faces[3 * fi + 1],
                   faces[3 * fi + 2]);
}

// basic vector helpers used by ray/triangle and ray/AABB
static inline float3 cross_f3(const float3 a, const float3 b)
{
    return (float3)(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

static inline float3 normalize_f3(const float3 v)
{
    float inv = rsqrt(v.x * v.x + v.y * v.y + v.z * v.z + 1e-20f);
    return (float3)(v.x * inv, v.y * inv, v.z * inv);
}

static inline float length_f3(const float3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// dot/cross for float3
static inline float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float3 cross3(const float3 a, const float3 b)
{
    return (float3)(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

// Möller–Trumbore: ray_o + t*ray_d vs triangle (v0,v1,v2)
// Returns true on hit; outputs t (distance), u,v (barycentrics). w = 1-u-v.
// tMin/tMax control valid interval; set tMin=0.0f, tMax=FLT_MAX for "any".
static inline bool intersect_ray_triangle(const float3 ray_o,
                                          const float3 ray_d,
                                          const float3 v0,
                                          const float3 v1,
                                          const float3 v2,
                                          const float tMin,
                                          const float tMax,
                                          const bool backface_cull,
                                          __private float *t,
                                          __private float *u,
                                          __private float *v)
{
    // Edges
    const float3 e1 = (float3)(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const float3 e2 = (float3)(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    // pvec = d x e2
    const float3 pvec = cross3(ray_d, e2);
    const float det = dot3(e1, pvec);

    const float eps = 1e-8f; // geometric epsilon

    if (backface_cull)
    {
        if (det <= eps)
            return false;
    }
    else
    {
        if (fabs(det) <= eps)
            return false;
    }

    const float invDet = 1.0f / det;

    // s = o - v0; u = (s·pvec)/det
    const float3 s = (float3)(ray_o.x - v0.x,
                              ray_o.y - v0.y,
                              ray_o.z - v0.z);
    *u = dot3(s, pvec) * invDet;
    if (*u < 0.0f || *u > 1.0f)
        return false;

    // qvec = s x e1; v = (d·qvec)/det
    const float3 qvec = cross3(s, e1);
    *v = dot3(ray_d, qvec) * invDet;
    if (*v < 0.0f || (*u + *v) > 1.0f)
        return false;

    // t = (e2·qvec)/det
    *t = dot3(e2, qvec) * invDet;
    if (*t < tMin || *t > tMax)
        return false;

    return true;
}

// Convenience wrapper: any-hit with default interval [0, +inf)
static inline bool intersect_ray_triangle_any(const float3 ray_o,
                                              const float3 ray_d,
                                              const float3 v0,
                                              const float3 v1,
                                              const float3 v2,
                                              bool backface_cull,
                                              __private float *t,
                                              __private float *u,
                                              __private float *v)
{
    return intersect_ray_triangle(ray_o, ray_d, v0, v1, v2,
                                  0.0f, FLT_MAX, backface_cull,
                                  t, u, v);
}

// Ray vs AABB slab test
// ray_o + t * ray_d, t in [tMin, tMax]
// Returns true on hit; outputs tHitNear (entry) and tHitFar (exit)
static inline bool intersect_ray_aabb(const float3 ray_o,
                                      const float3 ray_d,
                                      const AABBGPU box,
                                      float tMin,
                                      float tMax,
                                      __private float *tHitNear,
                                      __private float *tHitFar)
{
    float t0 = tMin;
    float t1 = tMax;

    const float eps = 1e-8f; // geometric epsilon for parallel check

    // X slab
    if (fabs(ray_d.x) < eps)
    {
        // Ray is parallel to X planes; must be inside slab
        if (ray_o.x < box.min_x || ray_o.x > box.max_x)
            return false;
    }
    else
    {
        float invDx = 1.0f / ray_d.x;
        float tx0 = (box.min_x - ray_o.x) * invDx;
        float tx1 = (box.max_x - ray_o.x) * invDx;
        if (invDx < 0.0f)
        {
            float tmp = tx0;
            tx0 = tx1;
            tx1 = tmp;
        }
        if (tx0 > t0)
            t0 = tx0;
        if (tx1 < t1)
            t1 = tx1;
        if (t1 < t0)
            return false;
    }

    // Y slab
    if (fabs(ray_d.y) < eps)
    {
        if (ray_o.y < box.min_y || ray_o.y > box.max_y)
            return false;
    }
    else
    {
        float invDy = 1.0f / ray_d.y;
        float ty0 = (box.min_y - ray_o.y) * invDy;
        float ty1 = (box.max_y - ray_o.y) * invDy;
        if (invDy < 0.0f)
        {
            float tmp = ty0;
            ty0 = ty1;
            ty1 = tmp;
        }
        if (ty0 > t0)
            t0 = ty0;
        if (ty1 < t1)
            t1 = ty1;
        if (t1 < t0)
            return false;
    }

    // Z slab
    if (fabs(ray_d.z) < eps)
    {
        if (ray_o.z < box.min_z || ray_o.z > box.max_z)
            return false;
    }
    else
    {
        float invDz = 1.0f / ray_d.z;
        float tz0 = (box.min_z - ray_o.z) * invDz;
        float tz1 = (box.max_z - ray_o.z) * invDz;
        if (invDz < 0.0f)
        {
            float tmp = tz0;
            tz0 = tz1;
            tz1 = tmp;
        }
        if (tz0 > t0)
            t0 = tz0;
        if (tz1 < t1)
            t1 = tz1;
        if (t1 < t0)
            return false;
    }

    *tHitNear = t0;
    *tHitFar = t1;
    return true;
}

// Convenience wrapper: any-hit with default interval [0, +inf)
static inline bool intersect_ray_aabb_any(const float3 ray_o,
                                          const float3 ray_d,
                                          const AABBGPU box,
                                          __private float *tHitNear,
                                          __private float *tHitFar)
{
    return intersect_ray_aabb(ray_o, ray_d, box,
                              0.0f, FLT_MAX,
                              tHitNear, tHitFar);
}
