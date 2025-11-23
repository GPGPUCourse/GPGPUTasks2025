#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "geometry_helpers.cl"

static inline float max(
    const float x,
    const float y,
    const float z)
{
    return max(x, max(y, z));
}

static inline float min(
    const float x,
    const float y,
    const float z)
{
    return min(x, min(y, z));
}

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
static inline uint expandBits(uint v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
static inline MortonCode morton3D(
    const float x,
    const float y,
    const float z)
{
    // Map and clamp to integer grid [0, 1023]
    const uint ix = min(max((int)(x * 1024.0f), 0), 1023);
    const uint iy = min(max((int)(y * 1024.0f), 0), 1023);
    const uint iz = min(max((int)(z * 1024.0f), 0), 1023);

    const uint xx = expandBits(ix);
    const uint yy = expandBits(iy);
    const uint zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void pre_build_lbvh(
    __global const float      * vertices,
    __global const uint       * faces,
    __global       Prim       * prims, // size = N
             const uint         N)     //      > 2
{
    const uint index = get_global_id(0);

    if (index >= N) {
        return;
    }

    // 1) Compute per-triangle AABB and centroids
    float3 cMin;
    cMin.x = +INFINITY;
    cMin.y = +INFINITY;
    cMin.z = +INFINITY;

    float3 cMax;
    cMax.x = -INFINITY;
    cMax.y = -INFINITY;
    cMax.z = -INFINITY;

    __global Prim * prim = prims + index;

    // Centroid
    float3 c;

    {
        const uint3 face = loadFace(faces, index);
        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = min(v0.x, v1.x, v2.x);
        aabb.min_y = min(v0.y, v1.y, v2.y);
        aabb.min_z = min(v0.z, v1.z, v2.z);
        aabb.max_x = max(v0.x, v1.x, v2.x);
        aabb.max_y = max(v0.y, v1.y, v2.y);
        aabb.max_z = max(v0.z, v1.z, v2.z);

        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        prim->triIndex = index;
        prim->aabb     = aabb;

        // Update centroid bounds
        cMin.x = min(cMin.x, c.x);
        cMin.y = min(cMin.y, c.y);
        cMin.z = min(cMin.z, c.z);
        cMax.x = max(cMax.x, c.x);
        cMax.y = max(cMax.y, c.y);
        cMax.z = max(cMax.z, c.z);
    }

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = max(cMax.x - cMin.x, eps);
    const float dy = max(cMax.y - cMin.y, eps);
    const float dz = max(cMax.z - cMin.z, eps);

    {
        float nx = (c.x - cMin.x) / dx;
        float ny = (c.y - cMin.y) / dy;
        float nz = (c.z - cMin.z) / dz;

        // Clamp to [0,1]
        nx = min(max(nx, 0.0f), 1.0f);
        ny = min(max(ny, 0.0f), 1.0f);
        nz = min(max(nz, 0.0f), 1.0f);

        prim->morton = morton3D(nx, ny, nz);
    }
}
