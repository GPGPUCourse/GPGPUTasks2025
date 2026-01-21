#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/morton_code_gpu_shared.h"

static inline uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline MortonCode morton3D(float x, float y, float z) {
    uint ix = min(max((int)(x * 1024.0f), 0), 1023);
    uint iy = min(max((int)(y * 1024.0f), 0), 1023);
    uint iz = min(max((int)(z * 1024.0f), 0), 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);

    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_compute_morton_codes(
    __global const float* triangle_centroids,
    __global MortonCode* morton_codes,
    __global uint* sorted_indices,
    float centroid_min_x, float centroid_min_y, float centroid_min_z,
    float centroid_max_x, float centroid_max_y, float centroid_max_z,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) return;

    float3 c = (float3)(triangle_centroids[3*i], triangle_centroids[3*i+1], triangle_centroids[3*i+2]);
    float3 centroid_min = (float3)(centroid_min_x, centroid_min_y, centroid_min_z);
    float3 centroid_max = (float3)(centroid_max_x, centroid_max_y, centroid_max_z);

    float eps = 1e-9f;
    float3 d = fmax(centroid_max - centroid_min, eps);
    float3 n = (c - centroid_min) / d;
    n = clamp(n, 0.0f, 1.0f);

    morton_codes[i] = morton3D(n.x, n.y, n.z);
    sorted_indices[i] = i;
}
