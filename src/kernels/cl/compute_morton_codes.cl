#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"

#include "camera_helpers.cl"
#include "centroids_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

static inline uint expandBits(uint v)
{
    rassert(v == (v & 0x3FFu), 8914765981);

    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

static inline uint morton3D(float x, float y, float z)
{
    unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
compute_morton_codes(
    __global const CentroidGPU* centroids,
    __global const CentroidGPU* minCentroid,
    __global const CentroidGPU* maxCentroid,
    __global uint* morton_codes,
    uint nfaces)
{
    const uint index = get_global_id(0);
    if (index >= nfaces) {
        return;
    }

    float3 c = loadCentroid(centroids, index);
    float3 cMin = loadCentroid(minCentroid, 0);
    float3 cMax = loadCentroid(maxCentroid, 0);

    const float eps = 1e-9f;
    float dx = max(cMax.x - cMin.x, eps);
    float dy = max(cMax.y - cMin.y, eps);
    float dz = max(cMax.z - cMin.z, eps);

    float nx = (c.x - cMin.x) / dx;
    float ny = (c.y - cMin.y) / dy;
    float nz = (c.z - cMin.z) / dz;

    nx = min(max(nx, 0.0f), 1.0f);
    ny = min(max(ny, 0.0f), 1.0f);
    nz = min(max(nz, 0.0f), 1.0f);

    morton_codes[index] = morton3D(nx, ny, nz);
}
