#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "geometry_helpers.cl"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
unsigned int expandBits(unsigned int v)
{
    // Ensure we have only lowest 10 bits
    rassert(v == (v & 0x3FFu), 76389413321);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void lbvh_morton(
    __global BVHPrimGPU*       prims,
    __global const AABBGPU*    minmax,
    uint                       nfaces)
{
    const uint globalIdx = get_global_id(0);

    AABBGPU bbox = *minmax;

    const float eps = 1e-9f;
    const float dx = max(bbox.max_x - bbox.min_x, eps);
    const float dy = max(bbox.max_y - bbox.min_y, eps);
    const float dz = max(bbox.max_z - bbox.min_z, eps);

    for (uint i = globalIdx * BOX_BLOCK_SIZE; i < min((globalIdx + 1) * BOX_BLOCK_SIZE, nfaces); ++i) {
        const float3 c = (float3)(prims[i].centroidX, prims[i].centroidY, prims[i].centroidZ);
        float nx = (c.x - bbox.min_x) / dx;
        float ny = (c.y - bbox.min_y) / dy;
        float nz = (c.z - bbox.min_z) / dz;

        nx = min(max(nx, 0.0f), 1.0f);
        ny = min(max(ny, 0.0f), 1.0f);
        nz = min(max(nz, 0.0f), 1.0f);

        prims[i].morton = morton3D(nx, ny, nz);
    }
}
