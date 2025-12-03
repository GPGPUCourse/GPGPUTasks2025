#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"


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
static inline MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    uint ix = min(max((int) (x * 1024.0f), 0), 1023);
    uint iy = min(max((int) (y * 1024.0f), 0), 1023);
    uint iz = min(max((int) (z * 1024.0f), 0), 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
morton_build(
    __global const float* vertices,
    __global const uint*  faces,
    __global MortonCode*  morton_codes,
    const uint            N,
    const float           minX,
    const float           minY,
    const float           minZ,
    const float           dx,
    const float           dy,
    const float           dz)
{
    const uint i = get_global_id(0);
    if (i >= N) {
        return;
    }

    uint3 f = loadFace(faces, i);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    float cX = (a.x + b.x + c.x) / 3.0f;
    float cY = (a.y + b.y + c.y) / 3.0f;
    float cZ = (a.z + b.z + c.z) / 3.0f;

    float nx = clamp((cX - minX) / dx, 0.0f, 1.0f);
    float ny = clamp((cY - minY) / dy, 0.0f, 1.0f);
    float nz = clamp((cZ - minZ) / dz, 0.0f, 1.0f);

    morton_codes[i] = morton3D(nx, ny, nz);
}