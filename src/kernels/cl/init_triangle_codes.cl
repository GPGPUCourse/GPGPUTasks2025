
#include "helpers/rassert.cl"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "../defines.h"

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
__kernel void init_triangle_codes(
    __global MortonCode* codes,
    __global int* indices,
    __global const float*     vertices,
    __global const uint*      faces,
             int n,
             float x_min, float y_min, float z_min,
             float x_max, float y_max, float z_max
)
{
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    indices[i] = i;

    uint3  f = loadFace(faces, i);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 c;
    c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    c.x = (c.x - x_min) / (x_max - x_min + 1e-9f);
    c.y = (c.y - y_min) / (y_max - y_min + 1e-9f);
    c.z = (c.z - z_min) / (z_max - z_min + 1e-9f);

    codes[i] = morton3D(c.x, c.y, c.z);
}
