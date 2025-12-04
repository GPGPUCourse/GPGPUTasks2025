#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/morton_code_gpu_shared.h"

uint expandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    
    uint xx = expandBits((uint)x);
    uint yy = expandBits((uint)y);
    uint zz = expandBits((uint)z);
    
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void lbvh_1_morton_codes(
    __global const float* vertices,
    __global const uint* faces,
    __global MortonCode* morton_codes,
    __global uint* indices,
    int n_faces,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    uint i0 = faces[3 * i + 0];
    uint i1 = faces[3 * i + 1];
    uint i2 = faces[3 * i + 2];

    float3 v0 = (float3)(vertices[3 * i0 + 0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]);
    float3 v1 = (float3)(vertices[3 * i1 + 0], vertices[3 * i1 + 1], vertices[3 * i1 + 2]);
    float3 v2 = (float3)(vertices[3 * i2 + 0], vertices[3 * i2 + 1], vertices[3 * i2 + 2]);

    float3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);

    float dx = max(max_x - min_x, 1e-9f);
    float dy = max(max_y - min_y, 1e-9f);
    float dz = max(max_z - min_z, 1e-9f);

    float nx = (centroid.x - min_x) / dx;
    float ny = (centroid.y - min_y) / dy;
    float nz = (centroid.z - min_z) / dz;

    MortonCode code = morton3D(nx, ny, nz);

    morton_codes[i] = code;
    indices[i] = i;
}

