#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"

#include "camera_helpers.cl"
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
    __global const float* vertices,
    __global const uint* faces,
    __global uint* morton_codes,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ,
    uint nfaces)
{
    const uint index = get_global_id(0);
    if (index >= nfaces) {
        return;
    }

    uint3 face = loadFace(faces, index);
    float3 v0 = loadVertex(vertices, face.x);
    float3 v1 = loadVertex(vertices, face.y);
    float3 v2 = loadVertex(vertices, face.z);

    const float inv3 = 1.0f / 3.0f;
    float cX = (v0.x + v1.x + v2.x) * inv3;
    float cY = (v0.y + v1.y + v2.y) * inv3;
    float cZ = (v0.z + v1.z + v2.z) * inv3;

    const float eps = 1e-9f;
    float dx = max(maxX - minX, eps);
    float dy = max(maxY - minY, eps);
    float dz = max(maxZ - minZ, eps);

    float nx = (cX - minX) / dx;
    float ny = (cY - minY) / dy;
    float nz = (cZ - minZ) / dz;

    nx = clamp(nx, 0.0f, 1.0f);
    ny = clamp(ny, 0.0f, 1.0f);
    nz = clamp(nz, 0.0f, 1.0f);

    morton_codes[index] = morton3D(nx, ny, nz);
}
