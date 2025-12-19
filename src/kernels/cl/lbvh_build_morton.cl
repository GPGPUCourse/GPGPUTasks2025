#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "geometry_helpers.cl"

static inline uint expand_bits_10(uint v)
{
    v &= 1023u;
    v = (v | (v << 16)) & 0x30000FFu;
    v = (v | (v << 8)) & 0x300F00Fu;
    v = (v | (v << 4)) & 0x30C30C3u;
    v = (v | (v << 2)) & 0x9249249u;
    return v;
}

static inline uint morton3D_30(float x, float y, float z)
{
    x = clamp(x * 1024.0f, 0.0f, 1023.0f);
    y = clamp(y * 1024.0f, 0.0f, 1023.0f);
    z = clamp(z * 1024.0f, 0.0f, 1023.0f);

    uint xx = expand_bits_10((uint)x);
    uint yy = expand_bits_10((uint)y);
    uint zz = expand_bits_10((uint)z);
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void lbvh_build_morton(
    __global const float* vertices,
    __global const uint* faces,
    __global uint* morton,
    __global uint* triId,
    __global const float* scene_aabb6,
    uint nfaces)
{
    int i = (int)get_global_id(0);
    int n = (int)nfaces;
    if (i >= n) return;

    uint3 f = loadFace(faces, (uint)i);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    float3 ctr = (float3)((a.x + b.x + c.x) * (1.0f / 3.0f),
        (a.y + b.y + c.y) * (1.0f / 3.0f),
        (a.z + b.z + c.z) * (1.0f / 3.0f));

    float3 mn = (float3)(scene_aabb6[0], scene_aabb6[1], scene_aabb6[2]);
    float3 mx = (float3)(scene_aabb6[3], scene_aabb6[4], scene_aabb6[5]);
    float3 ext = (float3)(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z);

    float3 p = (float3)(
        ext.x > 0 ? (ctr.x - mn.x) / ext.x : 0.0f,
        ext.y > 0 ? (ctr.y - mn.y) / ext.y : 0.0f,
        ext.z > 0 ? (ctr.z - mn.z) / ext.z : 0.0f
    );

    morton[i] = morton3D_30(p.x, p.y, p.z);
    triId[i] = (uint)i;
}
