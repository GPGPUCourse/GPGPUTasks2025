#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline uint expandBits(uint v) {
    // Ensure we have only lowest 10 bits
    rassert(v == (v & 0x3FFu), 76389413321);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

static inline uint morton3d(float x, float y, float z) {
    // Map and clamp to integer grid [0, 1023]
    uint ix = clamp(floor(x * 1024.0f), 0.0f, 1023.0f);
    uint iy = clamp(floor(y * 1024.0f), 0.0f, 1023.0f);
    uint iz = clamp(floor(z * 1024.0f), 0.0f, 1023.0f);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void calculate_prims(
    __global const float* vertices,
    __global const uint* faces,
    __global PrimGPU* prims,
    const uint nFaces,
    const float minSceneX,
    const float minSceneY,
    const float minSceneZ,
    const float maxSceneX,
    const float maxSceneY,
    const float maxSceneZ
) {
    const float eps = 1e-9f;
    const uint i = get_global_id(0);

    if (i >= nFaces) {
        return;
    }

    const uint3 face = loadFace(faces, i);
    const float3 u = loadVertex(vertices, face.x);
    const float3 v = loadVertex(vertices, face.y);
    const float3 w = loadVertex(vertices, face.z);

    const float div = 1.0f / 3.0f;
    float centroidX = (u.x + v.x + w.x) * div;
    float centroidY = (u.y + v.y + w.y) * div;
    float centroidZ = (u.z + v.z + w.z) * div;

    // Normalize by scene bounding box
    centroidX = (centroidX - minSceneX) / max(maxSceneX - minSceneX, eps);
    centroidY = (centroidY - minSceneY) / max(maxSceneY - minSceneY, eps);
    centroidZ = (centroidZ - minSceneZ) / max(maxSceneZ - minSceneZ, eps);

    // Clamp to [0, 1]
    centroidX = clamp(centroidX, 0.0f, 1.0f);
    centroidY = clamp(centroidY, 0.0f, 1.0f);
    centroidZ = clamp(centroidZ, 0.0f, 1.0f);

    uint mortonCode = morton3d(centroidX, centroidY, centroidZ);

    prims[i].faceIndex = i;
    prims[i].mortonCode = mortonCode;
}