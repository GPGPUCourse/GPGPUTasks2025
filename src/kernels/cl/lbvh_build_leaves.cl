#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "geometry_helpers.cl"

static inline AABBGPU aabb_empty()
{
    AABBGPU b;
    b.min_x = b.min_y = b.min_z = 1e30f;
    b.max_x = b.max_y = b.max_z = -1e30f;
    return b;
}

static inline void aabb_extend(__private AABBGPU* b, float3 p)
{
    b->min_x = fmin(b->min_x, p.x);
    b->min_y = fmin(b->min_y, p.y);
    b->min_z = fmin(b->min_z, p.z);
    b->max_x = fmax(b->max_x, p.x);
    b->max_y = fmax(b->max_y, p.y);
    b->max_z = fmax(b->max_z, p.z);
}

__kernel void lbvh_build_leaves(
    __global const float* vertices,
    __global const uint* faces,
    __global const uint* triId_sorted,
    __global BVHNodeGPU* nodes,
    __global uint* leafTriIndices,
    uint nfaces)
{
    int i = (int)get_global_id(0);
    int n = (int)nfaces;
    if (i >= n) return;

    uint triId = triId_sorted[i];
    leafTriIndices[i] = triId;

    uint3 f = loadFace(faces, triId);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    AABBGPU box = aabb_empty();
    aabb_extend(&box, a);
    aabb_extend(&box, b);
    aabb_extend(&box, c);

    int leafStart = n - 1;
    int nodeIdx = leafStart + i;

    nodes[nodeIdx].aabb = box;
    nodes[nodeIdx].leftChildIndex = 0;
    nodes[nodeIdx].rightChildIndex = 0;
}
