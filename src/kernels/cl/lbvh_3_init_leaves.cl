#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__kernel void lbvh_3_init_leaves(
    __global const uint* indices,
    __global const float* vertices,
    __global const uint* faces,
    __global BVHNodeGPU* bvh_nodes,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    uint tri_idx = indices[i];

    uint i0 = faces[3 * tri_idx + 0];
    uint i1 = faces[3 * tri_idx + 1];
    uint i2 = faces[3 * tri_idx + 2];

    float3 v0 = (float3)(vertices[3 * i0 + 0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]);
    float3 v1 = (float3)(vertices[3 * i1 + 0], vertices[3 * i1 + 1], vertices[3 * i1 + 2]);
    float3 v2 = (float3)(vertices[3 * i2 + 0], vertices[3 * i2 + 1], vertices[3 * i2 + 2]);

    float3 min_v = min(v0, min(v1, v2));
    float3 max_v = max(v0, max(v1, v2));

    AABBGPU aabb;
    aabb.min_x = min_v.x;
    aabb.min_y = min_v.y;
    aabb.min_z = min_v.z;
    aabb.max_x = max_v.x;
    aabb.max_y = max_v.y;
    aabb.max_z = max_v.z;

    int leaf_idx = (n_faces - 1) + i;
    
    bvh_nodes[leaf_idx].aabb = aabb;
    bvh_nodes[leaf_idx].leftChildIndex = 0xFFFFFFFF;
    bvh_nodes[leaf_idx].rightChildIndex = 0xFFFFFFFF;
}


