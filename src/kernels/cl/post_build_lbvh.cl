#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "geometry_helpers.cl"

__kernel void post_build_lbvh(
    __global       BVHNodeGPU * outNodes,
             const uint         N)
{
    const uint index = get_global_id(0);

    if (index >= N) {
        return;
    }

    __global       BVHNodeGPU * node = outNodes + index;
    __global const BVHNodeGPU * left  = outNodes + node->leftChildIndex;
    __global const BVHNodeGPU * right = outNodes + node->rightChildIndex;

    AABBGPU aabb;
    aabb.min_x = min(left->aabb.min_x, right->aabb.min_x);
    aabb.min_y = min(left->aabb.min_y, right->aabb.min_y);
    aabb.min_z = min(left->aabb.min_z, right->aabb.min_z);
    aabb.max_x = max(left->aabb.max_x, right->aabb.max_x);
    aabb.max_y = max(left->aabb.max_y, right->aabb.max_y);
    aabb.max_z = max(left->aabb.max_z, right->aabb.max_z);

    node->aabb = aabb;
}


