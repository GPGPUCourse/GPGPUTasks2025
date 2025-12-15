#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__kernel void calc_bvh_aabb(
    __global const uint* triIndexes,
    __global const float* aabbXMin, __global const float* aabbXMax,
    __global const float* aabbYMin, __global const float* aabbYMax,
    __global const float* aabbZMin, __global const float* aabbZMax,
    __global const uint* parents,
    __global uint* counters,
    __global BVHNodeGPU* bvhNodes,
    const uint nfaces)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    uint curNode = i + nfaces - 1;
    const uint triIndex = triIndexes[i];
    do {
        rassert(curNode >= 0 && curNode < nfaces * 2 - 1, 76487125);
        curNode = parents[curNode];
        const uint old = atomic_inc(counters + curNode);
        if (old == 0) {
            return;
        }
        BVHNodeGPU node = bvhNodes[curNode];
        AABBGPU leftAABB = bvhNodes[node.leftChildIndex].aabb;
        AABBGPU rightAABB = bvhNodes[node.rightChildIndex].aabb;
        node.aabb.min_x = min(leftAABB.min_x, rightAABB.min_x);
        node.aabb.min_y = min(leftAABB.min_y, rightAABB.min_y);
        node.aabb.min_z = min(leftAABB.min_z, rightAABB.min_z);
        node.aabb.max_x = max(leftAABB.max_x, rightAABB.max_x);
        node.aabb.max_y = max(leftAABB.max_y, rightAABB.max_y);
        node.aabb.max_z = max(leftAABB.max_z, rightAABB.max_z);
        bvhNodes[curNode].aabb = node.aabb;
    } while (curNode != 0);
}