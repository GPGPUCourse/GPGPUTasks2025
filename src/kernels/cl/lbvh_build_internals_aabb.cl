#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

__kernel void lbvh_build_internals_aabb(
    __global BVHNodeGPU*       bvhNodes,
    __global const uint*       inFlags,
    __global uint*             outFlags,
    uint                       nfaces)
{
    const uint globalIdx = get_global_id(0);
    if (globalIdx >= nfaces - 1) {
        return;
    }

    uint flag = inFlags[globalIdx];
    if (flag) {
        outFlags[globalIdx] = 1;
        return;
    }

    BVHNodeGPU node = bvhNodes[globalIdx];
    uint left = node.leftChildIndex;
    flag = left >= nfaces - 1 || inFlags[left];
    uint right = node.rightChildIndex;
    flag &= right >= nfaces - 1 || inFlags[right];
    if (!flag) {
        return;
    }

    AABBGPU lBbox = bvhNodes[left].aabb;
    AABBGPU rBbox = bvhNodes[right].aabb;

    AABBGPU aabb;

    aabb.min_x = min(lBbox.min_x, rBbox.min_x);
    aabb.min_y = min(lBbox.min_y, rBbox.min_y);
    aabb.min_z = min(lBbox.min_z, rBbox.min_z);
    aabb.max_x = max(lBbox.max_x, rBbox.max_x);
    aabb.max_y = max(lBbox.max_y, rBbox.max_y);
    aabb.max_z = max(lBbox.max_z, rBbox.max_z);

    bvhNodes[globalIdx].aabb = aabb;

    outFlags[globalIdx] = 1;
}
