#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#define INVALID 0xffffffff

__kernel void lbvh_build_leaves(
    __global const BVHPrimGPU* prims,
    __global BVHNodeGPU*       bvhNodes,
    __global uint*             leafTriIndices,
    __global uint*             sortedCodes,
    uint                       nfaces)
{
    const uint globalIdx = get_global_id(0);

    for (uint i = globalIdx * BOX_BLOCK_SIZE; i < min((globalIdx + 1) * BOX_BLOCK_SIZE, nfaces); ++i) {
        sortedCodes[i] = prims[i].morton;
        leafTriIndices[i] = prims[i].triIndex;

        uint leafIndex = (nfaces - 1) + i;
        BVHNodeGPU leaf;

        leaf.aabb = prims[i].aabb;
        leaf.leftChildIndex  = INVALID;
        leaf.rightChildIndex = INVALID;

        bvhNodes[leafIndex] = leaf;
    }
}
