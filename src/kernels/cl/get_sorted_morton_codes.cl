#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__kernel void get_sorted_morton_codes(
    __global const uint *triIndexes,
    __global const MortonCode *mortonCodes,
    const uint nfaces,
    __global MortonCode *sortedCodes,
    __global const float* aabbXMin, __global const float* aabbXMax,
    __global const float* aabbYMin, __global const float* aabbYMax,
    __global const float* aabbZMin, __global const float* aabbZMax,
    __global BVHNodeGPU* nodes)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    const uint triIdx = triIndexes[i];
    sortedCodes[i] = mortonCodes[triIdx];

    AABBGPU aabb = {aabbXMin[triIdx], aabbYMin[triIdx], aabbZMin[triIdx],
        aabbXMax[triIdx], aabbYMax[triIdx], aabbZMax[triIdx]};
    nodes[i + nfaces - 1].aabb = aabb;
}