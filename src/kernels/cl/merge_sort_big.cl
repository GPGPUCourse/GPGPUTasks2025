#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "geometry_helpers.cl"

void mergeSortImpl(
    const __global BVHPrimGPU* input_data,
    __global BVHPrimGPU* output_data,
    uint idx,
    uint blockStart,
    uint neighbourBlockStart,
    uint blockPairOffset,
    bool isLeft,
    uint n,
    int* startOffset,
    int* endOffset
)
{
    BVHPrimGPU val = input_data[idx];
    int start = neighbourBlockStart + *startOffset;
    int end = min(neighbourBlockStart + *endOffset, n);
    while (end - start > 1) {
        uint mid = (start + end) / 2;
        BVHPrimGPU midVal = input_data[mid];
        bool moveStart = false;
        if (isLeft) {
            moveStart = midVal.morton < val.morton;
        } else {
            moveStart = midVal.morton <= val.morton;
        }
        if (moveStart) {
            start = mid;
        } else {
            end = mid;
        }
    }
    uint writeIdx = blockPairOffset + (idx - blockStart) + max(end - (int)neighbourBlockStart, 0);
    output_data[writeIdx] = val;
    *startOffset = start - neighbourBlockStart;
    *endOffset = max(end - (int)neighbourBlockStart, 0);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_big(
    __global const BVHPrimGPU* input_data,
    __global       BVHPrimGPU* output_data,
                   uint  sorted_k_pow,
                   uint  n)
{
    const unsigned int bigBlockIdx = get_global_id(0);
    uint globalIdx = bigBlockIdx << BIG_BLOCK_POW;
    uint blockIdx = globalIdx >> sorted_k_pow;
    uint blockStart = blockIdx << sorted_k_pow;
    bool isLeft = (blockIdx & 1) == 0;
    uint neighbourBlockIdx = blockIdx + 2 * isLeft - 1;
    uint neighbourBlockStart = neighbourBlockIdx << sorted_k_pow;
    int sorted_k = 1 << sorted_k_pow;
    uint blockPairOffset = (blockIdx - !isLeft) << sorted_k_pow;
    int startOffset = -1;
    int endOffset = sorted_k;
#if SMALL_BLOCK_SIZE == 1
    mergeGroupImpl(input_data, output_data, localBlockIdx, blockStart, neighbourBlockStart, blockPairOffset, isLeft, &startOffset, &endOffset);
#else
    const unsigned int localIdx = get_local_id(0);
    const unsigned int groupIdx = get_group_id(0);
    const unsigned int groupOffset = groupIdx * GROUP_SIZE * BIG_BLOCK_SIZE;
    unsigned int bufIdxStart = groupOffset + localIdx;
    unsigned int count = min((uint)BIG_BLOCK_SIZE, (n - bufIdxStart + GROUP_SIZE - 1) >> GROUP_SIZE_POW);
    bool isOdd = count & 1;
    unsigned int bufIdxEnd = bufIdxStart + ((count - 1) << GROUP_SIZE_POW);
    for (uint i = 0; i < count >> 1; ++i) {
        int fakeEnd = endOffset;
        mergeSortImpl(input_data, output_data, bufIdxStart, blockStart, neighbourBlockStart, blockPairOffset, isLeft, n, &startOffset, &fakeEnd);
        int fakeStart = startOffset;
        mergeSortImpl(input_data, output_data, bufIdxEnd, blockStart, neighbourBlockStart, blockPairOffset, isLeft, n, &fakeStart, &endOffset);
        bufIdxStart += GROUP_SIZE;
        bufIdxEnd -= GROUP_SIZE;
    }
    if (isOdd) {
        mergeSortImpl(input_data, output_data, bufIdxStart, blockStart, neighbourBlockStart, blockPairOffset, isLeft, n, &startOffset, &endOffset);
    }
#endif
}
