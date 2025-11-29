#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "geometry_helpers.cl"

void mergeThread(const __local BVHPrimGPU* a, const __local BVHPrimGPU* b, __local BVHPrimGPU* res, uint n) {
    uint aIdx = 0;
    uint bIdx = 0;

    BVHPrimGPU aVal = *a;
    BVHPrimGPU bVal = *b;
    while (true) {
        if (aVal.morton <= bVal.morton) {
            *(res++) = aVal;
            ++aIdx;
            if (aIdx == n) {
                break;
            }
            aVal = a[aIdx];
        } else {
            *(res++) = bVal;
            ++bIdx;
            if (bIdx == n) {
                break;
            }
            bVal = b[bIdx];
        }
    }

    if (aIdx < n) {
        while (true) {
            *(res++) = aVal;
            ++aIdx;
            if (aIdx == n) {
                break;
            }
            aVal = a[aIdx];
        }
    } else {
        while (true) {
            *(res++) = bVal;
            ++bIdx;
            if (bIdx == n) {
                break;
            }
            bVal = b[bIdx];
        }
    }
}

void mergeGroupImpl(
    const __local BVHPrimGPU* input_data,
    __local BVHPrimGPU* output_data,
    uint idx,
    uint blockStart,
    uint neighbourBlockStart,
    uint blockPairOffset,
    bool isLeft,
    int* startOffset,
    int* endOffset)
{
    BVHPrimGPU val = input_data[idx];
    int start = neighbourBlockStart + *startOffset;
    int end = neighbourBlockStart + *endOffset;
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
    *endOffset = end - neighbourBlockStart;
}

void mergeGroup(
    __local const BVHPrimGPU* input_data,
    __local       BVHPrimGPU* output_data,
                  uint  smallBlockIdx,
                  uint  sorted_k_pow,
                  uint  sorted_k)
{
    uint localBlockIdx = (smallBlockIdx << SMALL_BLOCK_POW);
    uint blockIdx = localBlockIdx >> sorted_k_pow;
    uint blockStart = blockIdx << sorted_k_pow;
    bool isLeft = (blockIdx & 1) == 0;
    uint neighbourBlockIdx = blockIdx + 2 * isLeft - 1;
    uint neighbourBlockStart = neighbourBlockIdx << sorted_k_pow;
    int startOffset = -1;
    int endOffset = sorted_k;
    uint blockPairOffset = (blockIdx - !isLeft) << sorted_k_pow;
#if SMALL_BLOCK_SIZE == 1
    mergeGroupImpl(input_data, output_data, localBlockIdx, blockStart, neighbourBlockStart, blockPairOffset, isLeft, &startOffset, &endOffset);
#else
    for (unsigned int j = 0; j < SMALL_BLOCK_SIZE / 2; ++j) {
        int fakeEnd = endOffset;
        mergeGroupImpl(input_data, output_data, localBlockIdx + j, blockStart, neighbourBlockStart, blockPairOffset, isLeft, &startOffset, &fakeEnd);
        int fakeStart = startOffset;
        mergeGroupImpl(input_data, output_data, localBlockIdx + (SMALL_BLOCK_SIZE - 1 - j), blockStart, neighbourBlockStart, blockPairOffset, isLeft, &fakeStart, &endOffset);
    }
#endif
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_small(
    __global const BVHPrimGPU* input_data,
    __global       BVHPrimGPU* output_data,
                         uint  n)
{
    __local BVHPrimGPU buf1[GROUP_SIZE * SMALL_BLOCK_SIZE];
    __local BVHPrimGPU buf2[GROUP_SIZE * SMALL_BLOCK_SIZE];
    const unsigned int globalIdx = get_global_id(0);
    const unsigned int localIdx = get_local_id(0);
    const unsigned int groupIdx = get_group_id(0);
    const unsigned int groupOffset = groupIdx * GROUP_SIZE * SMALL_BLOCK_SIZE;
    unsigned int bufIdx = localIdx;
    for (unsigned int i = 0; i < SMALL_BLOCK_SIZE; ++i) {
        if (groupOffset + bufIdx < n) {
            buf1[bufIdx] = input_data[groupOffset + bufIdx];
        } else {
            buf1[bufIdx].morton = 0xFFFFFFFF;
        }
        bufIdx += GROUP_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local BVHPrimGPU *in = buf1;
    __local BVHPrimGPU *out = buf2;
    for (unsigned int i = 1; i < SMALL_BLOCK_SIZE; i <<= 1) {
        for (unsigned int j = 0; j < SMALL_BLOCK_SIZE; j += 2 * i) {
            mergeThread(in + localIdx * SMALL_BLOCK_SIZE + j, in + localIdx * SMALL_BLOCK_SIZE + j + i, out + localIdx * SMALL_BLOCK_SIZE + j, i);
        }
        __local BVHPrimGPU *tmp = in;
        in = out;
        out = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sorted_k_pow = SMALL_BLOCK_POW;
    for (unsigned int i = SMALL_BLOCK_SIZE; i < SMALL_BLOCK_SIZE * GROUP_SIZE; i <<= 1) {
        mergeGroup(in, out, localIdx, sorted_k_pow, i);
        __local BVHPrimGPU *tmp = in;
        in = out;
        out = tmp;
        ++sorted_k_pow;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    bufIdx = localIdx;
    for (unsigned int i = 0; i < SMALL_BLOCK_SIZE; ++i) {
        if (groupOffset + bufIdx < n) {
            output_data[groupOffset + bufIdx] = in[bufIdx];
        }
        bufIdx += GROUP_SIZE;
    }
}
