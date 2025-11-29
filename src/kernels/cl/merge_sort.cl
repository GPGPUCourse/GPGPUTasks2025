#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "geometry_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const BVHPrimGPU* input_data,
    __global       BVHPrimGPU* output_data,
                   uint  sorted_k_pow,
                   uint  n)
{
    const unsigned int globalIdx = get_global_id(0);
    if (globalIdx >= n) {
        return;
    }
    int sorted_k = 1 << sorted_k_pow;
    BVHPrimGPU val = input_data[globalIdx];
    uint blockIdx = globalIdx >> sorted_k_pow;
    uint blockStart = blockIdx << sorted_k_pow;
    bool isLeft = (blockIdx & 1) == 0;
    uint neighbourBlockIdx = blockIdx + 2 * isLeft - 1;
    uint neighbourBlockStart = neighbourBlockIdx << sorted_k_pow;
    int start = neighbourBlockStart - 1;
    int end = min(neighbourBlockStart + sorted_k, n);
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
    uint blockPairOffset = (blockIdx - !isLeft) << sorted_k_pow;
    uint writeIdx = blockPairOffset + (globalIdx - blockStart) + max(end - (int)neighbourBlockStart, 0);
    output_data[writeIdx] = val;
}