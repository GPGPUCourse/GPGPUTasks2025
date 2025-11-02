#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k_pow,
                   uint  n)
{
    const unsigned int globalIdx = get_global_id(0);
    if (globalIdx >= n) {
        return;
    }
    int sorted_k = 1 << sorted_k_pow;
    uint val = input_data[globalIdx];
    uint blockIdx = globalIdx >> sorted_k_pow;
    uint blockStart = blockIdx << sorted_k_pow;
    bool isLeft = (blockIdx & 1) == 0;
    uint neighbourBlockIdx = blockIdx + 2 * isLeft - 1;
    uint neighbourBlockStart = neighbourBlockIdx << sorted_k_pow;
    int start = neighbourBlockStart - 1;
    int end = min(neighbourBlockStart + sorted_k, n);
    while (end - start > 1) {
        uint mid = (start + end) / 2;
        uint midVal = input_data[mid];
        bool moveStart = false;
        if (isLeft) {
            moveStart = midVal < val;
        } else {
            moveStart = midVal <= val;
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
