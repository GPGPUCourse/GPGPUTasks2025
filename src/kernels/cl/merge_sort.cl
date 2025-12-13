#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/morton_code_gpu_shared.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* triIndexes,
    __global const MortonCode* mortonCodes,
    __global uint* outputTriIndexes,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int sortedK2 = sorted_k * 2;
    const int block = i / sortedK2;
    const int iInBlock = i % sortedK2;
    const uint triIdx = triIndexes[i];
    const MortonCode val = mortonCodes[triIdx];

    int beg[2];
    beg[0] = block * sortedK2;
    beg[1] = beg[0] + sorted_k;

    int isB = ((iInBlock < sorted_k) ? 0 : 1);
    int l = beg[1 - isB] - 1, r = beg[1 - isB] + sorted_k;
    while (r - l > 1) {
        int md = (r + l) / 2;
        MortonCode otherVal;
        if (md < n) {
            otherVal = mortonCodes[input_data[md]];
        } else {
            otherVal = UINT_MAX;
        }
        if (otherVal < val || (otherVal == val && isB)) {
            l = md;
        } else {
            r = md;
        }
    }

    output_data[r - beg[1 - isB] + i % sorted_k + beg[0]] = triIdx;
}