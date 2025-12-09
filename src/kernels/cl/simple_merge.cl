#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void simple_merge(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int sortedK2 = sorted_k * 2;
    const int block = i / sortedK2;
    const int iInBlock = i % sortedK2;
    const uint val = input_data[i];

    int beg[2];
    beg[0] = block * sortedK2;
    beg[1] = beg[0] + sorted_k;

    int isB = ((iInBlock < sorted_k) ? 0 : 1);
    int l = beg[1 - isB] - 1, r = beg[1 - isB] + sorted_k;
    while (r - l > 1) {
        int md = (r + l) / 2;
        uint otherVal;
        if (md < n) {
            otherVal = input_data[md];
        } else  {
            otherVal = UINT_MAX;
        }
        if (otherVal < val || (otherVal == val && isB)) {
            l = md;
        } else {
            r = md;
        }
    }

    // printf("i: %d  block: %d  sorted_k %d  val: %d\nisB: %d\nbeg: %d %d  r: %d  idx: %d\n\n",
    //     i, block, sorted_k, val,
    //     isB,
    //     beg[0], beg[1], r, r - beg[1 - isB] + 1 + i % sorted_k + beg[0]);

    output_data[r - beg[1 - isB] + i % sorted_k + beg[0]] = val;
}
