#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_01_reduction(__global uint* buffer1, __global uint* buffer2, __global uint* prefix,
    unsigned int k,
    unsigned int n)
{
    uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    
    uint first = index * 2;
    uint second = index * 2 + 1;
    uint res = (first < n >> k) ? buffer1[first] : 0;
    res += (second < n >> k) ? buffer1[second] : 0;
    buffer2[index] = res;

    uint group_num = (index + 1) >> k;
    if (group_num % 2 == 1) {
        prefix[index] = (k != 0) * prefix[index] + buffer1[group_num - 1];
    }
}
