#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* arr,
    __global       uint* zero_positions,
    __global       uint* one_positions,
    unsigned int bit,
    unsigned int n)
{
    int i = get_global_id(0);
    if (i > n) {
        return;
    }
    int val = (arr[i] >> bit) & 1;
    one_positions[i] = val;
    zero_positions[i] = 1 - val;
}
