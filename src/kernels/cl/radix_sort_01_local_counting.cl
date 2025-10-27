#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* arr,
    __global uint* bit_flags,
    unsigned int bit_shift,
    unsigned int bit_mask,
    unsigned int n)
{
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int val = (arr[i] >> bit_shift) & bit_mask;
    bit_flags[i + val * n] = 1;
}
