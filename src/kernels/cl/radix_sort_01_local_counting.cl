#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* count_zeros,
    unsigned int buf_size,
    unsigned int bit_number)
{
    uint idx = get_global_id(0);

    if (idx >= buf_size) {
        return;
    }

    if (((input[idx] >> bit_number) & 1) == 0) {
        count_zeros[idx] = 1;
        return;
    }

    count_zeros[idx] = 0;
}
