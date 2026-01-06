#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* current_sum_buffer,
    __global       uint* next_sum_buffer,
    unsigned int n)
{
    const uint i = get_global_id(0);
    if (2 * i < n) {
        next_sum_buffer[i] = current_sum_buffer[2 * i] + (2 * i + 1 < n ? current_sum_buffer[2 * i + 1] : 0);
    }
}
