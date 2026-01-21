#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* local_histograms,
    __global const uint* global_offsets,
    __global uint* local_offsets,
    uint num_groups)
{
    uint digit = get_local_id(0);
    uint wg_id = get_group_id(0);

    if (wg_id != 0) return;

    uint global_base = global_offsets[digit];
    uint running_sum = 0;

    for (uint g = 0; g < num_groups; g++) {
        local_offsets[g * 256 + digit] = global_base + running_sum;
        running_sum += local_histograms[g * 256 + digit];
    }
}
