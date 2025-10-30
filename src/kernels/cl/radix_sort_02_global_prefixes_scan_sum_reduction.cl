#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts_local,
    __global uint* group_offsets_exc_sum,
    __global uint* counts_reduced,
    const uint num_groups)
{
    const uint d = get_global_id(0);
    if (d >= RADIX) return;

    uint run_sum = 0u;
    for (uint g = 0; g < num_groups; ++g) {
        uint count = counts_local[g * RADIX + d];
        group_offsets_exc_sum[g * RADIX + d] = run_sum;
        run_sum += count;
    }
    counts_reduced[d] = run_sum;
}

