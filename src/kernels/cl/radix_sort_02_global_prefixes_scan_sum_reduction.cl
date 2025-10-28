#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts_local,
    __global uint* counts_reduced,
    const uint num_groups)
{
    const uint d = get_global_id(0);
    if (d >= RADIX) return;

    uint run_sum = 0;
    for (uint g = 0; g < num_groups; ++g) {
        const uint idx = g * RADIX + d;
        uint c = counts_local[idx];
        ((__global uint*)counts_local)[idx] = run_sum;
        run_sum += c;
    }
    counts_reduced[d] = run_sum;
}
