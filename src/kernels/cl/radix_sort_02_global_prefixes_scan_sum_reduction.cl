#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* local_histograms,
    __global uint* global_offsets,
    uint num_groups)
{
    uint local_id = get_local_id(0);
    uint digit = local_id;

    __local uint scan_temp[256];
    __local uint digit_sums[256];

    if (digit < 256) {
        uint sum = 0;
        for (uint g = 0; g < num_groups; g++) {
            sum += local_histograms[g * 256 + digit];
        }
        digit_sums[digit] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (digit < 256) {
        scan_temp[digit] = digit_sums[digit];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < 256; stride *= 2) {
        uint val = 0;
        if (digit >= stride) {
            val = scan_temp[digit - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (digit >= stride) {
            scan_temp[digit] += val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (digit < 256) {
        global_offsets[digit] = (digit == 0) ? 0 : scan_temp[digit - 1];
    }
}
