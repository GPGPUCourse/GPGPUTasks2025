#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    uint n,
    uint pow2)
{
    const uint global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    uint block_idx = (global_id + 1) >> pow2;

    if ((block_idx & 1) != 0) {
        uint parent_idx = block_idx - 1;
        prefix_sum_accum[global_id] += pow2_sum[parent_idx];
    }
}