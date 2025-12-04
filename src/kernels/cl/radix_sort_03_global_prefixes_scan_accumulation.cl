#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* partial_sums,
    __global uint* target_accum,
    uint total_count,
    uint step_pow2)
{
    const uint gid = get_global_id(0);

    if (gid >= total_count) {
        return;
    }

    if (step_pow2 == 0) {
        uint current_val = partial_sums[gid];
        uint extra = (gid & 1) ? partial_sums[gid - 1] : 0;
        target_accum[gid] = current_val + extra;
    } else {
        uint bit_mask = 1 << step_pow2;
        if ((gid & bit_mask) != 0) {
            uint source_idx = (gid >> (step_pow2 + 1)) << 1;
            target_accum[gid] += partial_sums[source_idx];
        }
    }
}
