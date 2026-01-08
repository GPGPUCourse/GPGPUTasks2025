#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    uint level_pow2,
    uint total_count,
    __global const uint* partial_sums,
    __global       uint* prefix_out)
{
    const uint gid = get_global_id(0);
    if (gid >= total_count)
        return;

    const uint block = (gid + 1) >> level_pow2;
    if (block & 1u) {
        const uint parent = block - 1;
        prefix_out[gid] += partial_sums[parent];
    }
}
