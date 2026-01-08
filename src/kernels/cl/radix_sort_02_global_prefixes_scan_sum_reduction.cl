#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* partial_in,
    uint partial_size,
    __global       uint* partial_out)
{
    const uint gid = get_global_id(0);
    const uint out_len = (partial_size + 1) >> 1;
    if (gid >= out_len)
        return;

    uint base = gid << 1;
    uint left = partial_in[base];
    uint right = (base + 1 < partial_size) ? partial_in[base + 1] : 0;
    partial_out[gid] = left + right;
}
