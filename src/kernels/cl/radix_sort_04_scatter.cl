#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global       uint* output,
    __global const uint* zero_offsets,
    __global const uint* one_offsets,
    __global const uint* zero_prefix_inclusive,
    unsigned int groups,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    uint total_zero = zero_prefix_inclusive[groups - 1];

    uint value = 0;
    uint flag = 0;
    if (gid < n) {
        value = input[gid];
        flag = (value >> bit) & 1u;
    }

    __local uint zero_scan[GROUP_SIZE];
    __local uint one_scan[GROUP_SIZE];
    zero_scan[lid] = (gid < n && flag == 0) ? 1u : 0u;
    one_scan[lid] = (gid < n && flag == 1) ? 1u : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        uint add_zero = (lid >= stride) ? zero_scan[lid - stride] : 0u;
        uint add_one = (lid >= stride) ? one_scan[lid - stride] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        zero_scan[lid] += add_zero;
        one_scan[lid] += add_one;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        uint zero_prefix = zero_scan[lid] - ((flag == 0) ? 1u : 0u);
        uint one_prefix = one_scan[lid] - ((flag == 1) ? 1u : 0u);
        if (flag == 0) {
            output[zero_offsets[group] + zero_prefix] = value;
        } else {
            output[one_offsets[group] + one_prefix] = value;
        }
    }
}
