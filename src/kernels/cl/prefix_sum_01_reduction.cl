#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* data,
    __global uint* prefixes,
    __global uint* blocks,
    unsigned int n)
{
    const unsigned int g_ind = get_global_id(0);
    const unsigned int l_ind = get_local_id(0);

    __local uint buf[GROUP_SIZE];

    if (g_ind < n)
        buf[l_ind] = data[g_ind];
    else
        buf[l_ind] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    prefixes[g_ind] = 0;

    for (uint pow2 = 1; pow2 < GROUP_SIZE; pow2 *= 2) {
        uint i = (l_ind + 1) * pow2 * 2 - 1;
        if (i < GROUP_SIZE)
            buf[i] += buf[i - pow2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (l_ind == 0)
        buf[GROUP_SIZE - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pow2 = GROUP_SIZE / 2; pow2 > 0; pow2 /= 2) {
        uint i = (l_ind + 1) * pow2 * 2 - 1;
        if (i < GROUP_SIZE) {
            uint temp_val = buf[i - pow2];
            buf[i - pow2] = buf[i];
            buf[i] += temp_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_ind < n) {
        prefixes[g_ind] = buf[l_ind] + data[g_ind];
        // printf("pref_sum %d=%d d=%d\n", g_ind, prefixes[g_ind], data[g_ind]);
    }

    if (l_ind == GROUP_SIZE - 1 && g_ind < n) {
        // printf("block %d %d %d\n", g_ind / GROUP_SIZE, prefixes[g_ind], g_ind);
        blocks[g_ind / GROUP_SIZE] = prefixes[g_ind];
    }
}
