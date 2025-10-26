#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in,
    __global const uint* prefix_ones_per_group,
    __global       uint* out,
    const uint n,
    const uint bit_pos,
    const uint num_groups)
{
    const uint gid  = get_global_id(0);
    const uint lid  = get_local_id(0);
    const uint wgid = get_group_id(0);

    __local uint lpred[GROUP_SIZE];
    __local uint lscan[GROUP_SIZE];

    uint v = 0u, p = 0u;
    if (gid < n) {
        v = in[gid];
        p = (v >> bit_pos) & 1u;
    }
    lpred[lid] = p;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint acc = 0u;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            acc += lpred[i];
            lscan[i] = acc;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        const uint ones_before_group = (wgid == 0) ? 0u : prefix_ones_per_group[wgid - 1];
        const uint total_ones = prefix_ones_per_group[num_groups - 1];
        const uint zeros_total = n - total_ones;
        const uint P_global = ones_before_group + lscan[lid];
        uint pos;
        if (p == 0u)
            pos = gid - P_global;
        else
            pos = zeros_total + (P_global - 1u);
        out[pos] = v;
    }
}
