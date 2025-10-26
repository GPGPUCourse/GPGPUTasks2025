#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* in,
    __global       uint* ones_per_group,
    const uint n,
    const uint bit_pos)
{
    const uint gid  = get_global_id(0);
    const uint lid  = get_local_id(0);
    const uint wgid = get_group_id(0);

    __local uint lflag[GROUP_SIZE];

    uint p = 0u;
    if (gid < n) {
        const uint v = in[gid];
        p = (v >> bit_pos) & 1u;
    }
    lflag[lid] = p;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint sum = 0u;
        for (uint i = 0; i < GROUP_SIZE; ++i) sum += lflag[i];
        ones_per_group[wgid] = sum;
    }
}
